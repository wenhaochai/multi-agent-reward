"""The agent's command runner (mini-swe-agent's bash action), apart from fcs_agent so that the judge's case workers
can import it without miles: every command runs in a case worker (fcs_judge._in_proc), since a Popen with a preexec_fn
in the rollout process forks 13.5 GB under its mmap lock (fcs_judge, FCS_CASE_PROC). "seconds" is timed inside the
worker, so a command that waits for a free worker is not charged for the wait."""
from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import time
from pathlib import Path

from . import fcs_judge as J

TIMEOUT = int(os.environ.get("MA_AG_TIMEOUT", "60"))
CMD_MEM = 4 << 30            # address space of a command and its children (g++ -O2 of a large file fits)
# Stub `submit` and `post` executables for the long-horizon game. The harness handles both only as the whole action
# (fcs_agent_long.SUBMIT_RE); Flint first looked them up with `which submit`, found nothing, and never submitted
# (smoke 16024925: 4 episodes, 0 submits). The stubs make the lookup succeed and say how to call them.
HARNESS_BIN = str(Path(__file__).resolve().parent / "harness_bin")


# ---- the sandbox: the judge's Landlock rules for the compiler, plus no TCP --------------------------------------------
_NET_TCP = (1 << 0) | (1 << 1)  # LANDLOCK_ACCESS_NET_BIND_TCP | CONNECT_TCP (ABI >= 4)


def _restrict_net(rules):
    """fcs_judge._restrict with TCP handled (and no TCP rule, so no bind or connect) when the kernel has it."""
    if J.LANDLOCK_ABI < 4:
        return J._restrict(rules)
    scope = J._SCOPE_SIGNAL if J.LANDLOCK_ABI >= 6 else 0
    attr = ctypes.create_string_buffer(struct.pack("<QQQ", J._HANDLED, _NET_TCP, scope) if J.LANDLOCK_ABI >= 6
                                       else struct.pack("<QQ", J._HANDLED, _NET_TCP))
    rs = J._libc.syscall(J._LL_CREATE, attr, ctypes.c_size_t(len(attr.raw)), ctypes.c_uint32(0))
    if rs < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, access in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = ctypes.create_string_buffer(struct.pack("<Qi", access, fd))
                if J._libc.syscall(J._LL_ADD, ctypes.c_int(rs), ctypes.c_int(1), rule, ctypes.c_uint32(0)) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(fd)
        if J._libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
        if J._libc.syscall(J._LL_RESTRICT, ctypes.c_int(rs), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(rs)


def _preexec(work: str, nproc: int, harness: bool = False):
    rules = J._compile_sandbox(J.Path(work))
    if rules is not None and harness:
        rules = rules + J._landlock_rules([(HARNESS_BIN, J._FS_RX)])

    def f():
        import resource
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 1,) * 2)
        resource.setrlimit(resource.RLIMIT_AS, (CMD_MEM, CMD_MEM))
        resource.setrlimit(resource.RLIMIT_FSIZE, (J.STDOUT_MAX, J.STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        if rules is not None:
            _restrict_net(rules)
    return f


def run_command(command: str, work: str, harness: bool = False) -> dict:
    """One action in the episode's work dir (blocking; run in fcs_agent's _POOL). mini-swe-agent's output dict, plus
    "seconds", the command's wall time (the run time that the reward charges). harness: HARNESS_BIN first on PATH."""
    return J._in_proc(_timed, command, work, harness)


def _timed(command: str, work: str, harness: bool = False) -> dict:
    t0 = time.monotonic()
    out = _run_command(command, work, harness)
    out["seconds"] = time.monotonic() - t0
    return out


def _run_command(command: str, work: str, harness: bool = False) -> dict:
    env = {"PATH": (HARNESS_BIN + ":" if harness else "") + "/usr/bin:/bin", "LANG": "C", "HOME": work, "TMPDIR": work, "PAGER": "cat", "MANPAGER": "cat",
           "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
    try:
        p = subprocess.Popen(["bash", "-c", command], cwd=work, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             preexec_fn=_preexec(work, J._nproc_limit(), harness))
    except Exception as e:
        return {"output": "", "returncode": -1, "exception_info": f"An error occurred while executing the command: {e}"}
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
        return {"output": out.decode("utf-8", errors="replace"), "returncode": p.returncode, "exception_info": ""}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        return {"output": (out or b"").decode("utf-8", errors="replace"), "returncode": -1,
                "exception_info": f"An error occurred while executing the command: Command '{command}' timed out "
                                  f"after {TIMEOUT} seconds"}
    finally:
        try:  # background children of the command die with it: each action is a fresh subshell
            os.killpg(p.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
