"""The agent's command runner (mini-swe-agent's bash action), in a module of its own so that the judge's case workers
can import it cheaply: every command runs in a case worker (judge._in_proc), since a Popen with a preexec_fn
in the rollout process forks 13.5 GB under its mmap lock (judge.py, FCS_CASE_PROC). "seconds" is timed inside the
worker, so a command that waits for a free worker is not charged for the wait.

Output goes to an unlinked temp file, not a pipe, and only the command's own shell is waited for: a background child
that keeps the output open (`setsid sleep 1000 &`, GNU `timeout`, `nohup`) cannot hold the call past MA_AG_TIMEOUT (with
a pipe, communicate() waited for it without a limit; audit 2026-10-24). When the call runs in a case worker, the worker is
a child subreaper for the call's length, so every process the command left behind, in its process group or not, is
killed and reaped before the call returns (FCS_CASE_PROC=0: the command's environment tag finds them instead, best
effort). The harness's own git (run_git) runs under the same sandbox, so repo config an agent wrote (filters,
gpg.program) runs with the agent's rights only."""
from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from . import judge as J

TIMEOUT = int(os.environ.get("MA_AG_TIMEOUT", "60"))
CMD_MEM = 4 << 30            # address space of a command and its children (g++ -O2 of a large file fits)
# Stub `submit` and `post` executables for the long-horizon game. The harness handles both only as the whole action
# (game.SUBMIT_RE); Flint first looked them up with `which submit`, found nothing, and never submitted
# (smoke 16024925: 4 episodes, 0 submits). The stubs make the lookup succeed and say how to call them.
HARNESS_BIN = str(Path(__file__).resolve().parent / "harness_bin")


# ---- the sandbox: the judge's Landlock rules for the compiler, plus no TCP --------------------------------------------
_NET_TCP = (1 << 0) | (1 << 1)  # LANDLOCK_ACCESS_NET_BIND_TCP | CONNECT_TCP (ABI >= 4)
_SCOPE_ABSTRACT_UNIX = 1 << 0   # LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET (ABI 6): no connect to an abstract socket made
# outside the domain, so one command cannot reach a listener that an earlier command or a judged program left behind


def _restrict_net(rules):
    """judge._restrict with TCP handled (and no TCP rule, so no bind or connect) when the kernel has it."""
    if J.LANDLOCK_ABI < 4:
        return J._restrict(rules)
    scope = J._SCOPE_SIGNAL | _SCOPE_ABSTRACT_UNIX if J.LANDLOCK_ABI >= 6 else 0
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


def _sandbox(work: str, harness: bool = False, read: tuple = ()):
    """The command's Landlock rules: the compiler's, plus /dev/urandom (git add and commit read it), for a long-game
    command the stubs and its private /tmp, and `read` read-only (the harness's git between agents' repositories)."""
    rules = J._compile_sandbox(J.Path(work))
    if rules is not None:
        rules = rules + J._landlock_rules([("/dev/urandom", J._FS_READ_FILE)]
                                          + ([(HARNESS_BIN, J._FS_RX), (private_tmp(work), J._HANDLED)] if harness else [])
                                          + [(p, J._FS_RX) for p in read])
    return rules


# ---- the long game's /tmp ---------------------------------------------------------------------------------------------
# Flint writes scratch files to /tmp out of habit (50 of 866 smoke turns failed on it; smoke w153 again). A long-game
# command (harness=True) runs in its own user and mount namespace with private_tmp(work) bound on /tmp: a directory
# beside the repository, so the auto-commits never see it, removed with the work dir (game.rm_work). The
# node's /tmp stays closed, and Landlock then refuses the command any mount of its own.
_MS_BIND, _MS_REC, _MS_PRIVATE = 4096, 1 << 14, 1 << 18


def private_tmp(work: str) -> str:
    return work.rstrip("/") + ".tmp"


def _bind_tmp(tmp: str, uid: int, gid: int) -> None:
    """In the forked child: a new user and mount namespace (same uid and gid) with tmp on /tmp."""
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS)
    for name, text in (("setgroups", "deny"), ("uid_map", f"{uid} {uid} 1"), ("gid_map", f"{gid} {gid} 1")):
        with open(f"/proc/self/{name}", "w") as f:
            f.write(text)
    for src, dst, flags in ((None, "/", _MS_REC | _MS_PRIVATE), (tmp, "/tmp", _MS_BIND)):
        if J._libc.mount(src and src.encode(), dst.encode(), None, ctypes.c_ulong(flags), None) != 0:
            raise OSError(ctypes.get_errno(), f"mount {dst}")


def _preexec(work: str, nproc: int, harness: bool = False):
    tmp = private_tmp(work) if harness else None
    if tmp:
        os.makedirs(tmp, exist_ok=True)   # again on every command: the agent may have removed it
    rules = _sandbox(work, harness)
    uid, gid = os.getuid(), os.getgid()

    def f():
        import resource
        os.setsid()
        if tmp:
            _bind_tmp(tmp, uid, gid)
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 1,) * 2)
        resource.setrlimit(resource.RLIMIT_AS, (CMD_MEM, CMD_MEM))
        resource.setrlimit(resource.RLIMIT_FSIZE, (J.STDOUT_MAX, J.STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        if rules is not None:
            _restrict_net(rules)
    return f


def run_command(command: str, work: str, harness: bool = False) -> dict:
    """One action in the episode's work dir (blocking; run in game.POOL). mini-swe-agent's output dict, plus
    "seconds", the command's wall time (the run time that the reward charges). harness (the long game): HARNESS_BIN
    first on PATH and a private /tmp."""
    return J._in_proc(_timed, command, work, harness, J.CASE_PROC)


def _timed(command: str, work: str, harness: bool = False, reap: bool = False) -> dict:
    t0 = time.monotonic()
    out = _run_command(command, work, harness, reap)
    out["seconds"] = time.monotonic() - t0
    return out


_PR_SET_CHILD_SUBREAPER = 36


def _children() -> list:
    """This process's live and zombie children (/proc/<pid>/task/*/children is not in della's kernel)."""
    me, kids = os.getpid(), []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat", "rb") as f:
                    st = f.read()
                if int(st[st.rindex(b")") + 2:].split()[1]) == me:
                    kids.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return kids


def _kill_orphans() -> None:
    """Kill and reap every child of this process (a case worker that is a subreaper and has finished its call, so all
    its children are the call's leftovers); the children of a killed one are reparented here, hence the loop."""
    for _ in range(100):
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return   # no child left
        kids = _children()
        for k in kids:
            try:
                os.kill(k, 9)
            except ProcessLookupError:
                pass
        for k in kids:
            try:
                os.waitpid(k, 0)
            except ChildProcessError:
                pass
        if not kids:
            time.sleep(0.01)


def _kill_tagged(tag: str) -> None:
    """FCS_CASE_PROC=0 (no worker to be a subreaper): kill this user's processes whose environment holds the call's
    tag. Best effort: a process that cleared its environment is missed."""
    uid, mark = os.getuid(), f"FCS_CMD_TAG={tag}".encode()
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if os.stat(f"/proc/{d}").st_uid != uid:
                    continue
                with open(f"/proc/{d}/environ", "rb") as f:
                    if mark in f.read().split(b"\0"):
                        os.kill(int(d), 9)
            except (OSError, ProcessLookupError):
                pass


def _killpg(pid: int) -> None:
    try:
        os.killpg(pid, 9)
    except (ProcessLookupError, PermissionError):
        pass


def _spawn(argv: list, work: str, env: dict, preexec, timeout: float, reap: bool, merge: bool = True):
    """Run argv in work with its output in unlinked temp files (opened here, outside the sandbox; under TMPDIR, not
    the work dir) and wait for argv's own process only, at most `timeout` s. Then its process group is killed, and with
    reap (a case worker: nothing else runs in it) every process it left behind. Returns (returncode, or None when it
    timed out, stdout bytes, stderr bytes; stderr is in stdout when merge)."""
    reap = reap and J._libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    tag = None if reap else uuid.uuid4().hex
    if tag:
        env = {**env, "FCS_CMD_TAG": tag}
    try:
        with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
            p = subprocess.Popen(argv, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=fo,
                                 stderr=subprocess.STDOUT if merge else fe, preexec_fn=preexec)
            rc = None
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            finally:   # background children die with the command: each action is a fresh subshell
                _killpg(p.pid)
                if rc is None:
                    p.wait()   # SIGKILLed: bounded
                if reap:
                    _kill_orphans()
                else:
                    _kill_tagged(tag)
            fo.seek(0)
            fe.seek(0)
            return rc, fo.read(), fe.read()
    finally:
        if reap:
            J._libc.prctl(_PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)


def _run_command(command: str, work: str, harness: bool = False, reap: bool = False) -> dict:
    env = {"PATH": (HARNESS_BIN + ":" if harness else "") + "/usr/bin:/bin", "LANG": "C", "HOME": work, "TMPDIR": work, "PAGER": "cat", "MANPAGER": "cat",
           "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
    try:
        rc, out, _ = _spawn(["bash", "-c", command], work, env, _preexec(work, J._nproc_limit(), harness), TIMEOUT, reap)
    except Exception as e:
        return {"output": "", "returncode": -1, "exception_info": f"An error occurred while executing the command: {e}"}
    out = out.decode("utf-8", errors="replace")
    if rc is None:
        return {"output": out, "returncode": -1,
                "exception_info": f"An error occurred while executing the command: Command '{command}' timed out "
                                  f"after {TIMEOUT} seconds"}
    return {"output": out, "returncode": rc, "exception_info": ""}


# ---- the harness's git, under the command sandbox ---------------------------------------------------------------------
def run_git(work: str, args: list, env: dict, timeout: float = 60, read: tuple = ()) -> subprocess.CompletedProcess:
    """git <args> in an episode's repo for the harness (repo.git), in a case worker and under the agents'
    sandbox: config the agents wrote into .git (filter.*.clean, gpg.program, core.pager, ...) then runs with their rights
    only, not the trainer's (audit 2026-10-24); `read`: paths it may also read (another agent's repository, team.git).
    Raises subprocess.TimeoutExpired after `timeout` s."""
    rc, out, err = J._in_proc(_git_proc, work, list(args), env, timeout, J.CASE_PROC, tuple(read))
    if rc is None:
        raise subprocess.TimeoutExpired(["git", *args[:1]], timeout)
    return subprocess.CompletedProcess(["git", *args], rc, out, err)


# RLIMIT_NPROC counts every task of the user on the node, and J._nproc_limit() is that count as of up to 5 s ago plus
# 128. When the user's other work on the node (other jobs, the judge, the agents' commands) starts more than 128 tasks
# inside that window, git cannot fork (git fetch forks upload-pack) and exits 128: own-repos fetches and publishes
# failed so for a round or two in smokes w260 and w412 (2026-10-26; reproduced by a stale cached count). The harness's
# git gets this much more room; a fork bomb planted in .git config is still bounded.
GIT_NPROC_MARGIN = 4096


def _git_proc(work: str, args: list, env: dict, timeout: float, reap: bool = False, read: tuple = ()):
    rules = _sandbox(work, read=read)
    nproc = J._nproc_limit() + GIT_NPROC_MARGIN

    def pre():
        import resource
        os.setsid()
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_FSIZE, (J.STDOUT_MAX, J.STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if rules is not None:
            _restrict_net(rules)
    rc, out, err = _spawn(["git", *args], work, env, pre, timeout, reap, merge=False)
    # surrogateescape: a file name that is not UTF-8 goes back to git as the same bytes (strict text=True raised)
    return rc, out.decode("utf-8", "surrogateescape"), err.decode("utf-8", "replace")
