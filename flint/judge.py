"""Local Frontier-CS v1.1 algorithmic judge (no Docker, no judge server): the scoring of the official go-judge engine
(Frontier-CS-v1.1/algorithmic/judge/src/judge_engine.js, problem_manager.js, utils.js at aec2aff0) re-implemented with
subprocesses and rlimits, so RL rewards on Della match what the judge server would return.

Rules copied from the official engine:
  * cases: n_cases per subtask from config.yaml (prefix/suffix default '' / '.in' / '.ans', numbered from 1), plus every
    other complete <n>.in + <n>.ans (or .out) pair in testdata/ at the last subtask's limits;
  * compile: g++ -O2 -pipe -std=gnu++17 (we add -static -s); a compile error scores 0;
  * run: CPU time <= the case time limit, wall <= 2x, address space and stack <= the memory limit, stdout <= 128 MiB;
    anything else (TLE, MLE, runtime error, non-zero exit) scores the case 0;
  * classic: checker `chk in.txt out.txt ans.txt` (10 s CPU, 256 MiB); interactive: the program and
    `interactor in.txt tout.txt ans.txt` (4x time and memory) joined by two pipes, scored from the interactor's stderr;
  * a checker / interactor message is trusted only when it exits 0 (testlib _ok) or 7 (testlib points); the case ratio
    is the engine's applyScoreRatio: the first match of /Ratio: ([\\d.]+)/ in a trusted message, read like JS parseFloat
    and not clamped; with no `Ratio:` label, 1 when the helper exited 0 and 0 otherwise (owner 2026-10-28: "用v1.1的
    以后都用v1.1"; the first match is the helper's own: no v1.1 helper prints contestant text before its Ratio, while
    247 and 243 print a contestant token after it, audit 2026-10-28);
  * problem score = 100 x mean case ratio.
Deviations (owner 2026-10-28: "如果v1.1错了那我们就应该修改啊": where the engine is wrong, this judge is not):
  * a trusted message with a `Ratio:` label whose value the engine cannot read (a negative such as %.12f of -1e-15,
    nan, a value that overflows to inf) scores the case 0; the engine falls back to 1 when the helper exited 0, so a
    checker's -0.000000000000 would be a full score. No v1.1 helper prints such a value today (audit 2026-10-28), and
    contestant text can only lower its own score this way, never raise it.
From the 2026-10-04 audit:
  * checkers and interactors get a 2 GiB address space (official: none; 256 MiB broke val 23's checker);
  * interactive programs: the official engine sets no address-space limit, only a memory limit, so the program gets a
    max(4 x ML, 1 GiB) address-space cap (node safety) and an MLE when its peak RSS exceeds ML;
  * untrusted code is sandboxed with Landlock (FCS_SANDBOX=0 disables): g++ reads only system dirs and its work dir
    (no #include of test answers), the program opens only its binary, /dev/null, /dev/{u,}random and /proc/self, with
    an empty environment;
  * infrastructure failures (an exception in a case, g++ killed by a signal or out of disk / processes) are flagged
    ("infra": True) instead of passing as plain zeros;
  * a program may create at most ~128 more tasks (the official procLimit): RLIMIT_NPROC counts all of the user's tasks,
    so the limit is set to the user's current task count + 128 (counted at most every 5 s), which stops a fork bomb
    from starving the training job and leaves single-threaded programs untouched;
  * Landlock ABI >= 6 also scopes signals: a program cannot signal anything outside its sandbox (kill(getppid()) or
    kill(-1) reached the trainer before; audit 2026-10-11);
  * g++ gets a 2 GiB address space and the task limit (official: 512 MiB, 10 s CPU, 50 processes).
usage: judge(problem_dir, cpp_source) -> {"score": 0-100, "status": ..., "cases": [...], "infra": bool}; the game
calls judge_code (the end of this file), which runs judge in a thread under a node-wide concurrency limit.
"""
from __future__ import annotations

import asyncio
import ctypes
import fcntl
import functools
import hashlib
import math
import os
import pickle
import queue
import re
import resource
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

FCS_ROOT = Path(os.environ.get("FCS_ROOT", "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic"))
TESTLIB_DIR = FCS_ROOT / "judge" / "include"   # testlib.h for the checkers and interactors
# work dirs (each case's stdout, up to 128 MiB) and the helper cache follow TMPDIR (a job's per-job scratch dir, removed by
# its EXIT trap) when FCS_JUDGE_TMP / FCS_JUDGE_CACHE are unset: on shared node /tmp a full disk cut outputs short and
# scored them WA, silently (2026-10-08), and a killed job left its fcsj_* dirs behind. Without TMPDIR: /tmp as before
# (not tempfile.gettempdir(): it also reads TEMP and TMP, and falls back to the current directory)
TMP = Path(os.environ.get("FCS_JUDGE_TMP") or os.environ.get("TMPDIR") or "/tmp")
CACHE = Path(os.environ.get("FCS_JUDGE_CACHE") or TMP / "fcs_judge_cache")
CASE_WORKERS = int(os.environ.get("FCS_CASE_WORKERS", "8"))


def job_cpus() -> int:
    """The cores this process may use: the CPU affinity (a Slurm job's cpuset) and every cgroup-v2 cpu.max quota from
    this process's cgroup up to the root (a vis node limits a user to 7 cores by quota while affinity shows 80).
    FCS_JOB_CPUS overrides. Inside apptainer the quota is visible only with --bind /sys/fs/cgroup."""
    if os.environ.get("FCS_JOB_CPUS"):
        return int(os.environ["FCS_JOB_CPUS"])
    n = len(os.sched_getaffinity(0))
    try:
        path = next(l.split("::", 1)[1].strip() for l in open("/proc/self/cgroup") if l.startswith("0::"))
        while True:
            f = Path("/sys/fs/cgroup") / path.lstrip("/") / "cpu.max"
            if f.exists():
                q, per = f.read_text().split()[:2]
                if q != "max":
                    n = min(n, max(1, -(-int(q) // int(per))))
            if path in ("/", ""):
                break
            path = str(Path(path).parent)
    except (OSError, StopIteration, ValueError):
        pass
    return n
COMPILE = ["g++", "-O2", "-pipe", "-static", "-s", "-std=gnu++17"]
STDOUT_MAX = 128 << 20
OK_EXIT, POINTS_EXIT = 0, 7
_RATIO = re.compile(r"Ratio: ([\d.]+)")  # the official engine's regex
_JS_FLOAT = re.compile(r"\d+(?:\.\d*)?|\.\d+")  # what parseFloat reads from a run of digits and dots
_INFRA_MARKERS = ("No space left on device", "Resource temporarily unavailable", "Cannot allocate memory",
                  "cannot fork", "virtual memory exhausted", "Too many open files")
SANDBOX = os.environ.get("FCS_SANDBOX", "1") == "1"


def _ratio(msg: str, trusted: bool, ok: bool) -> float:
    """A case's ratio from a checker / interactor message (trusted = it exited 0 or 7; ok = it exited 0), as the
    engine's parseRatio and applyScoreRatio give it: JS parseFloat of the first [\\d.]+ capture, i.e. its longest decimal
    prefix ("0.000000." of val 154's interactor is 0: audit 2026-10-07), kept when finite. A trusted message whose
    `Ratio:` label gives no such value (a negative, nan, 300 digits that overflow) scores 0, not the engine's fallback
    (the header's deviation); without the label, the fallback: accepted or not."""
    m = _RATIO.search(msg) if trusted else None
    v = _JS_FLOAT.match(m.group(1)) if m else None
    x = float(v.group(0)) if v else None
    if x is not None and math.isfinite(x):
        return x
    if trusted and "Ratio:" in msg:
        return 0.0
    return 1.0 if ok else 0.0


def to_seconds(s) -> float:
    if isinstance(s, (int, float)):
        return float(s) / 1e9  # a bare number is nanoseconds in the engine
    m = re.match(r"^([\d.]+)\s*(ms|s)?$", str(s).strip(), re.I)
    v = float(m.group(1)) if m else 0.0
    return v / 1000 if m and (m.group(2) or "s").lower() == "ms" else v


def to_bytes(s) -> int:
    if isinstance(s, (int, float)):
        return int(s)
    m = re.match(r"^([\d.]+)\s*([kmg]?)$", str(s).strip(), re.I)
    v = float(m.group(1)) if m else 0.0
    mul = {"g": 1 << 30, "m": 1 << 20, "k": 1 << 10}.get((m.group(2) if m else "").lower(), 1)
    return int(round(v * mul))


def load_problem(pdir: Path) -> dict:
    cfg = yaml.safe_load(open(pdir / "config.yaml")) or {}
    gtime, gmem = cfg.get("time_limit") or cfg.get("time") or "1s", cfg.get("memory_limit") or cfg.get("memory") or "256m"
    ipre, opre = cfg.get("input_prefix", ""), cfg.get("output_prefix", "")
    isuf, osuf = cfg.get("input_suffix", ".in"), cfg.get("output_suffix", ".ans")
    cases, cur = [], 1
    subtasks = cfg.get("subtasks") or []
    for st in subtasks:
        t, mem = st.get("time_limit") or st.get("time") or gtime, st.get("memory_limit") or st.get("memory") or gmem
        if st.get("n_cases") is not None:
            for i in range(int(st["n_cases"])):
                cases.append((f"{ipre}{cur + i}{isuf}", f"{opre}{cur + i}{osuf}", t, mem))
            cur += int(st["n_cases"])
        else:
            for c in st.get("cases", []):
                cases.append((c["input"], c["output"], c.get("time") or t, c.get("memory") or mem))
    fb = subtasks[-1] if subtasks else {}
    ft, fm = fb.get("time_limit") or fb.get("time") or gtime, fb.get("memory_limit") or fb.get("memory") or gmem
    have = {c[0] for c in cases}
    td = pdir / "testdata"
    for f in sorted(td.glob("*.in"), key=lambda p: (len(p.stem), p.stem)):
        if f.name in have:
            continue
        for ext in (".ans", ".out"):
            if (td / (f.stem + ext)).exists():
                cases.append((f.name, f.stem + ext, ft, fm))
                break
    interactive = str(cfg.get("type", "default")).lower() == "interactive"
    # a case's output files are named after its input and deleted when it is scored: names must not repeat
    assert len({c[0] for c in cases}) == len(cases), f"{pdir}: repeated test input names"
    return {"dir": pdir, "cases": cases, "interactive": interactive,
            "checker": cfg.get("checker", "chk.cc"), "interactor": cfg.get("interactor", "interactor.cc")}


def _compile_helper(pdir: Path, src_name: str) -> Path:
    """Compile a problem's checker or interactor once per machine (file lock; keyed by source hash)."""
    src = pdir / src_name
    h = hashlib.sha1(src.read_bytes()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    exe = CACHE / f"{pdir.name}_{Path(src_name).stem}_{h}"
    if exe.exists():
        return exe
    with open(str(exe) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not exe.exists():
            tmp = str(exe) + f".tmp{os.getpid()}"
            r = subprocess.run(["g++", "-O2", "-std=gnu++17", f"-I{TESTLIB_DIR}", f"-I{pdir}", str(src), "-o", tmp],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"helper compile failed for {src}: {r.stderr[-500:]}")
            os.replace(tmp, exe)
    return exe


# checkers and interactors are trusted problem code: official go-judge gives them no address-space limit; RLIMIT_AS
# 256 MiB broke val problem 23 (its checker has 214 MiB of static data: bad_alloc on 16 of 22 cases, every program
# capped at 27.27; audit 2026-10-04). 2 GiB only guards the node.
TRUSTED_AS = 2 << 30


# ---- Landlock (Linux >= 5.13, unprivileged): file-access sandbox for untrusted code ----
_LL_CREATE, _LL_ADD, _LL_RESTRICT = 444, 445, 446  # x86_64 syscall numbers
_FS_EXECUTE, _FS_WRITE_FILE, _FS_READ_FILE, _FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_FS_ALL_V1 = (1 << 13) - 1          # every ABI-1 filesystem right (execute ... make_sym)
_FS_FILE = _FS_EXECUTE | _FS_WRITE_FILE | _FS_READ_FILE
_FS_RX = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR
_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _landlock_abi() -> int:
    try:
        v = _libc.syscall(_LL_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))  # LANDLOCK_CREATE_RULESET_VERSION
        return int(v) if v > 0 else 0
    except Exception:
        return 0


LANDLOCK_ABI = _landlock_abi() if SANDBOX else 0
if SANDBOX and not LANDLOCK_ABI:
    import sys
    print("[fcs_judge] WARNING: Landlock is unavailable here: untrusted code runs without a file sandbox",
          file=sys.stderr, flush=True)
_HANDLED = {1: _FS_ALL_V1, 2: (1 << 14) - 1}.get(LANDLOCK_ABI, (1 << 15) - 1)  # ABI 2 adds REFER, 3 TRUNCATE
_SCOPE_SIGNAL = 1 << 1  # LANDLOCK_SCOPE_SIGNAL (ABI 6): no signals to processes outside the domain
if SANDBOX and 0 < LANDLOCK_ABI < 6:
    import sys
    print(f"[fcs_judge] WARNING: Landlock ABI {LANDLOCK_ABI} < 6: untrusted code can signal the trainer",
          file=sys.stderr, flush=True)


def _landlock_rules(rules):
    """Pre-resolved [(path, access)] for _limits: only paths that exist; a file rule keeps only file rights."""
    out = []
    for path, access in rules:
        if os.path.exists(path):
            out.append((path, access if os.path.isdir(path) else access & _FS_FILE))
    return out


def _restrict(rules):
    """In the child (preexec): allow only `rules` [(path, access)], deny every other handled file access."""
    if LANDLOCK_ABI >= 6:   # struct landlock_ruleset_attr {handled_access_fs, handled_access_net, scoped}
        attr = ctypes.create_string_buffer(struct.pack("<QQQ", _HANDLED, 0, _SCOPE_SIGNAL))
    else:
        attr = ctypes.create_string_buffer(struct.pack("<Q", _HANDLED))
    rs = _libc.syscall(_LL_CREATE, attr, ctypes.c_size_t(len(attr.raw)), ctypes.c_uint32(0))
    if rs < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, access in rules:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = ctypes.create_string_buffer(struct.pack("<Qi", access, fd))
                if _libc.syscall(_LL_ADD, ctypes.c_int(rs), ctypes.c_int(1), rule, ctypes.c_uint32(0)) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(fd)
        if _libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
        if _libc.syscall(_LL_RESTRICT, ctypes.c_int(rs), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(rs)


def _limits(cpu_s: float, mem_b: int | None, stack_b: int | None = None, sandbox=None, nproc: int | None = None):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(cpu_s)) + 1,) * 2)
        if mem_b:
            resource.setrlimit(resource.RLIMIT_AS, (mem_b, mem_b))
            s = stack_b or mem_b
            resource.setrlimit(resource.RLIMIT_STACK, (s, s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (STDOUT_MAX, STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if nproc:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        if sandbox is not None:
            _restrict(sandbox)
    return f


def _program_sandbox(sol: Path):
    """What a contestant program may open: its own binary, /dev/null, /dev/(u)random, /proc/self (read)."""
    if not LANDLOCK_ABI:
        return None
    return _landlock_rules([(str(sol), _FS_EXECUTE | _FS_READ_FILE), ("/dev/null", _FS_READ_FILE | _FS_WRITE_FILE),
                            ("/dev/urandom", _FS_READ_FILE), ("/dev/random", _FS_READ_FILE),
                            ("/proc/self", _FS_READ_FILE | _FS_READ_DIR)])


def _compile_sandbox(work: Path):
    """What g++ may touch: system dirs read/execute, /dev/null, its work dir read/write."""
    if not LANDLOCK_ABI:
        return None
    sys_dirs = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/proc"]
    return _landlock_rules([(d, _FS_RX) for d in sys_dirs] + [("/dev/null", _FS_READ_FILE | _FS_WRITE_FILE),
                                                             (str(work), _HANDLED)])


_CLEAN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C"}
NPROC_MARGIN = 128  # the official engine's procLimit
_ucount = [0.0, 0]
_ulock = threading.Lock()


def _nproc_limit() -> int:
    """The user's current task count (threads included, as RLIMIT_NPROC counts them) + NPROC_MARGIN; 5 s cache.
    One thread refreshes it while the others keep the cached count: when every case thread rescanned /proc at each
    expiry (1454 stats each, the GIL handed back after every one), 40-60 of the 88 judge slots sat in this scan and the
    eval's GPUs idled (15523396). Only the first call waits, since a count of 0 would leave a limit of 128 tasks."""
    if time.monotonic() - _ucount[0] > 5 and _ulock.acquire(blocking=_ucount[0] == 0.0):
        try:
            if time.monotonic() - _ucount[0] > 5:
                uid, n = os.getuid(), 0
                for p in os.listdir("/proc"):
                    if p.isdigit():
                        try:
                            if os.stat(f"/proc/{p}").st_uid == uid:
                                n += len(os.listdir(f"/proc/{p}/task"))
                        except OSError:
                            pass
                _ucount[1], _ucount[0] = n, time.monotonic()
        finally:
            _ulock.release()
    return _ucount[1] + NPROC_MARGIN


def _wait(p: subprocess.Popen, wall_s: float, rss: list | None = None):
    """Wait for p with a wall limit; returns (exit_code or None if killed, cpu seconds, timed_out).
    rss, if given, receives the peak resident set in bytes."""
    deadline = time.monotonic() + wall_s
    while True:
        # the clock is read before wait4, so a kill means the program still ran after its deadline. Read after wait4,
        # a thread that stalled between the two (seconds, when the rollout process forked under contention) killed a
        # program that had ended in 20 ms and scored the case 0 (ev4 2026-10-14: programs at 0 in the job score 6.9 to
        # 33 offline)
        late = time.monotonic() > deadline
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            p.returncode = os.waitstatus_to_exitcode(status)
            if rss is not None:
                rss.append(ru.ru_maxrss * 1024)
            return p.returncode, ru.ru_utime + ru.ru_stime, False
        if late:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, ru = os.wait4(p.pid, 0)
            if rss is not None:
                rss.append(ru.ru_maxrss * 1024)
            return None, ru.ru_utime + ru.ru_stime, True
        time.sleep(0.005)


def _case_classic_run(prob, sol: Path, chk: Path, case, work: Path) -> float:
    inp, outp, t, mem = case
    td = prob["dir"] / "testdata"
    tl, ml = to_seconds(t), to_bytes(mem)
    out = work / f"{inp}.out"
    with open(td / inp, "rb") as fi, open(out, "wb") as fo:
        p = subprocess.Popen([str(sol)], stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, cwd=work, env=_CLEAN_ENV,
                             preexec_fn=_limits(tl, ml, sandbox=_program_sandbox(sol), nproc=_nproc_limit()))
        code, cpu, killed = _wait(p, 2 * tl)
    if killed or code != 0 or cpu > tl or out.stat().st_size > STDOUT_MAX:
        return 0.0, f"run: code={code} cpu={cpu:.2f}/{tl} killed={killed}"
    ans = td / outp if (td / outp).exists() else td / outp.replace(".ans", ".out")
    r = subprocess.run([str(chk), str(td / inp), str(out), str(ans)], capture_output=True, text=True,
                       errors="replace", preexec_fn=_limits(10, TRUSTED_AS), timeout=20)
    ok = r.returncode == OK_EXIT
    msg = r.stdout or r.stderr or ""
    ratio = _ratio(msg, r.returncode in (OK_EXIT, POINTS_EXIT), ok)
    return ratio, f"chk={r.returncode} cpu={cpu:.2f}/{tl} {msg.strip()[:120]}"


def _own_hwm() -> int:
    """This process's peak resident set in bytes (VmHWM)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) << 10
    return 0


def _case_interactive_run(prob, sol: Path, inter: Path, case, work: Path) -> float:
    inp, outp, t, mem = case
    td = prob["dir"] / "testdata"
    tl, ml = to_seconds(t), to_bytes(mem)
    ans = td / outp if (td / outp).exists() else td / outp.replace(".ans", ".out")
    s2i_r, s2i_w = os.pipe()
    i2s_r, i2s_w = os.pipe()
    err = open(work / f"{inp}.ierr", "w+b")
    pi = ps = None
    try:
        pi = subprocess.Popen([str(inter), str(td / inp), str(work / f"{inp}.tout"), str(ans)], stdin=s2i_r,
                              stdout=i2s_w, stderr=err, cwd=work, preexec_fn=_limits(4 * tl, max(4 * ml, TRUSTED_AS)))
        # the official engine sets no address-space limit on interactive programs, only a memory limit: cap the
        # address space at max(4 x ML, 1 GiB) for the node's sake and score an MLE from the peak RSS
        ps = subprocess.Popen([str(sol)], stdin=i2s_r, stdout=s2i_w, stderr=subprocess.DEVNULL, cwd=work,
                              env=_CLEAN_ENV, preexec_fn=_limits(tl, max(4 * ml, 1 << 30), stack_b=ml,
                                                                 sandbox=_program_sandbox(sol), nproc=_nproc_limit()))
    except BaseException:
        if pi is not None:  # the interactor started but the program did not: do not leak it
            try:
                os.killpg(pi.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pi.wait()
        err.close()
        raise
    finally:
        for fd in (s2i_r, s2i_w, i2s_r, i2s_w):
            os.close(fd)
    peak = []
    scode, scpu, skilled = _wait(ps, 2 * tl, peak)
    icode, _, ikilled = _wait(pi, 8 * tl)
    err.seek(0)
    imsg = err.read().decode(errors="replace")
    err.close()
    mle = bool(peak) and peak[0] > ml
    if mle and _own_hwm() > ml:
        # ru_maxrss includes what the program's process held before exec, a copy of this one: from the rollout process
        # (13.5 GB) every interactive case read as an MLE and scored 0, in every job since 78994b9 (2026-10-03). A case
        # worker holds about 20 MiB; a caller this large cannot measure the program, so the case is an infra failure
        raise RuntimeError(f"peak RSS unmeasurable: the judge holds {_own_hwm() >> 20} MiB, over the {ml >> 20} MiB limit")
    sub_ok = not skilled and scode == 0 and scpu <= tl and not mle
    inter_scoring = not ikilled and icode in (OK_EXIT, POINTS_EXIT)
    info = (f"sol code={scode} cpu={scpu:.2f}/{tl} rss={peak[0] >> 20 if peak else -1}MiB/{ml >> 20} killed={skilled}; "
            f"inter code={icode} killed={ikilled} {imsg.strip()[:120]}")
    if not sub_ok:
        return 0.0, info                              # the run itself failed: no credit
    ok = icode == OK_EXIT and not ikilled
    return _ratio(imsg if inter_scoring else "", inter_scoring, ok), info


def _case_classic(prob, sol: Path, chk: Path, case, work: Path) -> float:
    """A classic case; its output file is deleted as soon as the case is scored (a program that prints 128 MiB on
    each of 130 cases otherwise holds 16 GiB of /tmp until the whole judge ends: della-vis2, 2026-10-04)."""
    try:
        return _case_classic_run(prob, sol, chk, case, work)
    finally:
        (work / f"{case[0]}.out").unlink(missing_ok=True)


def _case_interactive(prob, sol: Path, inter: Path, case, work: Path) -> float:
    """An interactive case; the interactor's output and stderr files are deleted once the case is scored."""
    try:
        return _case_interactive_run(prob, sol, inter, case, work)
    finally:
        for suffix in (".tout", ".ierr"):
            (work / f"{case[0]}{suffix}").unlink(missing_ok=True)


# FCS_CASE_PROC=1 (the default): every compile and test case runs in a small worker process, not in the caller. A Popen
# with a preexec_fn forks its caller, and fork copies the caller's page tables while it holds the caller's mmap lock. In
# the rollout process (13.5 GB resident, 319 threads) 40-88 of the 88 case threads sat in lock_mm_and_find_vma, page
# faults waiting for that lock, so interactors waited seconds for their program and the team_bonus eval 15523392 judged
# 98 of 680 episodes in 13 h with one request per engine. The case threads of judge() still decide what runs and in
# what order; a worker only runs one call at a time. Workers are plain `python3 -c` children (started with vfork, as they need
# no preexec_fn) that read pickled calls on stdin: multiprocessing would import the caller's __main__ (Ray's worker
# script) in each of them. At most FCS_CASE_PROCS workers (default: the job's cores); a worker that dies is replaced
# and its call retried once.
CASE_PROC = os.environ.get("FCS_CASE_PROC", "1") == "1"
_idle: "queue.LifoQueue" = queue.LifoQueue()
_wlock = threading.Lock()
_made = [0]


class _WorkerDied(Exception):
    pass


class _Worker:
    def __init__(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        self.p = subprocess.Popen([sys.executable, "-c", f"from {__name__} import _serve; _serve()"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def call(self, fn, a):
        data = pickle.dumps((fn, a))
        try:
            self.p.stdin.write(struct.pack("<Q", len(data)) + data)
            self.p.stdin.flush()
            hdr = self.p.stdout.read(8)
            if len(hdr) < 8:
                raise _WorkerDied(f"case worker exited ({self.p.poll()})")
            ok, v = pickle.loads(self.p.stdout.read(struct.unpack("<Q", hdr)[0]))
        except (OSError, EOFError, pickle.UnpicklingError) as e:
            raise _WorkerDied(f"case worker lost: {e}") from e
        if not ok:
            raise v
        return v

    def close(self):
        try:
            self.p.kill()
            self.p.wait()
        except OSError:
            pass


def _serve():
    """A case worker's loop: a call is an 8-byte length and a pickled (fn, args); the reply is (True, value) or
    (False, exception), framed the same way. Its stdout becomes its stderr, so a stray print cannot break the framing."""
    rd, wr = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    while True:
        hdr = rd.read(8)
        if len(hdr) < 8:
            return
        fn, a = pickle.loads(rd.read(struct.unpack("<Q", hdr)[0]))
        try:
            out = pickle.dumps((True, fn(*a)))
        except BaseException as e:
            try:
                out = pickle.dumps((False, e))
            except Exception:
                out = pickle.dumps((False, RuntimeError(repr(e)[:2000])))
        wr.write(struct.pack("<Q", len(out)) + out)
        wr.flush()


def _take() -> _Worker:
    try:
        return _idle.get_nowait()
    except queue.Empty:
        pass
    with _wlock:
        if _made[0] < (int(os.environ.get("FCS_CASE_PROCS", "0")) or job_cpus()):
            _made[0] += 1
            try:
                return _Worker()
            except BaseException:
                _made[0] -= 1
                raise
    return _idle.get()


def _in_proc(fn, *a):
    """fn(*a) in a case worker (FCS_CASE_PROC=0: in this thread). fn must be a module-level function, and fn, its
    arguments and its result must pickle."""
    if not CASE_PROC:
        return fn(*a)
    for attempt in (0, 1):
        w = _take()
        try:
            v = w.call(fn, a)
        except _WorkerDied:
            w.close()
            with _wlock:
                _made[0] -= 1
            if attempt:
                raise
            continue
        except BaseException:
            _idle.put(w)   # fn's own exception: the worker is fine
            raise
        _idle.put(w)
        return v


def _compile(work: Path, sol: Path):
    return subprocess.run(COMPILE + ["-o", str(sol), str(work / "sol.cpp")], capture_output=True, text=True,
                          errors="replace", timeout=120, env={**_CLEAN_ENV, "TMPDIR": str(work)},
                          preexec_fn=_limits(600, 2 << 30, stack_b=1 << 30, sandbox=_compile_sandbox(work),
                                             nproc=_nproc_limit()))


def judge(problem_dir, source: str, case_workers: int = CASE_WORKERS) -> dict:
    prob = load_problem(Path(problem_dir))
    if not prob["cases"]:
        return {"score": 0.0, "status": "no cases", "cases": []}
    work = Path(tempfile.mkdtemp(prefix="fcsj_", dir=TMP))
    try:
        (work / "sol.cpp").write_text(source)
        sol = work / "sol"
        r = _in_proc(_compile, work, sol)
        if r.returncode != 0:
            infra = r.returncode < 0 or any(m in r.stderr for m in _INFRA_MARKERS)
            return {"score": 0.0, "status": "infra error" if infra else "compile error", "cases": [],
                    "msg": r.stderr[:1 << 21], "infra": infra}  # whole compiler output (up to 2 MiB)
        helper = _compile_helper(prob["dir"], prob["interactor"] if prob["interactive"] else prob["checker"])
        fn = _case_interactive if prob["interactive"] else _case_classic
        with ThreadPoolExecutor(max_workers=max(1, case_workers)) as ex:
            res = list(ex.map(lambda c: _case(fn, prob, sol, helper, c, work), prob["cases"]))
        ratios = [r for r, _ in res]
        infos = [i for _, i in res]
        return {"score": 100.0 * sum(ratios) / len(ratios), "status": "done", "cases": ratios, "case_info": infos,
                "infra": any(i.startswith("INFRA") for i in infos)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # the case could not run (fork / pipe / disk / sandbox): an infrastructure failure
        return 0.0, f"INFRA exception: {e}"[:200]


def _case(fn, *a):
    """One case in a worker process; a worker that died twice is an infrastructure failure too."""
    try:
        return _in_proc(_safe, fn, *a)
    except Exception as e:
        return 0.0, f"INFRA exception: {e}"[:200]


# ---- the game's entry: judge_code ------------------------------------------------------------------------------------
_SEM = None
_THREADS = None


def _sem() -> asyncio.Semaphore:
    """At most FCS_RM_CONCURRENCY programs judged at once (default: the job's own cores, 8 kept for the servers, over
    CASE_WORKERS)."""
    global _SEM
    if _SEM is None:
        n = int(os.environ.get("FCS_RM_CONCURRENCY", max(1, (job_cpus() - 8) // CASE_WORKERS)))
        _SEM = asyncio.Semaphore(n)
        _SEM.size = n
    return _SEM


def _thread(fn, *a):
    """fn(*a) in this module's own thread pool, a thread per semaphore slot plus 8: the loop's default executor holds
    at most 32 threads on every node (audit 2026-10-07)."""
    global _THREADS
    if _THREADS is None:
        _THREADS = ThreadPoolExecutor(max_workers=_sem().size + 8, thread_name_prefix="fcs-rm")
    return asyncio.get_running_loop().run_in_executor(_THREADS, functools.partial(fn, *a))


def judge_full(problem_dir: str, code: str) -> dict:
    """Judge a program: {"score": [0, 1], "cases": per-case ratios (zeros when nothing ran), "status": ...,
    "infra_error": bool, "msg": the compiler output of a compile error, "case_info": the judge's per-case lines, cut to
    160 chars ([] when nothing ran)}. Infrastructure failures are logged and flagged, never a silent 0. case_info can
    quote the checker, which may print expected output: keep it out of what an agent sees."""
    n = len(load_problem(Path(problem_dir))["cases"])
    if not code:
        return {"score": 0.0, "cases": [0.0] * n, "status": "no code", "infra_error": False, "msg": "", "case_info": []}
    try:
        r = judge(problem_dir, code)
    except Exception as e:
        print(f"[fcs_rm] JUDGE-ERROR {problem_dir}: {e}"[:500], file=sys.stderr, flush=True)
        return {"score": 0.0, "cases": [0.0] * n, "status": "judge error", "infra_error": True, "msg": "",
                "case_info": []}
    cases = r.get("cases") or [0.0] * n
    infra = bool(r.get("infra")) or r.get("status") == "infra error"
    if infra:
        print(f"[fcs_rm] JUDGE-INFRA {problem_dir}: {r.get('status')} {(r.get('msg') or '')[-200:]}"[:500],
              file=sys.stderr, flush=True)
    return {"score": r["score"] / 100.0, "cases": [float(c) for c in cases], "status": r["status"],
            "infra_error": infra, "msg": r.get("msg", "") if r["status"] == "compile error" else "",
            "case_info": [str(i)[:160] for i in (r.get("case_info") or [])]}


async def judge_code(problem_dir: str, code: str) -> dict:
    """judge_full in a thread under the shared FCS_RM_CONCURRENCY semaphore. The slot is held until the thread ends,
    also when the caller is cancelled (an aborted episode): a cancelled await would otherwise free the slot while the
    judge keeps running, and the extra judges oversubscribe the node."""
    async with _sem():
        fut = asyncio.ensure_future(_thread(judge_full, problem_dir, code))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            await asyncio.wait([fut])
            raise
