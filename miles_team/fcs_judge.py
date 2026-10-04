"""Local Frontier-CS algorithmic judge (no Docker, no judge server): the scoring of the official go-judge engine
(Frontier-CS/algorithmic/judge/src/judge_engine.js, problem_manager.js, utils.js) re-implemented with subprocesses
and rlimits, so RL rewards on Della match what the judge server would return.

Rules copied from the official engine:
  * cases: n_cases per subtask from config.yaml (prefix/suffix default '' / '.in' / '.ans', numbered from 1), plus every
    other complete <n>.in + <n>.ans (or .out) pair in testdata/ at the last subtask's limits;
  * compile: g++ -O2 -pipe -static -s -std=gnu++17; a compile error scores 0;
  * run: CPU time <= the case time limit, wall <= 2x, address space and stack <= the memory limit, stdout <= 128 MiB;
    anything else (TLE, MLE, runtime error, non-zero exit) scores the case 0;
  * classic: checker `chk in.txt out.txt ans.txt` (10 s CPU, 256 MiB); interactive: the program and
    `interactor in.txt tout.txt ans.txt` (4x time and memory) joined by two pipes, scored from the interactor's stderr;
  * a checker / interactor message is trusted only when it exits 0 (testlib _ok) or 7 (testlib points); the case ratio
    is its "Ratio: x" if present, else 1 when it accepted and 0 otherwise;
  * problem score = 100 x mean case ratio.
Deviations, all from the 2026-10-04 audit:
  * training problems (outside FCS_ROOT: the FrontierSmith set) parse the full float ("Ratio: 5e-05" is 5e-05; the
    official regex reads digits and dots only, so 5e-05 becomes 5, and frontiersmith_112's checker prints %g) and clamp each ratio to [0, 1]
    (non-finite -> 0): an RL reward must not reward a near-zero ratio with 5. Val keeps the official regex and quirks;
  * checkers and interactors get a 2 GiB address space (official: none; 256 MiB broke val 23's checker);
  * interactive programs: the official engine sets no address-space limit, only a memory limit, so the program gets a
    max(4 x ML, 1 GiB) address-space cap (node safety) and an MLE when its peak RSS exceeds ML;
  * untrusted code is sandboxed with Landlock (FCS_SANDBOX=0 disables): g++ reads only system dirs and its work dir
    (no #include of test answers), the program opens only its binary, /dev/null, /dev/{u,}random and /proc/self, with
    an empty environment;
  * infrastructure failures (an exception in a case, g++ killed by a signal or out of disk / processes) are flagged
    ("infra": True) instead of passing as plain zeros.
usage: judge(problem_dir, cpp_source) -> {"score": 0-100, "status": ..., "cases": [...], "infra": bool}
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import math
import os
import re
import resource
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

FCS_ROOT = Path(os.environ.get("FCS_ROOT", "/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic"))
TESTLIB_DIR = FCS_ROOT / "judge" / "include"
CACHE = Path(os.environ.get("FCS_JUDGE_CACHE", "/tmp/fcs_judge_cache"))
TMP = Path(os.environ.get("FCS_JUDGE_TMP", "/tmp"))
CASE_WORKERS = int(os.environ.get("FCS_CASE_WORKERS", "8"))
COMPILE = ["g++", "-O2", "-pipe", "-static", "-s", "-std=gnu++17"]
STDOUT_MAX = 128 << 20
OK_EXIT, POINTS_EXIT = 0, 7
_RATIO = re.compile(r"Ratio: ([\d.]+)")  # the official engine's regex (val keeps its quirks)
_RATIO_FULL = re.compile(r"Ratio:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf(?:inity)?|[-+]?nan)", re.I)
_INFRA_MARKERS = ("No space left on device", "Resource temporarily unavailable", "Cannot allocate memory",
                  "cannot fork", "virtual memory exhausted", "Too many open files")
SANDBOX = os.environ.get("FCS_SANDBOX", "1") == "1"


def _ratio(msg: str, trusted: bool, ok: bool, train: bool) -> float:
    """A case's ratio from a checker / interactor message (trusted = it exited 0 or 7)."""
    if not train:
        m = _RATIO.search(msg) if trusted else None
        return float(m.group(1)) if m else (1.0 if ok else 0.0)
    m = _RATIO_FULL.search(msg) if trusted else None
    if not m:
        return 1.0 if ok else 0.0
    v = float(m.group(1))
    return min(1.0, max(0.0, v)) if math.isfinite(v) else 0.0


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
    try:
        train = not Path(pdir).resolve().is_relative_to(FCS_ROOT.resolve())
    except OSError:
        train = True
    return {"dir": pdir, "cases": cases, "interactive": interactive, "train": train,
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


def _landlock_rules(rules):
    """Pre-resolved [(path, access)] for _limits: only paths that exist; a file rule keeps only file rights."""
    out = []
    for path, access in rules:
        if os.path.exists(path):
            out.append((path, access if os.path.isdir(path) else access & _FS_FILE))
    return out


def _restrict(rules):
    """In the child (preexec): allow only `rules` [(path, access)], deny every other handled file access."""
    attr = ctypes.create_string_buffer(struct.pack("<Q", _HANDLED))
    rs = _libc.syscall(_LL_CREATE, attr, ctypes.c_size_t(8), ctypes.c_uint32(0))
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


def _limits(cpu_s: float, mem_b: int | None, stack_b: int | None = None, sandbox=None):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(cpu_s)) + 1,) * 2)
        if mem_b:
            resource.setrlimit(resource.RLIMIT_AS, (mem_b, mem_b))
            s = stack_b or mem_b
            resource.setrlimit(resource.RLIMIT_STACK, (s, s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (STDOUT_MAX, STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
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


def _wait(p: subprocess.Popen, wall_s: float, rss: list | None = None):
    """Wait for p with a wall limit; returns (exit_code or None if killed, cpu seconds, timed_out).
    rss, if given, receives the peak resident set in bytes."""
    deadline = time.monotonic() + wall_s
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            p.returncode = os.waitstatus_to_exitcode(status)
            if rss is not None:
                rss.append(ru.ru_maxrss * 1024)
            return p.returncode, ru.ru_utime + ru.ru_stime, False
        if time.monotonic() > deadline:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, ru = os.wait4(p.pid, 0)
            if rss is not None:
                rss.append(ru.ru_maxrss * 1024)
            return None, ru.ru_utime + ru.ru_stime, True
        time.sleep(0.005)


def _case_classic(prob, sol: Path, chk: Path, case, work: Path) -> float:
    inp, outp, t, mem = case
    td = prob["dir"] / "testdata"
    tl, ml = to_seconds(t), to_bytes(mem)
    out = work / f"{inp}.out"
    with open(td / inp, "rb") as fi, open(out, "wb") as fo:
        p = subprocess.Popen([str(sol)], stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, cwd=work, env=_CLEAN_ENV,
                             preexec_fn=_limits(tl, ml, sandbox=_program_sandbox(sol)))
        code, cpu, killed = _wait(p, 2 * tl)
    if killed or code != 0 or cpu > tl or out.stat().st_size > STDOUT_MAX:
        return 0.0, f"run: code={code} cpu={cpu:.2f}/{tl} killed={killed}"
    ans = td / outp if (td / outp).exists() else td / outp.replace(".ans", ".out")
    r = subprocess.run([str(chk), str(td / inp), str(out), str(ans)], capture_output=True, text=True,
                       errors="replace", preexec_fn=_limits(10, TRUSTED_AS), timeout=20)
    ok = r.returncode == OK_EXIT
    msg = r.stdout or r.stderr or ""
    ratio = _ratio(msg, r.returncode in (OK_EXIT, POINTS_EXIT), ok, prob["train"])
    return ratio, f"chk={r.returncode} cpu={cpu:.2f}/{tl} {msg.strip()[:120]}"


def _case_interactive(prob, sol: Path, inter: Path, case, work: Path) -> float:
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
                                                                 sandbox=_program_sandbox(sol)))
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
    sub_ok = not skilled and scode == 0 and scpu <= tl and not mle
    inter_scoring = not ikilled and icode in (OK_EXIT, POINTS_EXIT)
    info = (f"sol code={scode} cpu={scpu:.2f}/{tl} rss={peak[0] >> 20 if peak else -1}MiB/{ml >> 20} killed={skilled}; "
            f"inter code={icode} killed={ikilled} {imsg.strip()[:120]}")
    if not sub_ok:
        return 0.0, info                              # the run itself failed: no credit
    ok = icode == OK_EXIT and not ikilled
    return _ratio(imsg if inter_scoring else "", inter_scoring, ok, prob["train"]), info


def judge(problem_dir, source: str, case_workers: int = CASE_WORKERS) -> dict:
    prob = load_problem(Path(problem_dir))
    if not prob["cases"]:
        return {"score": 0.0, "status": "no cases", "cases": []}
    work = Path(tempfile.mkdtemp(prefix="fcsj_", dir=TMP))
    try:
        (work / "sol.cpp").write_text(source)
        sol = work / "sol"
        r = subprocess.run(COMPILE + ["-o", str(sol), str(work / "sol.cpp")], capture_output=True, text=True,
                           errors="replace", timeout=120, env={**_CLEAN_ENV, "TMPDIR": str(work)},
                           preexec_fn=_limits(600, None, sandbox=_compile_sandbox(work)))
        if r.returncode != 0:
            infra = r.returncode < 0 or any(m in r.stderr for m in _INFRA_MARKERS)
            return {"score": 0.0, "status": "infra error" if infra else "compile error", "cases": [],
                    "msg": r.stderr[-1000:], "infra": infra}
        helper = _compile_helper(prob["dir"], prob["interactor"] if prob["interactive"] else prob["checker"])
        fn = _case_interactive if prob["interactive"] else _case_classic
        with ThreadPoolExecutor(max_workers=max(1, case_workers)) as ex:
            res = list(ex.map(lambda c: _safe(fn, prob, sol, helper, c, work), prob["cases"]))
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
