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
usage: judge(problem_dir, cpp_source) -> {"score": 0-100, "status": ..., "cases": [...]}
"""
from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import resource
import shutil
import signal
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
_RATIO = re.compile(r"Ratio: ([\d.]+)")


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


def _limits(cpu_s: float, mem_b: int | None):
    def f():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(cpu_s)) + 1,) * 2)
        if mem_b:
            resource.setrlimit(resource.RLIMIT_AS, (mem_b, mem_b))
            resource.setrlimit(resource.RLIMIT_STACK, (mem_b, mem_b))
        resource.setrlimit(resource.RLIMIT_FSIZE, (STDOUT_MAX, STDOUT_MAX))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


def _wait(p: subprocess.Popen, wall_s: float):
    """Wait for p with a wall limit; returns (exit_code or None if killed, cpu seconds, timed_out)."""
    deadline = time.monotonic() + wall_s
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            p.returncode = os.waitstatus_to_exitcode(status)
            return p.returncode, ru.ru_utime + ru.ru_stime, False
        if time.monotonic() > deadline:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, ru = os.wait4(p.pid, 0)
            return None, ru.ru_utime + ru.ru_stime, True
        time.sleep(0.005)


def _case_classic(prob, sol: Path, chk: Path, case, work: Path) -> float:
    inp, outp, t, mem = case
    td = prob["dir"] / "testdata"
    tl, ml = to_seconds(t), to_bytes(mem)
    out = work / f"{inp}.out"
    with open(td / inp, "rb") as fi, open(out, "wb") as fo:
        p = subprocess.Popen([str(sol)], stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, cwd=work,
                             preexec_fn=_limits(tl, ml))
        code, cpu, killed = _wait(p, 2 * tl)
    if killed or code != 0 or cpu > tl or out.stat().st_size > STDOUT_MAX:
        return 0.0, f"run: code={code} cpu={cpu:.2f}/{tl} killed={killed}"
    ans = td / outp if (td / outp).exists() else td / outp.replace(".ans", ".out")
    r = subprocess.run([str(chk), str(td / inp), str(out), str(ans)], capture_output=True, text=True,
                       errors="replace", preexec_fn=_limits(10, TRUSTED_AS), timeout=20)
    ok = r.returncode == OK_EXIT
    msg = r.stdout or r.stderr or ""
    m = _RATIO.search(msg) if r.returncode in (OK_EXIT, POINTS_EXIT) else None
    return (float(m.group(1)) if m else (1.0 if ok else 0.0)), f"chk={r.returncode} cpu={cpu:.2f}/{tl} {msg.strip()[:120]}"


def _case_interactive(prob, sol: Path, inter: Path, case, work: Path) -> float:
    inp, outp, t, mem = case
    td = prob["dir"] / "testdata"
    tl, ml = to_seconds(t), to_bytes(mem)
    ans = td / outp if (td / outp).exists() else td / outp.replace(".ans", ".out")
    s2i_r, s2i_w = os.pipe()
    i2s_r, i2s_w = os.pipe()
    err = open(work / f"{inp}.ierr", "w+b")
    pi = subprocess.Popen([str(inter), str(td / inp), str(work / f"{inp}.tout"), str(ans)], stdin=s2i_r, stdout=i2s_w,
                          stderr=err, cwd=work, preexec_fn=_limits(4 * tl, max(4 * ml, TRUSTED_AS)))
    ps = subprocess.Popen([str(sol)], stdin=i2s_r, stdout=s2i_w, stderr=subprocess.DEVNULL, cwd=work,
                          preexec_fn=_limits(tl, ml))
    for fd in (s2i_r, s2i_w, i2s_r, i2s_w):
        os.close(fd)
    scode, scpu, skilled = _wait(ps, 2 * tl)
    icode, _, ikilled = _wait(pi, 8 * tl)
    err.seek(0)
    imsg = err.read().decode(errors="replace")
    err.close()
    sub_ok = not skilled and scode == 0 and scpu <= tl
    inter_scoring = not ikilled and icode in (OK_EXIT, POINTS_EXIT)
    judge_msg = imsg if inter_scoring else ""
    info = f"sol code={scode} cpu={scpu:.2f}/{tl} killed={skilled}; inter code={icode} killed={ikilled} {imsg.strip()[:120]}"
    if not sub_ok:
        return 0.0, info                              # the run itself failed: no credit
    ok = icode == OK_EXIT and not ikilled
    m = _RATIO.search(judge_msg)
    return (float(m.group(1)) if m else (1.0 if ok else 0.0)), info


def judge(problem_dir, source: str, case_workers: int = CASE_WORKERS) -> dict:
    prob = load_problem(Path(problem_dir))
    if not prob["cases"]:
        return {"score": 0.0, "status": "no cases", "cases": []}
    work = Path(tempfile.mkdtemp(prefix="fcsj_", dir=TMP))
    try:
        (work / "sol.cpp").write_text(source)
        sol = work / "sol"
        r = subprocess.run(COMPILE + ["-o", str(sol), str(work / "sol.cpp")], capture_output=True, text=True,
                           errors="replace", timeout=120)
        if r.returncode != 0:
            return {"score": 0.0, "status": "compile error", "cases": [], "msg": r.stderr[-1000:]}
        helper = _compile_helper(prob["dir"], prob["interactor"] if prob["interactive"] else prob["checker"])
        fn = _case_interactive if prob["interactive"] else _case_classic
        with ThreadPoolExecutor(max_workers=max(1, case_workers)) as ex:
            res = list(ex.map(lambda c: _safe(fn, prob, sol, helper, c, work), prob["cases"]))
        ratios = [r for r, _ in res]
        return {"score": 100.0 * sum(ratios) / len(ratios), "status": "done", "cases": ratios,
                "case_info": [i for _, i in res]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        return 0.0, f"exception: {e}"[:200]
