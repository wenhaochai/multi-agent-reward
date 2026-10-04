"""Frontier-CS reward for miles: FrontierSmith's code extraction (verl/verl/utils/reward_score/frontiercs.py,
strip_think + extract_cpp) scored by the local judge (fcs_judge.py, the official engine's rules).
sample.label is the problem directory; the reward is the judge score / 100 (EasyPPO's continuous 0-100 score, rescaled).
Judges run in threads (each a few subprocesses), at most FCS_RM_CONCURRENCY at once (default: cpus / FCS_CASE_WORKERS).
Use: --custom-rm-path miles_team.fcs_rm.fcs_rm
Code block rule (2026-10-04, user decision): the LAST ```cpp block is judged (FCS_EXTRACT=last, the default), as in
LiveCodeBench, open-r1, rllm/DeepCoder and verl. Frontier-CS's official harness and FrontierSmith take the LONGEST
block (FCS_EXTRACT=longest; Frontier-CS d5185d23, 2025-12-10, "likely the main solution"); on the SFT init's
multi-block answers the longest is usually the first draft (149 of 203) and the last block scores 4.88 vs 1.29.
For val problems the official (longest) score is also written to <run>/val_official.jsonl whenever the two rules pick
different code, so val can be reported both ways.
Every FCS_RM_TRACE_EVERY-th judged sample (default 32; 0 = off) is written in full to <run>/traces/solo_<pid>.jsonl
(<run> = the parent of --save): response, finish reason, extracted code, status, score; tools/read_traces.py reads it.
"""
import asyncio
import json
import os
import re
import sys
import time

from miles_team.fcs_judge import CASE_WORKERS, judge

_SEM = None


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


EXTRACT = os.environ.get("FCS_EXTRACT", "last")
assert EXTRACT in ("last", "longest"), f"FCS_EXTRACT must be last or longest, not {EXTRACT}"


def extract_cpp(response_text: str, rule: str | None = None) -> str:
    """Extract C++ code from model response (markdown or raw), ignoring <think> blocks; with several ```cpp blocks,
    the last one (rule "last", the default) or the longest one (rule "longest", Frontier-CS's official harness)."""
    if not response_text:
        return ""

    response_text = strip_think(response_text)
    if not response_text:
        return ""

    code = response_text.strip()

    # Try to extract from ```cpp blocks
    cpp_pattern = r'```(?:cpp|c\+\+)?\s*\n(.*?)```'
    matches = re.findall(cpp_pattern, code, re.DOTALL)
    if matches:
        return (max(matches, key=len) if (rule or EXTRACT) == "longest" else matches[-1]).strip()

    # Fallback: strip markdown if present
    if code.startswith("```cpp"):
        code = code[6:].strip()
    elif code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def score(response: str, problem_dir: str) -> float:
    code = extract_cpp(response or "")
    if not code:
        return 0.0
    try:
        r = judge(problem_dir, code)
        if r.get("infra"):  # a case or the compile could not run (fork / disk / sandbox): scored 0, made visible
            print(f"[fcs_rm] JUDGE-INFRA {problem_dir}: {r.get('status')} {(r.get('msg') or '')[-200:]}"[:500],
                  file=sys.stderr, flush=True)
        return r["score"] / 100.0
    except Exception as e:  # an infrastructure failure (helper compile, disk), not the policy's fault: make it visible
        print(f"[fcs_rm] JUDGE-ERROR {problem_dir}: {e}"[:500], file=sys.stderr, flush=True)
        return 0.0


def _sem():
    global _SEM
    if _SEM is None:
        n = int(os.environ.get("FCS_RM_CONCURRENCY", max(1, (os.cpu_count() or 8) // CASE_WORKERS)))
        _SEM = asyncio.Semaphore(n)
    return _SEM


_TRACE_EVERY = int(os.environ.get("FCS_RM_TRACE_EVERY", "32"))
_trace_n = 0


def _trace(args, sample, reward: float) -> None:
    """Logging only: the full sample of every _TRACE_EVERY-th call, with the status the reward came from."""
    global _trace_n
    _trace_n += 1
    if not _TRACE_EVERY or (_trace_n - 1) % _TRACE_EVERY or not getattr(args, "save", None):
        return
    try:
        resp = sample.response or ""
        code = extract_cpp(resp)
        r = judge_full(sample.label, code) if code else {"status": "no code", "score": 0.0}
        d = os.path.join(os.path.dirname(os.path.abspath(args.save)), "traces")
        os.makedirs(d, exist_ok=True)
        rec = {"time": round(time.time(), 1), "label": sample.label, "reward": reward, "status": r["status"], "rejudged_score": r["score"],
               "truncated": str(getattr(sample, "status", "")).endswith("TRUNCATED"),
               "response_length": getattr(sample, "response_length", None), "closed_think": "</think>" in resp,
               "text": resp}
        with open(os.path.join(d, f"solo_{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:  # tracing must never affect training
        print(f"[fcs_rm] trace failed: {e}"[:300], file=sys.stderr, flush=True)


def _val_official(args, sample, reward: float) -> None:
    """Val problems only: when the official (longest-block) rule picks other code than the rule in use, judge it too
    and write both scores to <run>/val_official.jsonl (logging only; the reward is unchanged)."""
    try:
        from miles_team.fcs_judge import FCS_ROOT
        from pathlib import Path
        if not getattr(args, "save", None) or not Path(sample.label).resolve().is_relative_to(FCS_ROOT.resolve()):
            return
        resp = sample.response or ""
        code, off = extract_cpp(resp), extract_cpp(resp, "longest")
        official = reward if off == code else (judge_full(sample.label, off)["score"] if off else 0.0)
        d = os.path.dirname(os.path.abspath(args.save))
        with open(os.path.join(d, "val_official.jsonl"), "a") as f:
            f.write(json.dumps({"time": round(time.time(), 1), "label": sample.label, "reward": reward,
                                "official": official, "same_code": off == code}) + "\n")
    except Exception as e:
        print(f"[fcs_rm] val_official failed: {e}"[:300], file=sys.stderr, flush=True)


async def _one(sample, args=None) -> float:
    async with _sem():
        reward = await asyncio.to_thread(score, sample.response or "", sample.label)
        if args is not None and EXTRACT != "longest":
            await asyncio.to_thread(_val_official, args, sample, reward)
        if args is not None and _TRACE_EVERY:
            await asyncio.to_thread(_trace, args, sample, reward)
        return reward


def judge_full(problem_dir: str, code: str) -> dict:
    """Judge already-extracted code: {"score": [0, 1], "cases": per-case ratios (zeros when nothing ran),
    "status": ..., "infra_error": bool}. Infrastructure failures are logged and flagged, never a silent 0."""
    from miles_team.fcs_judge import load_problem
    from pathlib import Path
    n = len(load_problem(Path(problem_dir))["cases"])
    if not code:
        return {"score": 0.0, "cases": [0.0] * n, "status": "no code", "infra_error": False, "msg": ""}
    try:
        r = judge(problem_dir, code)
    except Exception as e:
        print(f"[fcs_rm] JUDGE-ERROR {problem_dir}: {e}"[:500], file=sys.stderr, flush=True)
        return {"score": 0.0, "cases": [0.0] * n, "status": "judge error", "infra_error": True, "msg": ""}
    cases = r.get("cases") or [0.0] * n
    infra = bool(r.get("infra")) or r.get("status") == "infra error"
    if infra:
        print(f"[fcs_rm] JUDGE-INFRA {problem_dir}: {r.get('status')} {(r.get('msg') or '')[-200:]}"[:500],
              file=sys.stderr, flush=True)
    return {"score": r["score"] / 100.0, "cases": [float(c) for c in cases], "status": r["status"],
            "infra_error": infra, "msg": r.get("msg", "") if r["status"] == "compile error" else ""}


async def judge_code(problem_dir: str, code: str) -> dict:
    """judge_full in a thread under the shared FCS_RM_CONCURRENCY semaphore."""
    async with _sem():
        return await asyncio.to_thread(judge_full, problem_dir, code)


async def fcs_rm(args, sample, **kwargs):
    if isinstance(sample, list):
        return list(await asyncio.gather(*(_one(s, args) for s in sample)))
    return await _one(sample, args)
