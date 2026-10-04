"""Frontier-CS reward for miles: FrontierSmith's code extraction (verl/verl/utils/reward_score/frontiercs.py,
strip_think + extract_cpp, copied verbatim) scored by the local judge (fcs_judge.py, the official engine's rules).
sample.label is the problem directory; the reward is the judge score / 100 (EasyPPO's continuous 0-100 score, rescaled).
Judges run in threads (each a few subprocesses), at most FCS_RM_CONCURRENCY at once (default: cpus / FCS_CASE_WORKERS).
Use: --custom-rm-path miles_team.fcs_rm.fcs_rm
"""
import asyncio
import os
import re
import sys

from miles_team.fcs_judge import CASE_WORKERS, judge

_SEM = None


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_cpp(response_text: str) -> str:
    """Extract C++ code from model response (markdown or raw), ignoring <think> blocks."""
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
        return max(matches, key=len).strip()

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


async def _one(sample) -> float:
    async with _sem():
        return await asyncio.to_thread(score, sample.response or "", sample.label)


def judge_full(problem_dir: str, code: str) -> dict:
    """Judge already-extracted code: {"score": [0, 1], "cases": per-case ratios (zeros when nothing ran),
    "status": ..., "infra_error": bool}. Infrastructure failures are logged and flagged, never a silent 0."""
    from miles_team.fcs_judge import load_problem
    from pathlib import Path
    n = len(load_problem(Path(problem_dir))["cases"])
    if not code:
        return {"score": 0.0, "cases": [0.0] * n, "status": "no code", "infra_error": False}
    try:
        r = judge(problem_dir, code)
    except Exception as e:
        print(f"[fcs_rm] JUDGE-ERROR {problem_dir}: {e}"[:500], file=sys.stderr, flush=True)
        return {"score": 0.0, "cases": [0.0] * n, "status": "judge error", "infra_error": True}
    cases = r.get("cases") or [0.0] * n
    infra = bool(r.get("infra")) or r.get("status") == "infra error"
    if infra:
        print(f"[fcs_rm] JUDGE-INFRA {problem_dir}: {r.get('status')} {(r.get('msg') or '')[-200:]}"[:500],
              file=sys.stderr, flush=True)
    return {"score": r["score"] / 100.0, "cases": [float(c) for c in cases], "status": r["status"],
            "infra_error": infra}


async def judge_code(problem_dir: str, code: str) -> dict:
    """judge_full in a thread under the shared FCS_RM_CONCURRENCY semaphore."""
    async with _sem():
        return await asyncio.to_thread(judge_full, problem_dir, code)


async def fcs_rm(args, sample, **kwargs):
    if isinstance(sample, list):
        return list(await asyncio.gather(*(_one(s) for s in sample)))
    return await _one(sample)
