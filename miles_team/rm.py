"""Reward for miles runs of the labs-molt Qwen3.8-27B team study: molt's own math grader (examples/python/utils/
math_grader.py, the one team_math.py scores with), so a miles arm and a molt arm grade every answer identically.
Use: --custom-rm-path miles_team.rm.molt_math_rm
"""
import importlib.util
import os

_T = os.environ.get("MOLT_TEAM_PATH", "/scratch/gpfs/GROUP/USER/project/labs-molt-team")
_spec = importlib.util.spec_from_file_location("molt_math_grader", f"{_T}/examples/python/utils/math_grader.py")
_GRADER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_GRADER)


def grade(text: str, label) -> float:
    try:
        return float(_GRADER.score_response(text, "", label).get("reward", 0.0))
    except Exception:
        return 0.0


async def molt_math_rm(args, sample, **kwargs) -> float:
    return grade(sample.response or "", sample.label)
