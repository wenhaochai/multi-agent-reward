"""miles_team.team_core must equal molt's team_math: every module constant and every copied function on fixtures.
Runs in molt.sif (molt importable):
  PYTHONPATH=<labs-molt-team>:<this repo> MODEL_PATH=<Qwen3.8-27B> python3 tests/test_core_parity.py
"""
import importlib.util
import os
import sys

os.environ.update(MA_TM_MATES="3", MA_TM_LEAD_TURNS="4", MA_TM_MATE_TURNS="2", MA_TM_LEAD_THINK="0",
                  MA_TM_MATE_THINK="0", MA_TM_LEAD_BUDGET="12288", MA_TM_MATE_BUDGET="12288",
                  MA_TM_COLLAB_BONUS="0.1", MA_TM_TRACE_DIR="")
T = os.environ.get("MOLT_TEAM_PATH", "/scratch/gpfs/GROUP/USER/project/labs-molt-team")
spec = importlib.util.spec_from_file_location("molt_team_math", f"{T}/examples/python/agents/team_math.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from miles_team import team_core as C  # noqa: E402

names = [n for n in vars(M) if n.isupper() and not n.startswith("_")] + ["_DELEGATE_RE", "_SPECIAL_RE"]
bad = [n for n in names if getattr(M, n) != getattr(C, n, object())]
assert not bad, f"constants differ: {bad}"

texts = ["plain answer \\boxed{3}", "<think>x</think>\n<delegate to=1>a</delegate><delegate to=2> b </delegate>",
         "<delegate to=\"3\">c</delegate><delegate to=9>no</delegate><delegate to=1>last wins</delegate><delegate to=1>z</delegate>",
         "reason only, no close", "<｜end▁of▁sentence｜>after</think> tail<｜x｜>"]
for think in (True, False):
    for t in texts:
        assert M.visible(t, think) == C.visible(t, think)
        assert M.parse_delegations(t, thinking=think) == C.parse_delegations(t, thinking=think)
        for cut in (False, True):
            for failed in (False, True):
                assert (M.report_line(2, t, "7", cut, failed=failed, thinking=think)
                        == C.report_line(2, t, "7", cut, failed=failed, thinking=think))
rounds = [{"lead": (500, 300), "mates": [(900, 250), (1200, 260)]}, {"lead": (80, 1400), "mates": []}]
assert M.critical_path(rounds) == C.critical_path(rounds)
for args in [(1.0, True, True, 3000.0), (0.0, True, False, 90000.0), (1.0, False, False, 0.0)]:
    assert M.team_reward(*args) == C.team_reward(*args)
msgs = [{"role": "user", "content": "Q"}]
assert M._with_suffix(msgs, "\n\nS") == C._with_suffix(msgs, "\n\nS")
assert M._grade("so \\boxed{204}", "Q", "204") == C._grade("so \\boxed{204}", "Q", "204")
print(f"ALL OK: {len(names)} constants, visible/parse_delegations/report_line/critical_path/team_reward/_with_suffix/_grade")
