"""No answer, no score (owner decision 2026-10-07): miles_team.fcs_rm.answered/answer and the solo reward _one.
A reply cut off at its token limit, or a thinking reply without </think>, scores 0 and is never judged; every other
reply is judged exactly as before. python3 tests/test_fcs_noanswer.py (PYTHONPATH: this repo + miles-easyppo)"""
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FCS_RM_TRACE_EVERY", "0")
import miles_team.fcs_rm as R  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

CODE = "```cpp\nint main(){}\n```"
# answer(): the text after the last </think>; none when the reasoning never closed
assert R.answer(f"r</think>\n{CODE}") == f"\n{CODE}"
assert R.answer(f"draft {CODE} still thinking") == ""  # code inside unclosed reasoning is not an answer
assert R.answer("") == "" and R.answer(None) == ""
assert R.answer("a</think>b</think>c") == "c"  # the last close wins, as strip_think
assert R.answer(f"no think {CODE}", thinking=False) == f"no think {CODE}"
assert R.answer("</think>") == ""
assert R.answer("<think>x") == ""  # an opening tag alone is not a close
assert R.answer("x<|im_end|>") == ""  # special tokens after unclosed reasoning
# answered(): cut or unclosed -> False
assert R.answered(f"r</think>{CODE}", cut=False, thinking=True)
assert not R.answered(f"r</think>{CODE}", cut=True, thinking=True)
assert not R.answered(f"r {CODE}", cut=False, thinking=True)
assert R.answered(f"r {CODE}", cut=False, thinking=False) and not R.answered(f"r {CODE}", cut=True, thinking=False)
assert not R.answered("", cut=False, thinking=True)
# _cut() reads miles' Sample.Status
s = Sample()
s.status = Sample.Status.TRUNCATED
assert R._cut(s)
for st in (Sample.Status.COMPLETED, Sample.Status.ABORTED, Sample.Status.PENDING):
    s.status = st
    assert not R._cut(s), st
assert not R._cut(SimpleNamespace())

# _one: unanswered replies are never judged; answered ones get the judge's score
calls = []
R.score = lambda resp, label: calls.append(resp) or 0.5


def run(resp, status):
    sm = Sample()
    sm.response, sm.label, sm.status = resp, "p", status
    return asyncio.run(R.fcs_rm(SimpleNamespace(save=None), sm))


T, C = Sample.Status.TRUNCATED, Sample.Status.COMPLETED
assert run(f"r</think>{CODE}", C) == 0.5 and len(calls) == 1
assert run(f"r</think>{CODE}", T) == 0.0 and len(calls) == 1  # cut off after closing: still nothing
assert run(f"r {CODE}", C) == 0.0 and len(calls) == 1  # stopped inside the reasoning
assert run(f"r {CODE}", T) == 0.0 and len(calls) == 1
assert run("", C) == 0.0 and run(None, C) == 0.0 and len(calls) == 1
assert run("x" * 200000 + f"</think>{CODE}", C) == 0.5 and len(calls) == 2  # long reply, closed
print("test_fcs_noanswer: ok")
