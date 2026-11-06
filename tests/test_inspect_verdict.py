"""flint.miles_agent._inspect_verdict: which sample-check results stop a run (2026-11-06, w786 rollout 64).
python3 tests/test_inspect_verdict.py (no GPU, no miles)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flint.miles_agent import TRAINED_SHORT_MAX, _inspect_verdict  # noqa: E402

OK = {"failed episodes": 0, "judge infra submissions": 0, "rewards not S": 0, "trained != generated": 0,
      "top-k errors": 0, "R3 ids out of range": 0}


def summ(gen, trained, shared=0):
    return {"ok_generated_tokens": gen, "ok_trained_tokens": trained, "ok_shared_stop_tokens": shared}


def check(name, got, want):
    assert got == want, f"{name}: {got} != {want}"
    print("ok", name)


check("all good", _inspect_verdict(summ(2048000, 2048000), OK), ({}, {}))
# w786 rollout 64, first and second try: 12 and 7 tokens short
check("12 short warns", _inspect_verdict(summ(2048000, 2047988), {**OK, "trained != generated": 1}),
      ({}, {"trained short of generated": 12}))
check("7 short warns", _inspect_verdict(summ(2048000, 2047993), {**OK, "trained != generated": 1}),
      ({}, {"trained short of generated": 7}))
check("158 short warns (the third try)", _inspect_verdict(summ(2048000, 2047842), {**OK, "trained != generated": 1}),
      ({}, {"trained short of generated": 158}))
check("a shared stop token counts as trained", _inspect_verdict(summ(2048000, 2047999, 1), OK), ({}, {}))
check("the limit warns", _inspect_verdict(summ(2048000, 2048000 - TRAINED_SHORT_MAX), {**OK, "trained != generated": 1}),
      ({}, {"trained short of generated": TRAINED_SHORT_MAX}))
check("over the limit stops", _inspect_verdict(summ(2048000, 2048000 - TRAINED_SHORT_MAX - 1), {**OK, "trained != generated": 1}),
      ({"trained != generated": 1}, {}))
check("more trained than generated stops", _inspect_verdict(summ(2048000, 2048005), {**OK, "trained != generated": 1}),
      ({"trained != generated": 1}, {}))
check("another problem still stops", _inspect_verdict(summ(2048000, 2047990), {**OK, "trained != generated": 1, "top-k errors": 2}),
      ({"top-k errors": 2}, {"trained short of generated": 10}))
check("failed episodes do not stop", _inspect_verdict(summ(2048000, 2048000), {**OK, "failed episodes": 1}), ({}, {}))
p = {**OK, "trained != generated": 1}
_inspect_verdict(summ(2048000, 2047988), p)
check("the caller's dict is not changed", p["trained != generated"], 1)
print("ALL OK")
