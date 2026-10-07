"""Eval judged after training resumes (miles_team.fcs_rm: log_eval / log_rollout / score_pending): the logged eval
must equal the eval judged inline (miles' own log_eval_rollout_data on the same samples), for a background eval, one
an earlier segment left pending, the run's last eval and the offline scorer; cut-off and unclosed replies score 0.
python3 tests/test_fcs_defer.py   (in miles.sif; PYTHONPATH: this repo + miles-easyppo)"""
import asyncio
import copy
import glob
import gzip
import json
import logging
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["FCS_RM_TRACE_EVERY"] = "0"
import miles_team.fcs_rm as R  # noqa: E402
from miles.ray.rollout import metrics as M  # noqa: E402
from miles.utils import logging_utils  # noqa: E402
from miles.utils.tracking_utils import tracking  # noqa: E402
from miles.utils.types import Sample  # noqa: E402

logging_utils.configure_logger_raw("rollout_executor")
LINES, ROWS = [], []


class Grab(logging.Handler):
    def emit(self, rec):
        LINES.append(f"{rec.filename}:{rec.lineno} - {rec.getMessage()}")


logging.getLogger().addHandler(Grab())
tracking.log = lambda args, d, step_key: ROWS.append(dict(d))
JUDGED = []
R.score = lambda resp, label, pool=None: JUDGED.append(label) or (int(label[1:]) % 7) / 10  # deterministic fake judge
tmp = tempfile.mkdtemp()
ARGS = Namespace(save=f"{tmp}/run/ckpt", num_rollout=20, custom_eval_rollout_log_function_path="miles_team.fcs_rm.log_eval",
                 log_passrate=False, n_samples_per_eval_prompt=2, rollout_batch_size=4, n_samples_per_prompt=2,
                 global_batch_size=8, use_dynamic_global_batch_size=False, advantage_estimator="ppo",
                 eval_reward_key=None, reward_key=None, use_session_server=False, num_steps_per_rollout=None,
                 sglang_speculative_algorithm=None, ci_test=False, ci_tito_special_token_count_threshold=None,
                 custom_rollout_log_function_path="miles_team.fcs_rm.log_rollout", load_debug_rollout_data=None, log_reward_category=None,
                 rollout_num_gpus=8, wandb_always_use_train_step=False)
CODE = "```cpp\nint main(){}\n```"


def eval_data(n=12, defer=True):
    """An eval as eval_rollout_single_dataset returns it, rewards from the rm (deferred: placeholders)."""
    out = []
    for i in range(n):
        s = Sample()
        s.index, s.label, s.response_length = i, f"p{i}", 100 + i
        s.response = (f"r</think>{CODE}" if i % 4 else f"r {CODE}")  # every 4th never closed its reasoning
        s.status = Sample.Status.TRUNCATED if i % 5 == 3 else Sample.Status.COMPLETED
        s.metadata = {"pid": str(i), **({R.DEFER_KEY: 1} if defer else {})}
        out.append(s)
    for s in out:
        s.reward = asyncio.run(R.fcs_rm(ARGS, s))
    return {"fcs_val": {"rewards": [s.reward for s in out], "truncated": [s.status == Sample.Status.TRUNCATED for s in out],
                        "samples": out}}


def inline(rid):
    """What miles logs when the rm judges inline: the reference."""
    a = copy.copy(ARGS)
    a.custom_eval_rollout_log_function_path = None
    LINES.clear(); ROWS.clear()
    d = eval_data(defer=False)
    M.log_eval_rollout_data(rid, a, d, {})
    return ROWS[-1], [l for l in LINES if " - eval " in l][-1]


def same(row, ref):
    assert row == ref and list(row) == list(ref), (row, ref)  # same keys, same order, same values


ref_row, ref_line = inline(9)
ref_row_rewards = eval_data(defer=False)["fcs_val"]["rewards"]
assert ref_line.startswith("metrics.py:") and " - eval 9: {" in ref_line, ref_line
expect = sum((int(f"p{i}"[1:]) % 7) / 10 for i in range(12) if i % 4 and i % 5 != 3) / 12
assert abs(ref_row["eval/fcs_val"] - expect) < 1e-12, (ref_row, expect)

# 1) a deferred eval: placeholders from the rm, never judged during generation
JUDGED.clear()
d = eval_data()
assert d["fcs_val"]["rewards"] == [0.0] * 12 and not JUDGED and len(R._deferred(d)) == 12
# 2) log_eval: saved, judged in the background, logged at the next rollout through miles' own function
LINES.clear(); ROWS.clear()
assert R.log_eval(9, ARGS, d, {}) is True and not ROWS
pend = f"{tmp}/run/eval_pending/eval_9.pt"
assert os.path.exists(pend)
for t in R._bg:
    t.join()
assert not ROWS  # judged, not yet logged: logging waits for miles' own loop
assert R.log_rollout(10, ARGS, [], {}, 1.0) is False
assert len(ROWS) == 1 and ROWS[0]["eval/step"] == ref_row["eval/step"]
same(ROWS[0], ref_row)
line = [l for l in LINES if " - eval 9: {" in l]
assert line and line[0].startswith("metrics.py:"), LINES[-3:]
assert not os.path.exists(pend) and os.path.exists(pend[:-3] + ".jsonl.gz")
kept = [json.loads(l) for l in gzip.open(pend[:-3] + ".jsonl.gz", "rt")]
assert len(kept) == 12 and sum(r["reward"] for r in kept) / 12 == ref_row["eval/fcs_val"] and kept[1]["response"].startswith("r</think>")
assert sorted(JUDGED) == sorted(f"p{i}" for i in range(12) if i % 4 and i % 5 != 3)  # unanswered replies never judged
print("[ok] background eval == inline eval; unanswered replies never judged")

# 3) an eval an earlier segment left pending: judged at the next segment's first rollout
ref19, _ = inline(14)
d = eval_data()
R._save_pending(f"{tmp}/run/eval_pending/eval_14.pt", 14, ARGS, d, {})
R._resumed, R._bg[:] = False, []
LINES.clear(); ROWS.clear()
R.log_rollout(15, ARGS, [], {}, 1.0)
assert len(R._bg) == 1 and not ROWS
R._bg[0].join()
R.log_rollout(16, ARGS, [], {}, 1.0)
assert len(ROWS) == 1
same(ROWS[0], ref19)
print("[ok] pending eval of an earlier segment == inline eval")

# 4) the run's last eval: judged inline as before; first the evals still pending are judged and logged
ref17, _ = inline(17)
ref_last, _ = inline(19)
R._save_pending(f"{tmp}/run/eval_pending/eval_17.pt", 17, ARGS, eval_data(), {})
LINES.clear(); ROWS.clear()
R.log_rollout(19, ARGS, [], {}, 1.0)  # the rollout before the last eval
d = eval_data()
assert not R._deferred(d) and d["fcs_val"]["rewards"] != [0.0] * 12  # judged inline
M.log_eval_rollout_data(19, ARGS, d, {})  # what miles calls; log_eval settles the pending ones first, then miles logs
assert [r["eval/step"] for r in ROWS] == [17, 19], [r["eval/step"] for r in ROWS]
same(ROWS[0], ref17); same(ROWS[1], ref_last)
assert not glob.glob(f"{tmp}/run/eval_pending/*.pt")
R._final_next = False
print("[ok] last eval judged inline == inline eval; a pending eval is settled before it")

# 5) no deferred samples: miles logs as usual
assert R.log_eval(4, ARGS, eval_data(defer=False), {}) is False

# 6) the offline scorer (a segment that finds the run complete)
d = eval_data()
R._save_pending(f"{tmp}/run/eval_pending/eval_4.pt", 4, ARGS, d, {})
ref4, _ = inline(4)
LINES.clear()
logging_utils.configure_logger_raw = lambda name="": None  # keep the test's log handler (score_pending configures miles' format)
R.score_pending(f"{tmp}/run")
line = [l for l in LINES if " - eval 4: {" in l]
assert line and line[0].startswith("metrics.py:")
logged = eval(line[0].split(" - eval 4: ", 1)[1], {"nan": float("nan")})
assert logged["eval/fcs_val"] == ref4["eval/fcs_val"] and not glob.glob(f"{tmp}/run/eval_pending/*.pt")
print("[ok] offline scorer logs the same line")
# 7) misconfigured: without our rollout log function nothing is deferred (miles would log the placeholders)
a = copy.copy(ARGS); a.custom_rollout_log_function_path = None
s0 = eval_data(defer=False)["fcs_val"]["samples"][1]
s0.metadata[R.DEFER_KEY], s0.reward = 1, None
assert asyncio.run(R.fcs_rm(a, s0)) == (int("1") % 7) / 10 and not s0.metadata.get("fcs_deferred")
print("[ok] misconfigured run judges inline")

# 8) a corrupt pending file is set aside, training goes on
open(f"{tmp}/run/eval_pending/eval_7.pt", "wb").write(b"not a torch file")
R._resumed = False
assert R.log_rollout(8, ARGS, [], {}, 1.0) is False
assert os.path.exists(f"{tmp}/run/eval_pending/eval_7.pt.failed") and not os.path.exists(f"{tmp}/run/eval_pending/eval_7.pt")
print("[ok] corrupt pending eval set aside")

# 9) the save fails (disk full): judged at once, miles logs it
save, R._save_pending = R._save_pending, lambda *a: (_ for _ in ()).throw(OSError(28, "No space left on device"))
d = eval_data()
assert R.log_eval(9, ARGS, d, {}) is False and not R._deferred(d) and d["fcs_val"]["rewards"] == ref_row_rewards
R._save_pending = save
print("[ok] a failed save falls back to inline judging")
print("test_fcs_defer: ALL OK")
