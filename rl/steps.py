"""Per-step table of an RL run (rl/rl_inner.sh): miles' own logs in train.log (the training step's "step N: {...}", the
"perf N: {...}" of the trainer and of the rollout) joined with flint.miles_agent.log_rollout's steps.jsonl (the
episodes' S, submissions, the served-weights probe). Usage: steps.py <run dir>; prints the table, writes steps.json."""
import ast
import json
import os
import re
import sys
from collections import defaultdict

d = sys.argv[1]
steps = defaultdict(dict)
line_re = re.compile(r"\b(step|perf) (\d+): (\{.*\})\s*$")
for line in open(os.path.join(d, "train.log"), errors="replace"):
    m = line_re.search(line)
    if not m:
        continue
    try:
        steps[int(m.group(2))].update(ast.literal_eval(m.group(3)))
    except (ValueError, SyntaxError):
        continue
if os.path.exists(os.path.join(d, "steps.jsonl")):
    for line in open(os.path.join(d, "steps.jsonl")):
        r = json.loads(line)
        steps[r["rollout_id"]].update({f"flint/{k}": v for k, v in r.items()})
json.dump(steps, open(os.path.join(d, "steps.json"), "w"), indent=1, default=str)

COLS = [("S mean", "flint/reward_mean", "{:.4f}"), ("S var", "flint/reward_var", "{:.5f}"),
        ("subs", "flint/submissions", "{}"), ("failed", "flint/failed", "{}"),
        ("probe chg", "flint/probe_change_mean_abs", "{:.2e}"), ("probe rep", "flint/probe_repeat_max_abs", "{:.1e}"),
        ("KL(r||t)", "train/train_rollout_kl", "{:.4f}"), ("|dlogp|", "train/train_rollout_logprob_abs_diff", "{:.4f}"),
        ("sc corr", "train/sc_correction", "{:.4f}"), ("sc tail", "train/sc_tail_ratio", "{:.3f}"),
        ("loss", "train/loss", "{:.4f}"), ("grad", "train/grad_norm", "{:.3f}"),
        ("R3 pick", "train/r3_pick_mismatch", "{:.4f}"), ("R3 tok", "train/r3_token_mismatch", "{:.4f}"),
        ("rollout s", "flint/rollout_s", "{:.0f}"), ("train s", "perf/actor_train_time", "{:.0f}"),
        ("logp s", "perf/log_probs_time", "{:.0f}"), ("sync s", "perf/update_weights_time", "{:.0f}"),
        ("step s", "perf/step_time", "{:.0f}"), ("mem GB", "flint/job_mem_gb", "{:.0f}"), ("peak GB", "flint/job_mem_peak_gb", "{:.0f}")]
print("step " + " ".join(f"{c[0]:>10}" for c in COLS))
for i in sorted(steps):
    row = []
    for _, key, fmt in COLS:
        v = steps[i].get(key)
        row.append(f"{'-' if v is None else fmt.format(v):>10}"[-10:] if not isinstance(v, list) else f"{str(v):>10}")
    print(f"{i:>4} " + " ".join(row))
