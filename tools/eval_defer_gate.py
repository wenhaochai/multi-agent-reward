"""Gate for eval judged after training resumes (miles_team.fcs_rm.log_eval), on the solo smoke of
tools/eval_defer_smoke.sbatch (4 rollouts, evals after rollouts 1 and 3, the eval set with fcs_defer_score):
  1. the smoke ran the candidate code;
  2. eval 1 was saved and judged in the background: its "metrics.py - eval 1:" line comes after training went on
     (a "rollout 2" line), and eval 3 (the last) was logged too;
  3. nothing is left pending; each kept eval has every sample, its mean reward equals the logged eval/fcs_val, and an
     unanswered reply (cut off, or no </think>) has reward 0;
  4. judging eval 1's replies again here, alone, gives the same reward for >= 97% of them and a mean within 0.01 (the
     background judge ran beside training's; a time-limit-borderline program may flip).
  python3 tools/eval_defer_gate.py SMOKE_LOG RUN_DIR CANDIDATE_SHA   (inside miles.sif)"""
import ast
import glob
import gzip
import json
import os
import re
import sys

log, run, cand = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.environ.get("MQB_CODE", "/scratch/gpfs/GROUP/USER/project/miles-q38-build"))
fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


lines = open(log, errors="replace").read().splitlines()
code = next((l for l in lines if "[miles-run] code:" in l), "")
check(f"repo {cand[:7]}" in code, f"the smoke ran the candidate: {code[-80:]}")


def first(pat):
    return next((i for i, l in enumerate(lines) if re.search(pat, l)), None)


saved = first(r"\[fcs_rm\] eval 1: \d+ replies saved")
roll2 = first(r"log_utils\.py:\d+ - rollout 2:")
ev1, ev3 = first(r"metrics\.py:\d+ - eval 1: \{"), first(r"metrics\.py:\d+ - eval 3: \{")
check(saved is not None, "eval 1 saved for judging in the background")
check(None not in (saved, roll2, ev1) and saved < roll2 < ev1, f"training went on before eval 1 was logged (lines {saved}, {roll2}, {ev1})")
check(ev3 is not None, "eval 3 (the last) logged")
check(not glob.glob(f"{run}/eval_pending/*.pt"), "nothing left pending")
logged = {}
for i in (ev1, ev3):
    if i is not None:
        k = int(re.search(r"- eval (\d+):", lines[i]).group(1))
        logged[k] = ast.literal_eval(re.sub(r"(?<![\w'\"])(-?)(nan|inf)\b", r"'\1\2'", lines[i].split(f"- eval {k}: ", 1)[1]))
from miles_team.fcs_rm import answered, score  # noqa: E402

for k in (1, 3):
    f = f"{run}/eval_pending/eval_{k}.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(f, "rt")] if os.path.exists(f) else []
    check(len(rows) == 172, f"eval {k}: {len(rows)} kept replies")
    if rows and k in logged:
        m = sum(r["reward"] for r in rows) / len(rows)
        check(abs(m - logged[k]["eval/fcs_val"]) < 1e-9, f"eval {k}: kept mean {m:.6f} == logged {logged[k]['eval/fcs_val']:.6f}")
        bad = [r for r in rows if not answered(r["response"], r["status"] == "truncated") and r["reward"] != 0]
        check(not bad, f"eval {k}: unanswered replies all 0 ({sum(not answered(r['response'], r['status'] == 'truncated') for r in rows)} of them)")
    if k == 1 and rows:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(56) as ex:
            again = list(ex.map(lambda r: score(r["response"], r["label"]) if answered(r["response"], r["status"] == "truncated") else 0.0, rows))
        same = sum(abs(a - r["reward"]) < 1e-9 for a, r in zip(again, rows)) / len(rows)
        dm = abs(sum(again) - sum(r["reward"] for r in rows)) / len(rows)
        check(same >= 0.97 and dm <= 0.01, f"eval 1 judged again alone: {same:.1%} same reward, mean diff {dm:.4f}")
print("GATE", "FAIL: " + "; ".join(fails) if fails else "PASS", flush=True)
sys.exit(1 if fails else 0)
