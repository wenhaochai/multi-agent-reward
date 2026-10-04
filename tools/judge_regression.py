"""Old vs new judge on real programs (audit 2026-10-04): re-judge a stratified sample of the SFT probe's programs with
miles_team/fcs_judge.py and miles_team/fcs_judge_next.py and compare scores and statuses. Expected differences only:
wall-clock heuristics (timing noise), val problem 23 (checker address space), interactive memory (peak RSS instead of
address space), training ratios > 1 (clamped). Run in miles.sif on a CPU node:
  python3 tools/judge_regression.py [n_parallel]   -> runs/judge_regression.jsonl + a summary
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
import miles_team.fcs_judge as OLD  # noqa: E402
import miles_team.fcs_judge_next as NEW  # noqa: E402
from miles_team.fcs_rm import extract_cpp  # noqa: E402

P = B / "runs" / "fcs_probe_qwen35_9b_sft"
gen = {json.loads(l)["key"]: json.loads(l) for l in open(P / "gen.jsonl")}
sc = {json.loads(l)["key"]: json.loads(l) for l in open(P / "scores.jsonl")}
rng = random.Random(0)
rows = [dict(gen[k], rec_score=s["score"], rec_status=s["status"]) for k, s in sc.items() if k in gen]
inter = {r["label"]: OLD.load_problem(Path(r["label"]))["interactive"] for r in rows}


def pick(cond, n):
    c = [r for r in rows if cond(r)]
    rng.shuffle(c)
    return c[:n]


sample = (pick(lambda r: r["split"] == "val" and r["rec_status"] == "done" and r["rec_score"] > 0 and not inter[r["label"]], 12)
          + pick(lambda r: r["split"] == "val" and r["rec_status"] == "done" and r["rec_score"] > 0 and inter[r["label"]], 12)
          + pick(lambda r: r["split"] == "val" and r["rec_status"] == "done" and r["rec_score"] == 0, 8)
          + pick(lambda r: r["split"] == "train" and r["rec_status"] == "done" and r["rec_score"] > 0, 12)
          + pick(lambda r: r["rec_status"] == "compile error", 8))


def one(r):
    code = extract_cpp(r["response"])
    o = OLD.judge(r["label"], code, case_workers=4)
    n = NEW.judge(r["label"], code, case_workers=4)
    return {"key": r["key"], "label": r["label"], "interactive": inter[r["label"]], "rec": r["rec_score"],
            "rec_status": r["rec_status"], "old": o["score"], "old_status": o["status"], "new": n["score"],
            "new_status": n["status"], "new_infra": n.get("infra")}


with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
    res = list(ex.map(one, sample))
with open(B / "runs" / "judge_regression.jsonl", "w") as f:
    for x in res:
        f.write(json.dumps(x) + "\n")
same_status = sum(x["old_status"] == x["new_status"] for x in res)
diffs = sorted(((abs(x["new"] - x["old"]), x) for x in res), key=lambda t: -t[0])
print(f"programs {len(res)}  same status {same_status}  |new-old| mean {sum(d for d, _ in diffs) / len(res):.3f}"
      f"  exact {sum(d < 1e-9 for d, _ in diffs)}  infra {sum(bool(x['new_infra']) for x in res)}")
print("old vs recorded |diff| mean", round(sum(abs(x["old"] - 100 * x["rec"]) for x in res) / len(res), 3))
for d, x in diffs[:12]:
    if d > 1e-9 or x["old_status"] != x["new_status"]:
        print(f"  {x['key']:<28} inter={x['interactive']!s:<5} rec {100 * x['rec']:6.2f} old {x['old']:6.2f} "
              f"({x['old_status']}) new {x['new']:6.2f} ({x['new_status']})")
