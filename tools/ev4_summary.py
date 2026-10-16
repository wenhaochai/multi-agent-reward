"""ev4 summary (owner 2026-10-13: re-score the final oracle checkpoints with 4 episodes per val problem, every episode
traced; interactive problems skipped, owner 2026-10-08). For each arm, the eval episodes of its ev4 run and its ev4fill
run (the classic problems a timed-out ev4 job did not finish), restricted to the classic val problems
(data/fcs_val98_fo_<game>.jsonl labels) and to the first 4 episodes of each problem:
  score     100 x mean over problems of the per-problem mean V (V = the episode's scored submission: team, the lead's
            final; seq, round 5; par, the arm's own value), with the standard error over problems
  paired    each arm minus the bonus arm on the problems both scored (mean difference and its standard error)
  par       best of the first k attempts, k = 1..12 (the r12 run): one agent at k judge calls
Writes ev4_summary.json next to this script's output dir and prints a table.
  python3 tools/ev4_summary.py OUT_JSON"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

B = Path("/scratch/gpfs/GROUP/USER/project/miles-q38-build")
N = 4
ARMS = [("shared", "team", ["fcs_evalonly_ev4_team_shared_q9b-fcs-sft_s42", "fcs_evalonly_ev4fill_team_shared_q9b-fcs-sft_s42"]),
        ("bonus", "team", ["fcs_evalonly_ev4_team_bonus_q9b-fcs-sft_s42", "fcs_evalonly_ev4fill_team_bonus_q9b-fcs-sft_s42"]),
        ("difference, 1 sample", "team", ["fcs_evalonly_ev4_team_diff_q9b-fcs-sft_s42"]),
        ("difference, 4 samples", "team", ["fcs_evalonly_ev4_team_diff_q9b-fcs-sft_s42_cfk4"]),
        ("5 revisions", "seq", ["fcs_evalonly_ev4_seq_shared_q9b-fcs-sft_s42", "fcs_evalonly_ev4fill_seq_shared_q9b-fcs-sft_s42"]),
        ("best of 12 (r12)", "par", ["fcs_evalonly_ev4r12_par_indiv_q9b-fcs-sft_s42"])]


def classic(game):
    return {json.loads(l)["label"] for l in (B / f"data/fcs_val98_fo_{game}.jsonl").read_text().splitlines() if l.strip()}


def episodes(runs, labels):
    ep = defaultdict(list)
    for run in runs:
        for f in sorted((B / "runs" / run / "traces").glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r.get("eval") and r["label"] in labels and len(ep[r["label"]]) < N:
                    ep[r["label"]].append(r)
    return ep


def mean_se(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return 100 * m, 100 * sd / math.sqrt(len(xs))


out, per = {}, {}
for name, game, runs in ARMS:
    labels = classic(game)
    ep = episodes(runs, labels)
    per[name] = {k: sum(e["V"] for e in v) / len(v) for k, v in ep.items()}
    m, se = mean_se(list(per[name].values()))
    full = sum(len(v) == N for v in ep.values())
    out[name] = {"score": m, "se": se, "problems": len(ep), "problems_with_4": full, "episodes": sum(map(len, ep.values())),
                 "classic_labels": len(labels)}
    if game == "par":
        out[name]["best_at"] = {k: mean_se([sum(max(t["score"] or 0 for t in e["turns"][:k]) for e in v) / len(v)
                                            for v in ep.values()])[0] for k in range(1, 13)}
ref = per["bonus"]
for name in per:
    common = sorted(set(per[name]) & set(ref))
    d = [per[name][k] - ref[k] for k in common]
    out[name]["minus_bonus"], out[name]["minus_bonus_se"] = mean_se(d) if len(d) > 1 else (0.0, 0.0)
for name, o in out.items():
    print(f"{name:24s} {o['score']:5.1f} +- {o['se']:.1f}   problems {o['problems']}/{o['classic_labels']} "
          f"(4 episodes: {o['problems_with_4']}), episodes {o['episodes']}, minus bonus {o['minus_bonus']:+.1f} +- {o['minus_bonus_se']:.1f}")
    if "best_at" in o:
        print("   best of k:", " ".join(f"{k}:{v:.1f}" for k, v in o["best_at"].items()))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
