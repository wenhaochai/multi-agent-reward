"""The two S2 steps from miles' train dumps (step_job.sh: --save-debug-train-data train_<rollout>_<rank>.pt): per
sample, its reward and advantage, the trainer's mean log-prob of the generated tokens before step 1 and before step 2
(the actor's forward-only pass of each rollout, on the same batch), and the trainer-sampler gap before step 1 (mean
|log_probs - rollout_log_probs|: TE attention, grouped experts and the fused head against SGLang 0.5.16).
Usage: analyze_step.py OUT_DIR"""
import glob
import sys

import torch

out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{out}/train_*_*.pt")):
    d = torch.load(f, weights_only=False)
    rd, rid = d["rollout_data"], d["rollout_id"]
    for i in range(len(rd["log_probs"])):
        key = (d["rank"], i)
        m = torch.as_tensor(rd["loss_masks"][i]).bool()
        lp = torch.as_tensor(rd["log_probs"][i]).float()
        r = rows.setdefault(key, {"reward": float(torch.as_tensor(rd["rewards"][i]).float().mean()) if "rewards" in rd else None,
                                  "adv": float(torch.as_tensor(rd["advantages"][i]).float()[m].mean()), "n": int(m.sum())})
        r[f"lp{rid}"] = float(lp[m].mean())
        if rid == 0:
            ro = torch.as_tensor(rd["rollout_log_probs"][i]).float()
            r["gap"] = float((lp - ro)[m].abs().mean())
            r["gap_signed"] = float((lp - ro)[m].mean())
print(f"{'rank/i':8s} {'reward':>7s} {'adv':>7s} {'tokens':>7s} {'lp_r0':>9s} {'lp_r1':>9s} {'dlp':>10s} {'|train-sampler|':>16s}")
agree = 0
for (rank, i), r in sorted(rows.items()):
    d = r.get("lp1", float("nan")) - r["lp0"]
    agree += (d > 0) == (r["adv"] > 0)
    print(f"{rank}/{i:<6d} {r['reward'] if r['reward'] is not None else float('nan'):7.3f} {r['adv']:7.3f} {r['n']:7d} "
          f"{r['lp0']:9.5f} {r.get('lp1', float('nan')):9.5f} {d:10.2e} {r['gap']:9.4f} ({r['gap_signed']:+.4f})")
print(f"samples whose generated-token log-prob moved with the sign of their advantage: {agree} of {len(rows)}")
