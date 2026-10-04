"""Eval curve vs paper arms, plus the sequence trust region's rejection rate and the train/inference KL over training."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from style import apply_style, G_BLUE, G_RED, G_GREEN, REF_GREY, REF_DASH, twotone, fig_header_legend, finalize_headers

HERE = Path(__file__).resolve().parent
apply_style()
df = pd.read_csv(HERE.parent / "runs/fr_r1d_1p5b/scalars.csv")
tags = set(df.tag)
need = ["eval/eval_aime_2024_pass1", "eval/eval_aime_2025_pass1", "train/is_filter_ratio", "train/vllm_kl"]
missing = [t for t in need if t not in tags]
assert not missing, f"missing tags {missing}; have {sorted(t for t in tags if 'filter' in t or 'kl' in t)}"

a24 = df[df.tag == need[0]].sort_values("step"); a25 = df[df.tag == need[1]].sort_values("step")
m = a24.merge(a25, on="step", suffixes=("_24", "_25")); m["mean"] = 100 * (m.value_24 + m.value_25) / 2
gate = df[df.tag == need[2]].sort_values("step"); kl = df[df.tag == need[3]].sort_values("step")
last_step = int(gate.step.max())
WIN = 64
gate_s = gate.value.rolling(WIN, center=True, min_periods=8).mean() * 100
kl_s = kl.value.rolling(WIN, center=True, min_periods=8).mean() * 1e3

paper_trust = {0: 21.67, 1024: 25.73, 2048: 29.17, 3072: 30.00, 3968: 30.83, 4992: 32.24, 5900: 33.70}
paper_is = {0: 21.56, 1024: 25.83, 2048: 27.24, 3072: 29.79, 3968: 30.68, 4992: 30.63, 5900: 30.57}
ps = sorted(paper_trust)

dark, light = twotone(G_BLUE)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.6, 3.4), constrained_layout=True)

ax1.plot(m.step, m["mean"], color=dark, linewidth=1.5, marker="o", markersize=2.8, markeredgecolor="white", markeredgewidth=0.5)
ax1.plot(ps, [paper_trust[s] for s in ps], color=REF_GREY, linestyle=REF_DASH, linewidth=1.2, marker="s", markersize=3.8, markeredgecolor="white", markeredgewidth=0.6)
ax1.plot(ps, [paper_is[s] for s in ps], color=light, linestyle=REF_DASH, linewidth=1.2, marker="^", markersize=4.2, markeredgecolor="white", markeredgewidth=0.6)
ax1.set_title("Eval vs paper")
ax1.set_xlabel("Update step"); ax1.set_ylabel("avg@32 (%)"); ax1.set_xlim(0, 6000); ax1.set_xticks([0, 2000, 4000, 6000])

gd, gl = twotone(G_GREEN)
ax2.scatter(gate.step, gate.value * 100, s=3, color=gl, alpha=0.25, linewidths=0)
ax2.plot(gate.step, gate_s, color=gd, linewidth=1.6)
ax2.set_title("Sequences rejected")
ax2.set_xlabel("Update step"); ax2.set_ylabel("sequences rejected (%)"); ax2.set_xlim(0, 6000); ax2.set_xticks([0, 2000, 4000, 6000])

kd, kl_light = twotone(G_RED)
ax3.scatter(kl.step, kl.value * 1e3, s=3, color=kl_light, alpha=0.25, linewidths=0)
ax3.plot(kl.step, kl_s, color=kd, linewidth=1.6)
ax3.set_title("Train vs rollout KL")
ax3.set_xlabel("Update step"); ax3.set_ylabel(r"vllm_kl ($\times 10^{-3}$)"); ax3.set_xlim(0, 6000); ax3.set_xticks([0, 2000, 4000, 6000])

fig_header_legend(fig, [("Ours", dark, "-"), ("Paper, trust", REF_GREY, "--"), ("Paper, IS only", light, "--"),
                        (f"Rejection rate, {WIN}-step mean", gd, "-"), (f"vllm_kl, {WIN}-step mean", kd, "-")])
finalize_headers(fig)
fig.savefig(HERE / "trust_region_dynamics.pdf"); fig.savefig(HERE / "trust_region_dynamics.png", dpi=200)
g = gate.groupby(pd.cut(gate.step, [0, 2000, 3000, 4000, 4500, 6000])).value.mean() * 100
print("gate %% by phase:\n", g.round(2).to_string()); print("last step", last_step)
