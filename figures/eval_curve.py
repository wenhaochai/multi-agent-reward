"""AIME avg@32 learning curve of our FlashREINFORCE reproduction vs the paper's reported points."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from style import apply_style, G_BLUE, REF_GREY, REF_DASH, twotone, header_legend, finalize_headers

HERE = Path(__file__).resolve().parent
apply_style()

df = pd.read_csv(HERE.parent / "runs/fr_r1d_1p5b/scalars.csv")
a24 = df[df.tag == "eval/eval_aime_2024_pass1"].sort_values("step")
a25 = df[df.tag == "eval/eval_aime_2025_pass1"].sort_values("step")
m = a24.merge(a25, on="step", suffixes=("_24", "_25"))
m["mean"] = 100 * (m.value_24 + m.value_25) / 2
m["v24"], m["v25"] = 100 * m.value_24, 100 * m.value_25
last_step = int(m.step.max())

# FlashREINFORCE paper, Tab. 9/10 (Sanity-Test-R1D-1.5B, same recipe): step -> (AIME24, AIME25)
paper_trust = {0: (22.71, 20.63), 1024: (28.13, 23.33), 2048: (31.98, 26.35), 3072: (32.08, 27.92),
               3968: (35.42, 26.25), 4992: (36.15, 28.33), 5900: (38.85, 28.54)}
paper_notrust = {0: (22.71, 20.42), 1024: (27.60, 24.06), 2048: (28.85, 25.63), 3072: (33.13, 26.46),
                 3968: (34.58, 26.77), 4992: (33.85, 27.40), 5900: (33.33, 27.81)}
ps = sorted(paper_trust)
pt_mean = [sum(paper_trust[s]) / 2 for s in ps]
pn_mean = [sum(paper_notrust[s]) / 2 for s in ps]

dark, light = twotone(G_BLUE)
fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)

# Left: mean of the two sets, ours vs the two paper arms
axl.plot(m.step, m["mean"], color=dark, linewidth=1.6, marker="o", markersize=3.2,
         markeredgecolor="white", markeredgewidth=0.6)
axl.plot(ps, pt_mean, color=REF_GREY, linestyle=REF_DASH, linewidth=1.3, marker="s", markersize=4,
         markeredgecolor="white", markeredgewidth=0.6)
axl.plot(ps, pn_mean, color=light, linestyle=REF_DASH, linewidth=1.3, marker="^", markersize=4.5,
         markeredgecolor="white", markeredgewidth=0.6)
axl.set_title("Mean of AIME24 and AIME25")
axl.set_xlabel("Update step")
axl.set_ylabel("avg@32 (%)")
axl.set_xlim(0, 6000)
header_legend(axl, [("Ours", dark, "-"), ("Paper, trust", REF_GREY, "--"), ("Paper, IS only", light, "--")])

# Right: per set, ours vs the paper's with-trust arm
axr.plot(m.step, m.v24, color=dark, linewidth=1.6)
axr.plot(m.step, m.v25, color=light, linewidth=1.6)
axr.plot(ps, [paper_trust[s][0] for s in ps], color=REF_GREY, linestyle=REF_DASH, linewidth=1.2,
         marker="s", markersize=4, markeredgecolor="white", markeredgewidth=0.6)
axr.plot(ps, [paper_trust[s][1] for s in ps], color=REF_GREY, linestyle=REF_DASH, linewidth=1.2,
         marker="^", markersize=4.5, markeredgecolor="white", markeredgewidth=0.6)
axr.set_title("Per set, vs paper with trust")
axr.set_xlabel("Update step")
axr.set_xlim(0, 6000)
header_legend(axr, [("AIME24", dark, "-"), ("AIME25", light, "-"), ("Paper 24", REF_GREY, "s"), ("Paper 25", REF_GREY, "^")])

fig.suptitle(f"FlashREINFORCE on DeepSeek-R1-Distill-Qwen-1.5B: reproduction through step {last_step}",
             x=0.01, ha="left", fontweight="bold", color="#1a1a1a")
finalize_headers(fig)
fig.savefig(HERE / "eval_curve.pdf")
fig.savefig(HERE / "eval_curve.png", dpi=200)
m[["step", "v24", "v25", "mean"]].round(2).to_csv(HERE / "eval_points.csv", index=False)
print("points", len(m), "last", last_step, "last mean %.2f" % m["mean"].iloc[-1])
