"""Training curve of the agentic single-agent RL run: per rollout, the mean judged score S of the training episodes and
the share of episodes that submitted, from the '[fcs_agent] rollout N: {...}' lines of runs/<run>/console.log (later
segments win). Rollouts 0-9 train only the critic (the actor is the SFT model).
  python3 agent_curve.py RUN FIG_DIR   (FIG_DIR holds style.py and fonts/ from the writing:plot skill)"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from style import *  # noqa: E402

log = Path(f"/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs/{sys.argv[1]}/console.log")
d = {}
for l in open(log, errors="replace"):
    m = re.search(r'rollout (\d+): (\{"rollout/fo_S".*\})', re.sub(r"\x1b\[[0-9;]*m", "", l))
    if m:
        try:
            d[int(m.group(1))] = json.loads(m.group(2))
        except ValueError:
            pass
x = sorted(d)
apply_style()
fig, axes = canvas(2, 1, width=WIDTH_POST, panel_height=1.4, title=None,
                   legend=[("Actor frozen (critic only)", GREY), ("Actor trained", BLUE)], xlabel="Rollout")
for ax, key, name, hi in [(axes.flat[0], "rollout/fo_ag_submitted", "Episodes that submit (%)", 60),
                          (axes.flat[1], "rollout/fo_S", "Training score (0-100)", 20)]:
    y = [100 * d[r][key] for r in x]
    ax.plot([r for r in x if r <= 10], [v for r, v in zip(x, y) if r <= 10], color=GREY, marker="o", ms=2)
    ax.plot([r for r in x if r >= 10], [v for r, v in zip(x, y) if r >= 10], color=BLUE, marker="o", ms=2)
    nice_y(ax, 0, hi, zero=True)
    room(ax)
    panel_label(ax, name)
save(fig, Path(sys.argv[2]) / "agent_solo_curve")
print(x[-1], round(100 * d[x[-1]]["rollout/fo_S"], 1))
