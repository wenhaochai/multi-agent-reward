"""ev4: classic validation score of each final checkpoint, 96 classic val problems x 4 episodes, every episode traced.
Data: ev4_summary.json from tools/ev4_summary.py (score = 100 x mean over problems of the per-problem mean V; error bar =
1 standard error over problems). Best of 5 is the r12 run's best of its first five attempts (best_at[5]); the r12 run
does not log a standard error for it, so that bar has none.
  python3 ev4_plot.py SUMMARY_JSON FIG_DIR   (FIG_DIR holds style.py and fonts/ from the writing:plot skill)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from style import *  # noqa: E402

apply_style()
S = json.load(open(sys.argv[1]))
rows = [("Shared", S["shared"]["score"], S["shared"]["se"], BLUE),
        ("Bonus", S["bonus"]["score"], S["bonus"]["se"], MEDIUM[1]),
        ("Difference, 1 sample", S["difference, 1 sample"]["score"], S["difference, 1 sample"]["se"], tone("yellow", 300)),
        ("Difference, 4 samples", S["difference, 4 samples"]["score"], S["difference, 4 samples"]["se"], tone("yellow", 600)),
        ("One agent, 5 revisions", S["5 revisions"]["score"], S["5 revisions"]["se"], GREY),
        ("One agent, best of 5", S["best of 12 (r12)"]["best_at"]["5"], None, GREY)]
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=2.0, title=None, quantity="Classic validation score (0-100)")
ax = axes.flat[0]
for i, (name, v, se, c) in enumerate(rows):
    ax.barh(i, v, color=c, height=0.62, zorder=3)
    if se is not None:
        ax.errorbar(v, i, xerr=se, color=INK, lw=0.8, capsize=2, zorder=4)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 30)
ax.grid(False, axis="y")
save(fig, Path(sys.argv[2]) / "ev4_classic")
print([(r[0], round(r[1], 1)) for r in rows])
