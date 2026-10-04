"""Held-out pass@1 of the three Qwen3.8-27B LoRA arms of the Q38 team study (logs/peer_team_lead.yaml) at every
evaluated training step: the PRIMARY steps 100/150/200 plus the secondary evals (0, 250, 300 for the teams; 0, 300,
350, 400 for solo; step-50 and solo step-250 checkpoints were rotated out before their evals ran).

Data: eval_points.csv (extract.py): 153 problems (AIME 2024/2025/2026, HMMT Feb 2025/2026) x 4 samples at T 1.0;
pooled pass@1 = mean over the five competitions of each competition's mean correctness. Step 0 is the untrained
model under each arm's protocol; the team arms share one step-0 eval (lead's protocol is nob's).
Output: heldout_all.pdf and heldout_all.png, WIDTH_POST wide (a 1600 px PNG).
"""
import csv
from collections import defaultdict
from pathlib import Path

from style import *

HERE = Path(__file__).resolve().parent
apply_style()

ARMS = [('lead', 'Team, bonus', BLUE), ('nob', 'Team, no bonus', LIGHT), ('solo', 'Solo', GREY)]
pts = defaultdict(dict)
for r in csv.DictReader(open(HERE / 'eval_points.csv')):
    pts[r['arm']][int(r['step'])] = 100 * float(r['pooled'])
pts['nob'].setdefault(0, pts['lead'][0])

fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=1.6,
                   title='Held-out pass@1 by training step (draft title)',
                   legend=[(name, color) for _, name, color in ARMS],
                   quantity='Pass@1 (%)', xlabel='Training step',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes[0, 0]
vals = []
for arm, name, color in ARMS[::-1]:
    steps = sorted(pts[arm])
    ys = [pts[arm][s] for s in steps]
    vals += ys
    ax.plot(steps, ys, color=color, zorder=3)
    dots(ax, steps, ys, color, size=16, zorder=4)
ticks = [0, 100, 200, 300, 400]
ax.set_xticks(ticks, [str(t) for t in ticks])
ax.set_xlim(0, 400)
nice_y(ax, min(vals), max(vals), fmt='{:.0f}')
room(ax)
save(fig, HERE / 'heldout_all')
