"""Held-out pass@1 of the three Qwen3.8-27B LoRA arms of the Q38 team study (logs/peer_team_lead.yaml) at the
training steps of the pre-registered PRIMARY (100, 150, 200).

Data: eval_points.csv (extract.py), from the offline eval-only jobs runs/fr_peer_team_q38_<arm>_sc_ev<step>: 153
problems (AIME 2024/2025/2026, HMMT Feb 2025/2026) x 4 samples at T 1.0; pooled pass@1 = mean over the five
competitions of each competition's mean correctness. Arms: team with the 0.1 collaboration bonus (lead), team without
it (nob), the same lead alone (solo).
Output: heldout.pdf and heldout.png, WIDTH_POST wide (a 1600 px PNG).
"""
import csv
from pathlib import Path

from style import *

HERE = Path(__file__).resolve().parent
apply_style()

ARMS = [('lead', 'Team, bonus', BLUE), ('nob', 'Team, no bonus', LIGHT), ('solo', 'Solo', GREY)]
STEPS = [100, 150, 200]

pts = {(r['arm'], int(r['step'])): 100 * float(r['pooled']) for r in csv.DictReader(open(HERE / 'eval_points.csv'))}

fig, axes = canvas(rows=1, cols=1, width=WIDTH_POST, panel_height=1.6,
                   title='Held-out pass@1 by training step (draft title)',
                   legend=[(name, color) for _, name, color in ARMS],
                   quantity='Pass@1 (%)', xlabel='Training step',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
ax = axes[0, 0]
vals = []
for arm, name, color in ARMS[::-1]:
    ys = [pts[(arm, s)] for s in STEPS]
    vals += ys
    ax.plot(STEPS, ys, color=color, zorder=3)
    dots(ax, STEPS, ys, color, size=16, zorder=4)
ax.set_xticks(STEPS, [str(s) for s in STEPS])
ax.set_xlim(STEPS[0], STEPS[-1])
nice_y(ax, min(vals), max(vals), fmt='{:.1f}')
room(ax)
save(fig, HERE / 'heldout')
