"""Judge parity on Frontier-CS algorithmic: scores of the same 150 solutions from the official judge run on one Della
node (x) against our local judge on that node (left) and against the official January 2026 batch scores (right).

Data: parity_same_node.jsonl (copy of miles-q38-build/logs/fcs_parity2.jsonl, job 14850359): 150 official-batch
solutions whose solution file and problem directory hash-match the local copies, stratified by interactive x
{0, partial, 100}; official judge = Frontier-CS go-judge v1.11.1 + node orchestrator on the host; local judge =
miles_team/fcs_judge.py; both at 4 concurrent submissions, one after the other. Scores are 0-100 per solution.
Output: parity.pdf and parity.png, WIDTH_POST wide (a 1600 px PNG).
"""
import json
from pathlib import Path

from style import *

HERE = Path(__file__).resolve().parent
apply_style()

rows = [json.loads(l) for l in open(HERE / 'parity_same_node.jsonl')]
fig, axes = canvas(rows=1, cols=2, width=WIDTH_POST, panel_height=1.7,
                   title='Judge scores on the same solutions (draft title)',
                   legend=[('Classic', BLUE, 'dot'), ('Interactive', LIGHT, 'dot')],
                   quantity='Score (0-100)', xlabel='Official judge, same node',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
for ax, key, name in [(axes[0, 0], 'local', 'Our judge'), (axes[0, 1], 'jan', 'Official, January batch')]:
    ax.plot([0, 100], [0, 100], color=GREY_300, lw=0.8, zorder=1)
    for inter, color in [(False, BLUE), (True, LIGHT)]:
        rs = [r for r in rows if r['interactive'] == inter]
        dots(ax, [r['official_here'] for r in rs], [r[key] for r in rs], color, size=12, zorder=3)
    ax.set_xlim(-4, 108)
    ax.set_xticks([0, 50, 100], ['0', '50', '100'])
    nice_y(ax, 0, 103, zero=True, headroom=0.12, fmt='{:.0f}')
    room(ax)
    panel_label(ax, name)
save(fig, HERE / 'parity')
