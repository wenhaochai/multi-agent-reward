"""Training curves of the three Qwen3.8-27B LoRA arms of the Q38 team study (logs/peer_team_lead.yaml): the share of
training episodes answered correctly and the share in which the lead delegated, per training step.

Data: training.csv (extract.py), from the "Global step N" lines of every training segment's log
(logs/molt-fr-fr_peer_team_q38_<arm>_sc-<jid>.out); per episode: correct = tm_task_w / tm_inv_sessions, delegating =
tm_delegated_w / tm_inv_sessions (32 episodes a step). Each line is a centered 25-step rolling mean (partial windows
at the ends). A step re-run after a resume takes the later segment. The solo lead has no teammates, so it never
delegates.
Output: training.pdf and training.png, WIDTH_POST wide (a 1600 px PNG).
"""
import csv
from collections import defaultdict
from pathlib import Path

from style import *

HERE = Path(__file__).resolve().parent
apply_style()

ARMS = [('lead', 'Team, bonus', BLUE), ('nob', 'Team, no bonus', LIGHT), ('solo', 'Solo', GREY)]
WINDOW = 25

data = defaultdict(dict)
for r in csv.DictReader(open(HERE / 'training.csv')):
    data[r['arm']][int(r['step'])] = (100 * float(r['correct']), 100 * float(r['delegating']))


def rolling(steps, ys, w=WINDOW):
    h = w // 2
    return [sum(ys[max(0, i - h):i + h + 1]) / len(ys[max(0, i - h):i + h + 1]) for i in range(len(ys))]


fig, axes = canvas(rows=1, cols=2, width=WIDTH_POST, panel_height=1.5,
                   title='Training episodes by step (draft title)',
                   legend=[(name, color) for _, name, color in ARMS],
                   quantity='Share of training episodes (%)', xlabel='Training step',
                   title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
last = max(max(d) for d in data.values())
for col, (ax, panel) in enumerate(zip(axes[0], ['Correct', 'Delegating'])):
    vals = []
    for arm, name, color in ARMS[::-1]:
        steps = sorted(data[arm])
        ys = rolling(steps, [data[arm][s][col] for s in steps])
        vals += ys
        ax.plot(steps, ys, color=color, zorder=3)
    ticks = list(range(0, last + 1, 100))
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0, last)
    nice_y(ax, min(vals), max(vals), zero=(panel == 'Delegating'), fmt='{:.0f}')
    room(ax)
    panel_label(ax, panel)
save(fig, HERE / 'training')
