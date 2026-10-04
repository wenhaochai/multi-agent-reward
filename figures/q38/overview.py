"""The three Q38 curves in one row (logs/peer_team_lead.yaml): held-out pass@1 at the PRIMARY steps, and the share of
training episodes answered correctly and the share in which the lead delegated.

Data: eval_points.csv and training.csv (extract.py). Held-out: offline eval-only jobs, 153 problems (AIME 2024/2025/2026,
HMMT Feb 2025/2026) x 4 samples at T 1.0, pooled = mean over the five competitions of each one's mean correctness.
Training: per step (32 episodes) correct = tm_task_w / tm_inv_sessions, delegating = tm_delegated_w / tm_inv_sessions,
each a centered 25-step rolling mean (partial windows at the ends); a step re-run after a resume takes the later
segment. The solo lead has no teammates and never delegates.
Output: overview.pdf and overview.png, WIDTH_TEXT wide (a 1600 px PNG).
"""
import csv
from collections import defaultdict
from pathlib import Path

from style import *

HERE = Path(__file__).resolve().parent
apply_style()

ARMS = [('lead', 'Team, bonus', BLUE), ('nob', 'Team, no bonus', LIGHT), ('solo', 'Solo', GREY)]
STEPS = [100, 150, 200]
WINDOW = 25

pts = {(r['arm'], int(r['step'])): 100 * float(r['pooled']) for r in csv.DictReader(open(HERE / 'eval_points.csv'))}
train = defaultdict(dict)
for r in csv.DictReader(open(HERE / 'training.csv')):
    train[r['arm']][int(r['step'])] = (100 * float(r['correct']), 100 * float(r['delegating']))


def rolling(ys, w=WINDOW):
    h = w // 2
    return [sum(ys[max(0, i - h):i + h + 1]) / len(ys[max(0, i - h):i + h + 1]) for i in range(len(ys))]


fig, axes = canvas(rows=1, cols=3, width=WIDTH_TEXT, panel_height=1.5,
                   title='Held-out pass@1 and training episodes by step (draft title)',
                   legend=[(name, color) for _, name, color in ARMS],
                   quantity='Share (%)', xlabel='Training step')
ax0, ax1, ax2 = axes[0]

vals = []
for arm, name, color in ARMS[::-1]:
    ys = [pts[(arm, s)] for s in STEPS]
    vals += ys
    ax0.plot(STEPS, ys, color=color, zorder=3)
    dots(ax0, STEPS, ys, color, size=14, zorder=4)
ax0.set_xticks(STEPS, [str(s) for s in STEPS])
ax0.set_xlim(STEPS[0], STEPS[-1])
nice_y(ax0, min(vals), max(vals), fmt='{:.1f}')
room(ax0)
panel_label(ax0, 'Held-out pass@1')

last = max(max(d) for d in train.values())
for col, (ax, panel) in enumerate([(ax1, 'Training: correct'), (ax2, 'Training: delegating')]):
    vals = []
    for arm, name, color in ARMS[::-1]:
        steps = sorted(train[arm])
        ys = rolling([train[arm][s][col] for s in steps])
        vals += ys
        ax.plot(steps, ys, color=color, zorder=3)
    ticks = list(range(0, last + 1, 100))
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0, last)
    nice_y(ax, min(vals), max(vals), zero=(col == 1), fmt='{:.0f}')
    room(ax)
    panel_label(ax, panel)
save(fig, HERE / 'overview')
