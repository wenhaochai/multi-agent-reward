"""Blind team game (no feedback) on Frontier-CS: val score before training and at rollout 19 (10 actor steps), per arm.

Source: miles-q38-build/logs/miles-fcs_team_<arm>_q9b-fcs-sft_s42-<jid>.out, '[fcs_team] eval N <set>: {...}' lines:
  eval/fcs_val/ft_lead_score (team game, the lead's final program) and eval/fcs_val_solo/ft_lead_score (one agent,
  the solo prompt), x 100. Before training = the mean of the eval-0 lines present (the same frozen weights in every arm).
Output: blind_eval19.pdf / .png
"""
import json
import re
from pathlib import Path

from style import *

HERE = Path(__file__).parent
LOGS = Path('/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
ARMS = [('solocm', 'Single agent'), ('shared', 'Shared'), ('indiv', 'Individual'),
        ('mix50', 'Half and half')]
SETS = [('fcs_val', 'Team game, lead final (val score)'), ('fcs_val_solo', 'One agent alone (val score)')]


def evals(arm):
    out = {}
    for f in sorted(LOGS.glob(f'miles-fcs_team_{arm}_q9b-fcs-sft_s42-*.out')):
        for line in f.read_text(errors='replace').splitlines():
            m = re.search(r'\[fcs_team\] eval (\d+) (fcs_val\w*): (\{.*\})', ANSI.sub('', line))
            if m:
                out[(int(m[1]), m[2])] = 100 * json.loads(m[3])[f'eval/{m[2]}/ft_lead_score']
    return out


E = {arm: evals(arm) for arm, _ in ARMS}
apply_style()
fig, axes = canvas(2, 1, width=WIDTH_POST, panel_height=1.25, title='Blind team game, before and after 10 actor steps',
                   legend=[('Before training', GREY), ('Rollout 19', BLUE)], title_pt=9.2, tick_pt=7.35)
for (st, label), row in zip(SETS, axes):
    ax = row[0]
    before = [E[a][(0, st)] for a, _ in ARMS if (0, st) in E[a]]
    b0 = sum(before) / len(before)
    names = ['Untrained'] + [n for _, n in ARMS]
    vals = [b0] + [E[a][(19, st)] for a, _ in ARMS]
    ax.bar(range(len(vals)), vals, color=[GREY] + [BLUE] * len(ARMS), width=0.6)
    ax.set_xticks(range(len(vals)), names)
    ax.grid(axis='x', visible=False)
    print(st, 'before', [round(v, 2) for v in before], 'values', [round(v, 2) for v in vals])
    nice_y(ax, 0, max(vals), zero=True)
    panel_label(ax, label)
    room(ax)
save(fig, HERE / 'blind_eval19')
