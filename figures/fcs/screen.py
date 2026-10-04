"""G1 reward screen on Frontier-CS (miles, EasyPPO, Qwen3.5-9B SFT init): per-rollout training curves of the arms.

Source: miles-q38-build/logs/miles-fcs_team_<arm>_q9b-fcs-sft_s42-<jid>.out
  - '[fcs_team] rollout N: {...}' lines -> rollout/ft_lead_score (the lead's final score S; solocm: the single agent's
    score), rollout/ft_best_mate (best of the 3 teammates), rollout/ft_lead_bad (lead program fails to compile or has
    no code), means over the rollout's episodes (16 prompts x 8 episodes; solocm 16 x 32)
  - 'critic-step N:' lines (log_utils.py) -> train/critic-value_loss
Rollouts 0-9 are critic-only (actor frozen). Output: screen.pdf / .png
"""
import ast
import json
import re
from pathlib import Path

from style import *

HERE = Path(__file__).parent
LOGS = Path('/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
ARMS = [('solocm', 'Solo, compute-matched', GREY), ('shared', 'Shared', BLUE), ('indiv', 'Individual', LIGHT)]


def parse(arm):
    files = sorted(LOGS.glob(f'miles-fcs_team_{arm}_q9b-fcs-sft_s42-*.out'))
    ro, cr = {}, {}
    for f in files:
        for line in f.read_text(errors='replace').splitlines():
            line = ANSI.sub('', line)
            m = re.search(r'\[fcs_team\] rollout (\d+): (\{.*\})', line)
            if m:
                ro[int(m[1])] = json.loads(m[2])
            m = re.search(r'log_utils\.py:\d+ - critic-step (\d+): (\{.*\})', line)
            if m:
                cr[int(m[1])] = ast.literal_eval(m[2])
    return ro, cr


data = {arm: parse(arm) for arm, _, _ in ARMS}
for arm, (ro, cr) in data.items():
    print(arm, 'rollouts', len(ro), 'critic steps', len(cr),
          'S', [round(100 * ro[i]['rollout/ft_lead_score'], 1) for i in sorted(ro)])

apply_style()
fig, axes = canvas(3, 1, width=WIDTH_POST, panel_height=1.3, title='G1 reward screen, training rollouts',
                   legend=[(name, c) for _, name, c in ARMS], title_pt=9.2, tick_pt=7.35)
axs = [row[0] for row in axes]
top = 0
for arm, name, c in ARMS:
    ro, cr = data[arm]
    r = sorted(ro)
    y = [100 * ro[i]['rollout/ft_lead_score'] for i in r]
    axs[0].plot(r, y, color=c, lw=1.4)
    top = max([top] + y)
    if arm != 'solocm':
        axs[1].plot(r, [100 * ro[i]['rollout/ft_lead_bad'] for i in r], color=c, lw=1.4)
    s = sorted(cr)
    axs[2].plot(s, [cr[i]['train/critic-value_loss'] for i in s], color=c, lw=1.2)
nice_y(axs[0], 0, top, zero=True)
panel_label(axs[0], 'Final score per rollout (%)')
lb = [100 * v['rollout/ft_lead_bad'] for a, (ro, _) in data.items() if a != 'solocm' for v in ro.values()]
nice_y(axs[1], 0, max(lb), zero=True)
panel_label(axs[1], 'Lead program fails to compile (%)')
vl = [v['train/critic-value_loss'] for _, cr in data.values() for v in cr.values()]
nice_y(axs[2], 0, max(vl), zero=True, fmt='{:,.1f}')
panel_label(axs[2], 'Critic value loss per critic step')
for a in axs:
    room(a)
save(fig, HERE / 'screen')
