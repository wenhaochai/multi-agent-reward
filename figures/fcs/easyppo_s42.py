"""EasyPPO seed 42 on Frontier-CS (miles, Qwen3.5-9B SFT init, TP2 run 14869367), critic-only phase (rollouts 0-29; the
run died of OOM at the first actor step, rollout 30): rollout score and truncation, critic value loss and value clip
fraction, and val score of the frozen actor.

Source: miles-q38-build/logs/miles-fcs_easyppo_q9b-fcs-sft_s42-14869367.out
  - 'perf N:' lines -> rollout/raw_reward_unfiltered (mean judge score / 100 over 16 x 32 samples), rollout/truncated_ratio
  - 'critic-step N:' lines (log_utils.py) -> train/critic-value_loss, train/critic-value_clipfrac (4 critic steps/rollout)
  - 'eval N:' lines (metrics.py) -> eval/fcs_val (mean judge score / 100 over val 172 x 5)
Output: easyppo_s42.pdf / .png
"""
import ast
import re
from pathlib import Path

from style import *

HERE = Path(__file__).parent
LOG = Path("/scratch/gpfs/GROUP/USER/project/miles-q38-build/logs/miles-fcs_easyppo_q9b-fcs-sft_s42-14869367.out")
ANSI = re.compile(r'\x1b\[[0-9;]*m')

perf, critic, evals = {}, {}, {}
for line in LOG.read_text(errors='replace').splitlines():
    line = ANSI.sub('', line)
    m = re.search(r'metrics\.py:93 - perf (\d+): (\{.*\})', line)
    if m:
        perf[int(m[1])] = ast.literal_eval(m[2])
    m = re.search(r"metrics\.py:\d+ - eval (\d+): \{.*?'eval/fcs_val': ([0-9.]+)", line)
    if m:
        evals[int(m[1])] = 100 * float(m[2])
    m = re.search(r'log_utils\.py:\d+ - critic-step (\d+): (\{.*\})', line)
    if m:
        critic[int(m[1])] = ast.literal_eval(m[2])

r = sorted(perf)
c = sorted(critic)
reward = [100 * perf[i]['rollout/raw_reward_unfiltered'] for i in r]
trunc = [100 * perf[i]['rollout/truncated_ratio'] for i in r]
vloss = [critic[i]['train/critic-value_loss'] for i in c]
vclip = [critic[i]['train/critic-value_clipfrac'] for i in c]
print('rollouts', len(r), 'critic steps', len(c), 'evals', {k: round(v, 2) for k, v in sorted(evals.items())})
print('reward', [round(x, 1) for x in reward])
print('trunc', [round(x, 1) for x in trunc])
print('vloss first/last', vloss[:3], vloss[-3:])
print('vclip first/last', vclip[:3], vclip[-3:])

apply_style()
fig, axes = canvas(4, 1, width=WIDTH_POST, panel_height=1.1, title='EasyPPO seed 42, critic-only phase',
                   legend=[('Mean score', BLUE), ('Truncated', LIGHT)], title_pt=9.2, tick_pt=7.35)
axs = [row[0] for row in axes]

a = axs[0]
a.plot(r, reward, color=BLUE, lw=1.4)
a.plot(r, trunc, color=LIGHT, lw=1.4)
nice_y(a, 0, max(trunc), zero=True)
room(a)
panel_label(a, 'Per rollout (%)')

a = axs[1]
a.plot(c, vloss, color=BLUE, lw=1.2)
nice_y(a, 0, max(vloss), zero=True, fmt='{:,.1f}')
room(a)
panel_label(a, 'Value loss per critic step')

a = axs[2]
a.plot(c, vclip, color=BLUE, lw=1.2)
nice_y(a, 0, 1, zero=True, fmt='{:,.1f}')
room(a)
panel_label(a, 'Value clip fraction per critic step')

a = axs[3]
ev = sorted(evals)
dots(a, ev, [evals[i] for i in ev], BLUE)
nice_y(a, 0, max(evals.values()), zero=True)
a.set_xlim(axs[0].get_xlim())
panel_label(a, 'Val score of the frozen actor (%)')

save(fig, HERE / 'easyppo_s42')
