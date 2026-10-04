"""Score centering inside FlashREINFORCE (R1-Distill-Qwen-1.5B): held-out AIME and the train/inference mismatch for
the paper recipe (main line), score centering with weight one (SC), and weight one without centering (gate-only
control).

Source (console logs, the same parse as WS/sc_vs_main.py):
  SC        WS/logs/molt-fr-fr_sc_r1d_1p5b_s3-*.out
  gate-only WS/logs/molt-fr-fr_gateonly_r1d_1p5b-*.out
  main      WS/runs/fr_r1d_1p5b/console.log
  - 'Eval at step N: {...}' -> mean of eval_aime_2024_pass1 and eval_aime_2025_pass1 (avg@32), x100
  - 'Global step N: {...}' -> vllm_kl (trainer vs vLLM KL) and is_filter_ratio (share of sequences the trust region
    gates); later segments overwrite redone steps; plotted as 64-step trailing means
Output: sc_vs_main.pdf / .png
"""
import re
import glob
from pathlib import Path

import numpy as np

from style import *

HERE = Path(__file__).parent
W = '/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace'
RUNS = [('main', 'Paper recipe', GREY, [f'{W}/runs/fr_r1d_1p5b/console.log']),
        ('sc', 'Score centering', BLUE, sorted(glob.glob(f'{W}/logs/molt-fr-fr_sc_r1d_1p5b_s3-*.out'))),
        ('ctrl', 'Weight one, no centering', LIGHT, sorted(glob.glob(f'{W}/logs/molt-fr-fr_gateonly_r1d_1p5b-*.out')))]
WIN = 64
NUM = r'(-?[0-9.]+(?:e-?[0-9]+)?)'


def parse(paths):
    ev, tr = {}, {}
    for p in paths:
        txt = open(p, errors='replace').read()
        for m in re.finditer(r'Eval at step (\d+): \{([^}]*)\}', txt):
            a = re.search(r"'eval_aime_2024_pass1': " + NUM, m.group(2))
            b = re.search(r"'eval_aime_2025_pass1': " + NUM, m.group(2))
            if a and b:
                ev[int(m.group(1))] = 50 * (float(a.group(1)) + float(b.group(1)))
        for m in re.finditer(r'Global step (\d+): \{([^}]*)\}', txt):
            k = re.search(r"'vllm_kl': " + NUM, m.group(2))
            g = re.search(r"'is_filter_ratio': " + NUM, m.group(2))
            if k and g:
                tr[int(m.group(1))] = (float(k.group(1)), float(g.group(1)))
    return ev, tr


def trailing(steps, vals, win=WIN):
    v = np.array(vals, dtype=float)
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = [(c[i + 1] - c[max(0, i + 1 - win)]) / (i + 1 - max(0, i + 1 - win)) for i in range(len(v))]
    return np.array(steps), np.array(out)


data = {key: parse(paths) for key, _, _, paths in RUNS}
for key, name, _, paths in RUNS:
    ev, tr = data[key]
    print(key, 'files', len(paths), 'evals', len(ev), 'last eval', max(ev) if ev else None,
          'train steps', len(tr), 'last', max(tr) if tr else None)

apply_style()
fig, axes = canvas(3, 1, width=WIDTH_POST, panel_height=1.15, title='Score centering in FlashREINFORCE',
                   legend=[(name, c) for _, name, c, _ in RUNS], title_pt=9.2, tick_pt=7.35)
axs = [row[0] for row in axes]
xmax = 0
for key, name, c, _ in RUNS:
    ev, tr = data[key]
    s = sorted(ev)
    axs[0].plot(s, [ev[i] for i in s], color=c, lw=1.3)
    st = sorted(tr)
    x, k = trailing(st, [1e3 * tr[i][0] for i in st])
    axs[1].plot(x, k, color=c, lw=1.2)
    x, g = trailing(st, [100 * tr[i][1] for i in st])
    axs[2].plot(x, g, color=c, lw=1.2)
    xmax = max([xmax] + st)
evs = [v for ev, _ in data.values() for v in ev.values()]
nice_y(axs[0], min(evs), max(evs))
panel_label(axs[0], 'Held-out AIME24/25 avg@32 (%)')
ks = [1e3 * v[0] for _, tr in data.values() for v in tr.values()]
kmax = max(trailing(sorted(tr), [1e3 * tr[i][0] for i in sorted(tr)])[1].max() for _, tr in data.values())
nice_y(axs[1], 0, float(kmax), zero=True, fmt='{:,.0f}')
panel_label(axs[1], 'Train vs rollout KL (x1e-3), 64-step mean')
gs = [100 * v[1] for _, tr in data.values() for v in tr.values()]
gmax = max(trailing(sorted(tr), [100 * tr[i][1] for i in sorted(tr)])[1].max() for _, tr in data.values())
nice_y(axs[2], 0, float(gmax), zero=True)
panel_label(axs[2], 'Sequences gated (%), 64-step mean')
for a in axs:
    a.set_xlim(0, xmax * 1.02)
    room(a)
save(fig, HERE / 'sc_vs_main')
