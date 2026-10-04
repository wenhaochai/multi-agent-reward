"""Outcomes of base Qwen3.5-9B samples on the Frontier-CS study data at EasyPPO's eval settings (T 1.0, top-p 1.0,
32768 new tokens; train 200 problems x 4 samples, val 172 x 5), judged by miles_team/fcs_judge.py.

Data: probe_outcomes.csv (probe_extract.py) from miles-q38-build/runs/fcs_probe_qwen35_9b*/ (jobs 14849877 thinking,
14855561 no thinking, 14862729 EasyPPO SFT init HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1, thinking). Shares of samples: truncated (finish = length), finished but no compilable code, compiled and
scored 0, scored above 0. Output: probe.pdf and probe.png, WIDTH_POST wide (a 1600 px PNG).
"""
import csv
from pathlib import Path

from style import *

HERE = Path(__file__).resolve().parent
apply_style()

rows = list(csv.DictReader(open(HERE / 'probe_outcomes.csv')))
PARTS = [('truncated', 'Truncated', MEDIUM[0]), ('compile_error', 'No compilable code', MEDIUM[1]),
         ('zero', 'Scored 0', GREY_400), ('positive', 'Scored above 0', MEDIUM[3])]
probes = list(dict.fromkeys(r['probe'] for r in rows))
fig, axes = canvas(rows=1, cols=len(probes), width=WIDTH_TEXT, panel_height=1.4,
                   title='Qwen3.5-9B samples by outcome (draft title)',
                   legend=[(name, c, 'box') for _, name, c in PARTS],
                   quantity='Share of samples (%)', title_pt=9.2, tick_pt=TEXT_PT, note_pt=TICK_PT, side=0.10)
for ax, probe in zip(axes[0], probes):
    rs = [r for r in rows if r['probe'] == probe]
    xs = range(len(rs))
    bottom = [0.0] * len(rs)
    for key, _, c in PARTS:
        v = [100 * float(r[key]) for r in rs]
        ax.bar(xs, v, bottom=bottom, color=c, width=0.6, edgecolor='white', linewidth=0.6, zorder=3)
        bottom = [b + x for b, x in zip(bottom, v)]
    ax.set_xticks(list(xs), [r['split'].split(' ')[0] for r in rs])
    ax.set_xlim(-0.6, len(rs) - 0.4)
    ax.grid(axis='x', visible=False)
    nice_y(ax, 0, 100, zero=True, headroom=0.12, fmt='{:.0f}')
    panel_label(ax, probe)
save(fig, HERE / 'probe')
