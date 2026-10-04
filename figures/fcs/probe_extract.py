"""Outcome shares of the Qwen3.5-9B difficulty probes (miles-q38-build/runs/fcs_probe_qwen35_9b*/gen.jsonl and
scores_last.jsonl: every answer re-judged with the current judge and the LAST code block, tools/rescore_probe.py,
2026-10-04; the original scores.jsonl judged the longest block)
-> probe_outcomes.csv: per probe x split, the share of samples that were truncated at 32768 tokens, finished but
failed to compile (or had no code), compiled and scored 0, or scored > 0; mean score."""
import csv, json, sys
from pathlib import Path

R = Path('/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs')
rows = []
for probe, d in [('Base, thinking', 'fcs_probe_qwen35_9b'), ('Base, no thinking', 'fcs_probe_qwen35_9b_nothink'),
                 ('EasyPPO SFT', 'fcs_probe_qwen35_9b_sft')]:
    SC = R / d / 'scores_last.jsonl'
    if not SC.exists():
        continue
    g = {json.loads(l)['key']: json.loads(l) for l in open(R / d / 'gen.jsonl')}
    s = [json.loads(l) for l in open(SC)]
    for split, name in [('train', 'Train (200)'), ('val', 'Val (172)')]:
        rs = [r for r in s if r['split'] == split]
        n = len(rs)
        trunc = sum(g[r['key']]['finish'] == 'length' for r in rs)
        fin = [r for r in rs if g[r['key']]['finish'] != 'length']
        ce = sum(r['status'] in ('compile error', 'no code') for r in fin)
        zero = sum(r['status'] == 'done' and r['score'] == 0 for r in fin)
        pos = sum(r['score'] > 0 for r in fin)
        rows.append({'probe': probe, 'split': name, 'n': n, 'truncated': trunc / n, 'compile_error': ce / n,
                     'zero': zero / n, 'positive': pos / n, 'mean_score': sum(r['score'] for r in rs) / n})
with open(Path(__file__).parent / 'probe_outcomes.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for r in rows: print(r)
