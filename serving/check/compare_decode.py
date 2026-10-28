"""Decode vs prefill gaps (decode_check.py outputs) of several servers side by side: mean |decode - prefill| logprob
of the sampled tokens, overall, by the token's prefill probability and by prefix length.
Usage: compare_decode.py name=file.pt [...]"""
import sys

import torch

BINS = [(-1e9, -4.6), (-4.6, -2.3), (-2.3, -0.69), (-0.69, -0.105), (-0.105, 1.0)]   # p < .01, .01-.1, .1-.5, .5-.9, > .9
print(f"{'server':<8} {'n':>6} {'mean|d|':>8}  by prefill p: <.01 .01-.1 .1-.5 .5-.9 >.9 (n/mean)   by prefix: 4k 16k")
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    res = torch.load(path)
    dec, pre = torch.cat([r["decode"] for r in res]), torch.cat([r["prefill"] for r in res])
    d = (dec - pre).abs()
    cells = []
    for lo, hi in BINS:
        m = (pre >= lo) & (pre < hi)
        cells.append(f"{int(m.sum())}/{d[m].mean():.3f}" if m.any() else "0/-")
    by_prefix = []
    for n in (4000, 16000):
        dd = [(r["decode"] - r["prefill"]).abs() for r in res if r["prefix"] == n]
        by_prefix.append(f"{torch.cat(dd).mean():.4f}" if dd else "-")
    print(f"{name:<8} {d.numel():>6} {d.mean():.4f}  {' '.join(cells)}   {' '.join(by_prefix)}")
