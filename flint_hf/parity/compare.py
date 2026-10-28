"""Parity table from the saved results in OUTDIR. Per pair of scorers, over every predicted position: mean, p99 and
max of |delta logprob| of the actual next token; top-1 agreement; and the head KL, sum over the reference's top-32 ids
v of q_v (log q_v - log p_v) plus the tail term (1 - sum q) log((1 - sum q) / (1 - sum p)), q the reference and p the
other scorer; for an SGLang scorer only positions whose 32 reference ids it also returned count (n_kl). Positions are
split by the role of the predicted token: assistant text (what RL trains on) or user text (prompt and command output,
which the loss masks). Reference A = the kit's SGLang 0.5.16 at TP2, chunked prefill 4096 (the eval servers).
Usage: compare.py seqs.json OUTDIR"""
import json
import sys
from pathlib import Path

import torch

USER, ASSISTANT = 150003, 150004   # <|user|>, <|assistant|>
seqs = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
A = torch.load(out / "sglang_A.pt")
HF = torch.load(out / "hf.pt")
others = {n: torch.load(out / f"sglang_{n}.pt") for n in ("B", "B2", "C") if (out / f"sglang_{n}.pt").exists()}
KEYS = ("lp", "top_v", "top_i")


def pool(rs, keys=KEYS):
    return {k: torch.cat([r[k] for r in rs]) for k in keys}


def at_ids(other, ids):
    """Another SGLang scorer's logprobs at the reference ids, from its own top-32 (nan where absent)."""
    out = torch.full(ids.shape, float("nan"), dtype=torch.float64)
    match = other["top_i"][:, None, :] == ids[:, :, None]
    hit = match.any(-1)
    out[hit] = (match.double() * other["top_v"][:, None, :]).sum(-1)[hit]
    return out


def row(name, ref, oth, at, mask=None):
    keep = torch.ones(len(ref["lp"]), dtype=torch.bool) if mask is None else mask
    d = (oth["lp"] - ref["lp"]).abs()[keep]
    top1 = (oth["top_i"][:, 0] == ref["top_i"][:, 0])[keep].double()
    ok = keep & ~torch.isnan(at).any(1)
    q, p = ref["top_v"][ok].exp(), at[ok].exp()
    tq, tp = (1 - q.sum(1)).clamp(min=1e-12), (1 - p.sum(1)).clamp(min=1e-12)
    kl = (q * (ref["top_v"][ok] - at[ok])).sum(1) + tq * (tq.log() - tp.log())
    return (f"{name:<40} {d.numel():>7} {d.mean():.2e} {d.quantile(0.99):.2e} {d.max():.2e} {top1.mean():.4f} "
            f"{kl.mean():.2e} {int(ok.sum()):>7}")


def roles(ids):
    """For each predicted position p (token p + 1): True when the token is in an assistant message."""
    r, cur = [], False
    for t in ids[1:]:
        r.append(cur)
        cur = True if t == ASSISTANT else False if t == USER else cur
    return torch.tensor(r)


refA = pool(A["solo"])
hf = pool(HF["seqs"], KEYS + ("at_head",))
asst = torch.cat([roles(s["ids"]) for s in seqs])
pos = torch.cat([torch.arange(len(s["ids"]) - 1) for s in seqs])
print(f"{'pair (reference first)':<40} {'n':>7} {'mean|d|':>8} {'p99|d|':>8} {'max|d|':>8} {'top1':>6} {'headKL':>8} "
      f"{'n_kl':>7}")
for label, mask in (("all positions", None), ("assistant tokens", asst), ("user tokens", ~asst)):
    print(f"-- {label}")
    print(row("A alone vs A in a batch of 8", refA, pool(A["batch"]), at_ids(pool(A["batch"]), refA["top_i"]), mask))
    for n, o in others.items():
        s = pool(o["solo"])
        print(row(f"A vs {n}", refA, s, at_ids(s, refA["top_i"]), mask))
    if "B" in others and "B2" in others:
        b, b2 = pool(others["B"]["solo"]), pool(others["B2"]["solo"])
        print(row("B vs B2", b, b2, at_ids(b2, b["top_i"]), mask))
    print(row("A vs HF", refA, hf, hf["at_head"], mask))
print("-- HF vs A by position (assistant tokens)")
for lo, hi in ((0, 513), (513, 8192), (8192, 32768), (32768, 60000)):
    print(row(f"positions {lo}-{hi - 1}", refA, hf, hf["at_head"], asst & (pos >= lo) & (pos < hi)))
print("-- 1024 tokens sampled by A (temperature 1) after 4k, 16k and 48k tokens")
dec = pool(A["samples"])
pre = pool([{k: x["prefill"][k][-len(x["ids"]):] for k in KEYS} for x in A["samples"]])
hfs = pool(HF["samples"], KEYS + ("at_head",))
print(row("A decode vs A prefill", dec, pre, at_ids(pre, dec["top_i"])))
print(row("A decode vs HF", dec, hfs, hfs["at_head"]))
