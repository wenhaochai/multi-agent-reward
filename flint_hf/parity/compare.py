"""Parity table from the saved results. Per pair of scorers, over every predicted position: mean, p99 and max of
|delta logprob| of the actual next token; top-1 agreement; and the head KL, sum over the reference's top-32 ids v of
q_v (log q_v - log p_v) plus the tail term (1 - sum q) log((1 - sum q) / (1 - sum p)), q the reference (SGLang server A
alone) and p the other scorer; for an SGLang scorer only positions whose 32 ids it also returned count (n_kl).
Usage: compare.py seqs.json sglang_A.pt sglang_C.pt hf.pt"""
import json
import sys

import torch

seqs = json.load(open(sys.argv[1]))
A, C, HF = (torch.load(p) for p in sys.argv[2:5])


def head_kl(qv, qi, pv_at):
    """qv, qi: reference head [N, 32]; pv_at: the other scorer's logprobs at qi (nan where unknown)."""
    ok = ~torch.isnan(pv_at).any(1)
    q, p = qv[ok].exp(), pv_at[ok].exp()
    tq, tp = (1 - q.sum(1)).clamp(min=1e-12), (1 - p.sum(1)).clamp(min=1e-12)
    kl = (q * (qv[ok] - pv_at[ok])).sum(1) + tq * (tq.log() - tp.log())
    return kl, int(ok.sum())


def at_ids(other, ids):
    """The other SGLang scorer's logprobs at the reference ids, from its own top-32 (nan if absent)."""
    out = torch.full(ids.shape, float("nan"), dtype=torch.float64)
    match = other["top_i"][:, None, :] == ids[:, :, None]          # [N, 32 ref, 32 other]
    hit = match.any(-1)
    out[hit] = (match.double() * other["top_v"][:, None, :]).sum(-1)[hit]
    return out


def row(name, ref, oth, at, mask=None):
    d = (oth["lp"] - ref["lp"]).abs()
    top1 = (oth["top_i"][:, 0] == ref["top_i"][:, 0]).double()
    kl, n = head_kl(ref["top_v"], ref["top_i"], at)
    if mask is not None:
        d, top1 = d[mask], top1[mask]
    return (f"{name:<42} {d.numel():>8} {d.mean():.2e} {d.quantile(0.99):.2e} {d.max():.2e} {top1.mean():.4f} "
            f"{kl.mean():.2e} {n:>8}")


def cat(rs, key):
    return torch.cat([r[key] for r in rs])


print(f"{'pair (reference A = SGLang 0.5.16 TP2, alone)':<42} {'n':>8} {'mean|d|':>8} {'p99|d|':>8} {'max|d|':>8} "
      f"{'top1':>6} {'headKL':>8} {'n_kl':>8}")
pool = lambda rs: {k: cat(rs, k) for k in ("lp", "top_v", "top_i")}
refA, batchA, solC = pool(A["solo"]), pool(A["batch"]), pool(C["solo"])
hf = {k: cat(HF["seqs"], k) for k in ("lp", "top_v", "top_i", "at_head")}
print(row("floor: A in a batch of 8 vs A alone", refA, batchA, at_ids(batchA, refA["top_i"])))
print(row("SGLang 0.5.21 (miles.sif) vs A", refA, solC, at_ids(solC, refA["top_i"])))
print(row("HF eager fp32 vs A", refA, hf, hf["at_head"]))
pos = torch.cat([torch.arange(len(s["ids"]) - 1) for s in seqs])
print(row("  HF vs A, positions < 513 (window not full)", refA, hf, hf["at_head"], pos < 513))
print(row("  HF vs A, positions >= 513", refA, hf, hf["at_head"], pos >= 513))
print(row("  HF vs A, positions >= 32768", refA, hf, hf["at_head"], pos >= 32768))
print(row("  floor, positions >= 32768", refA, batchA, at_ids(batchA, refA["top_i"]), pos >= 32768))
for s, a, h in zip(seqs, A["solo"], HF["seqs"]):
    print(row(f"  HF vs A, {len(s['ids'])}-token sequence", a, h, h["at_head"]))
dec = pool(A["samples"])   # decode-time logprobs of the sampled tokens
pre = pool([{k: x["prefill"][k][-len(x["ids"]):] for k in ("lp", "top_v", "top_i")} for x in A["samples"]])
hfs = {k: cat(HF["samples"], k) for k in ("lp", "top_v", "top_i", "at_head")}
print(row("sampled tokens: A prefill vs A decode", dec, pre, at_ids(pre, dec["top_i"])))
print(row("sampled tokens: HF vs A decode", dec, hfs, hfs["at_head"]))
