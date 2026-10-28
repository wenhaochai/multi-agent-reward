"""Parity of two Flint SGLang servers on the S1 sequences (query_sglang.py outputs). Per predicted position: |delta
logprob| of the actual next token (mean, p99, max), top-1 agreement, and the top-32 head KL (the reference's head;
positions where the other server's 32 ids miss one of the reference's are left out of the KL: n_kl), as in S1's
flint_hf/parity/compare.py. Rows by the role of the predicted token and, for assistant tokens, by position.
Usage: compare_servers.py seqs.json ref.pt other.pt [ref_old.pt]
(ref = the kit, SGLang 0.5.16; other = the port on 0.5.23 with --floor results; ref_old = S1's run of the kit)"""
import json
import sys

import torch

USER, ASSISTANT = 150003, 150004   # <|user|>, <|assistant|>
KEYS = ("lp", "top_v", "top_i")
seqs = json.load(open(sys.argv[1]))
ref, oth = torch.load(sys.argv[2]), torch.load(sys.argv[3])
old = torch.load(sys.argv[4]) if len(sys.argv) > 4 else None


def pool(rs):
    return {k: torch.cat([r[k] for r in rs]) for k in KEYS}


def at_ids(other, ids):
    out = torch.full(ids.shape, float("nan"), dtype=torch.float64)
    match = other["top_i"][:, None, :] == ids[:, :, None]
    hit = match.any(-1)
    out[hit] = (match.double() * other["top_v"][:, None, :]).sum(-1)[hit]
    return out


def row(name, r, o, mask=None):
    keep = torch.ones(len(r["lp"]), dtype=torch.bool) if mask is None else mask
    d = (o["lp"] - r["lp"]).abs()[keep]
    top1 = (o["top_i"][:, 0] == r["top_i"][:, 0])[keep].double()
    at = at_ids(o, r["top_i"])
    ok = keep & ~torch.isnan(at).any(1)
    q, p = r["top_v"][ok].exp(), at[ok].exp()
    tq, tp = (1 - q.sum(1)).clamp(min=1e-12), (1 - p.sum(1)).clamp(min=1e-12)
    kl = (q * (r["top_v"][ok] - at[ok])).sum(1) + tq * (tq.log() - tp.log())
    return (f"{name:<38} {d.numel():>7} {d.mean():.2e} {d.quantile(0.99):.2e} {d.max():.2e} {top1.mean():.4f} "
            f"{kl.mean():.2e} {int(ok.sum()):>7}")


def roles(ids):
    r, cur = [], False
    for t in ids[1:]:
        r.append(cur)
        cur = True if t == ASSISTANT else False if t == USER else cur
    return torch.tensor(r)


R, O = pool(ref["solo"]), pool(oth["solo"])
asst = torch.cat([roles(s["ids"]) for s in seqs])
pos = torch.cat([torch.arange(len(s["ids"]) - 1) for s in seqs])
print(f"{'pair (reference first)':<38} {'n':>7} {'mean|d|':>8} {'p99|d|':>8} {'max|d|':>8} {'top1':>6} {'headKL':>8} "
      f"{'n_kl':>7}")
for label, mask in (("all positions", None), ("assistant tokens", asst), ("user tokens", ~asst)):
    print(f"-- {label}")
    print(row("kit 0.5.16 vs port 0.5.23", R, O, mask))
    if "batch" in oth:
        print(row("port alone vs port in a batch of 8", O, pool(oth["batch"]), mask))
    if old is not None:
        print(row("kit now vs kit in S1", pool(old["solo"]), R, mask))
print("-- kit vs port by position (assistant tokens)")
for lo, hi in ((0, 513), (513, 8192), (8192, 32768), (32768, 60000)):
    print(row(f"positions {lo}-{hi - 1}", R, O, asst & (pos >= lo) & (pos < hi)))
if "samples" in oth:
    print("-- 1024 tokens sampled by the port (temperature 1) after 4k, 16k and 48k tokens")
    dec = pool(oth["samples"])
    pre = pool([{k: x["prefill"][k][-len(x["ids"]):] for k in KEYS} for x in oth["samples"]])
    print(row("port decode vs port prefill", dec, pre))
