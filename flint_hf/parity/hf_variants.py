"""Is the HF-vs-SGLang gap on user tokens a bug or Flint's sensitivity there? Re-scores sequences with numerical
variants of flint_hf that are all correct implementations (each mimics one rounding step of the SGLang kernels) and
compares every variant with the base HF and with SGLang A, split by the role of the predicted token. Also reports the
norm of the final hidden state by role. Usage: hf_variants.py seqs.json OUTDIR [i,j,...] (sequence indices, default
1,3,5: 4k, 16k and 32k tokens; reads OUTDIR/sglang_A.pt)"""
import json
import sys
from pathlib import Path

import torch

import flint_hf.modeling_flint as M
from flint_hf import load_flint

USER, ASSISTANT = 150003, 150004
MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
seqs = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
A = torch.load(out / "sglang_A.pt")
devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
model = load_flint(MODEL, devs).eval()
base_attn, base_norm, base_experts = M.eager_sink_attention, M.FlintRMSNorm.forward, M.FlintSparseMoeBlock.experts_fn


def attn_p_bf16(q, k, v, sinks, scale, window, cu_seqlens):
    """FA3 multiplies bf16 probabilities with bf16 values: round P before P @ V."""
    T, H, D = q.shape
    rep = H // k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().repeat_interleave(rep, dim=1).transpose(0, 1)
    vb = v.repeat_interleave(rep, dim=1).transpose(0, 1)
    sink = sinks.float().view(H, 1, 1)
    outs = []
    for a, b in zip(cu_seqlens.tolist()[:-1], cu_seqlens.tolist()[1:]):
        span = (b - a) if window is None else min(b - a, window + 2048)
        step = max(64, min(2048, M.SCORE_ELEMS // (H * span)))
        for s in range(a, b, step):
            e = min(b, s + step)
            lo = a if window is None else max(a, s - window + 1)
            sc = torch.matmul(qf[:, s:e], kf[:, lo:e].transpose(1, 2)) * scale
            i = torch.arange(s, e, device=q.device)[:, None]
            j = torch.arange(lo, e, device=q.device)[None, :]
            ok = (j <= i) if window is None else (j <= i) & (i - j < window)
            sc = torch.cat([sc.masked_fill(~ok, float("-inf")), sink.expand(H, e - s, 1)], dim=-1)
            p = torch.softmax(sc, dim=-1)[..., :-1]
            outs.append(torch.matmul(p.to(v.dtype), vb[:, lo:e]))
    return torch.cat(outs, dim=1).transpose(0, 1).to(q.dtype)


def norm_rounded_sum(self, x, residual=None):
    """Normalize the bf16-rounded sum instead of the fp32 sum."""
    if residual is None:
        return self._norm(x.float(), x.dtype)
    s = (x.float() + residual.float()).to(x.dtype)
    return self._norm(s.float(), x.dtype), s


def experts_round_each(experts, x, weights, ids):
    """Triton's fused MoE: each expert's weighted output rounded to bf16, then summed."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    flat_ids, flat_w = ids.flatten(), weights.flatten()
    order = torch.argsort(flat_ids, stable=True)
    tok = order // ids.shape[1]
    start = 0
    for e, n in enumerate(torch.bincount(flat_ids, minlength=len(experts)).tolist()):
        if n:
            sel, t = order[start:start + n], tok[start:start + n]
            y = (experts[e](x[t]).float() * flat_w[sel, None]).to(x.dtype)
            out.index_add_(0, t, y.float())
            start += n
    return out.to(x.dtype)


VARIANTS = {"base": {}, "P_bf16": {"attn": attn_p_bf16}, "norm_rounded_sum": {"norm": norm_rounded_sum},
            "experts_round_each": {"experts": experts_round_each},
            "all_three": {"attn": attn_p_bf16, "norm": norm_rounded_sum, "experts": experts_round_each}}


@torch.no_grad()
def score(ids):
    x = torch.tensor(ids, device=devs[0])
    h = model.model(x[:-1])
    lp, _, ti = model.token_logprobs(h, x[1:], topk=1)
    return lp.double().cpu(), ti[:, 0].cpu(), h.float().norm(dim=-1).cpu()


def roles(ids):
    r, cur = [], False
    for t in ids[1:]:
        r.append(cur)
        cur = True if t == ASSISTANT else False if t == USER else cur
    return torch.tensor(r)


pick = [int(i) for i in (sys.argv[3] if len(sys.argv) > 3 else "1,3,5").split(",")]
res = {}
for name, v in VARIANTS.items():
    M.ATTENTION_FUNCTIONS["eager"] = v.get("attn", base_attn)
    M.FlintRMSNorm.forward = v.get("norm", base_norm)
    M.FlintSparseMoeBlock.experts_fn = staticmethod(v.get("experts", base_experts))
    res[name] = [score(seqs[j]["ids"]) for j in pick]
    print("scored", name, flush=True)
asst = torch.cat([roles(seqs[j]["ids"]) for j in pick])
pos = torch.cat([torch.arange(len(seqs[j]["ids"]) - 1) for j in pick])
a_lp = torch.cat([A["solo"][j]["lp"] for j in pick])
a_top = torch.cat([A["solo"][j]["top_i"][:, 0] for j in pick])
base_lp = torch.cat([r[0] for r in res["base"]])
base_top = torch.cat([r[1] for r in res["base"]])
print(f"{'variant':<20} {'role':<10} {'vs base mean|d|':>16} {'top1':>6} {'vs A mean|d|':>13} {'top1':>6}")
for name in VARIANTS:
    lp = torch.cat([r[0] for r in res[name]])
    top = torch.cat([r[1] for r in res[name]])
    for role, m in (("assistant", asst), ("user", ~asst), ("asst >=32k", asst & (pos >= 32768))):
        if not m.any():
            continue
        d_base, d_a = (lp - base_lp).abs()[m].mean(), (lp - a_lp).abs()[m].mean()
        t_base, t_a = (top == base_top)[m].double().mean(), (top == a_top)[m].double().mean()
        print(f"{name:<20} {role:<10} {d_base:16.3e} {t_base:6.4f} {d_a:13.3e} {t_a:6.4f}")
hn = torch.cat([r[2] for r in res["base"]])
print(f"final hidden norm (after the last norm): assistant median {hn[asst].median():.1f}, "
      f"user median {hn[~asst].median():.1f}")
