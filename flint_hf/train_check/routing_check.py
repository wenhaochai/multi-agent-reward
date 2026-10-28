"""Why the whole-model gradients of kernel_check.py's two paths differ (relative L2 ~1 while the per-token log-probs
differ by 0.046 on average): expert-routing flips, or a wrong gradient in the training kernels? On the same real 8k
sequence, both paths run (a) with their own routing and (b) with the routing fixed to the one an fp32 eager no-grad
forward chose (every FlintTopKRouter.select replays it, in the forward and in the checkpoint recompute). With fixed
routing the paths differ only by their kernels' rounding. Also counts how many routing decisions the training path
flips against the eager one. Usage: routing_check.py p0_seqs.json"""
import json
import sys
import time

import torch

import flint_hf.modeling_flint as M
from flint_hf import load_flint

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def set_path(model, path):
    for m in model.modules():
        if isinstance(m, M.FlintAttention):
            m.attn_impl = "eager" if path == "eager" else "te"
        if isinstance(m, M.FlintSparseMoeBlock):
            m.experts_fn = M.experts_reference if path == "eager" else M.grouped_experts


def routers(model):
    return [m.mlp.gate for m in model.model.layers if isinstance(m.mlp, M.FlintSparseMoeBlock)]


def record(model, ids, path):
    """The experts each router selects in a no-grad forward of ``path``."""
    chosen = {}
    for r in routers(model):
        r.select = (lambda r: lambda s, k: chosen.setdefault(r, torch.topk(s, k, dim=-1, sorted=False)[1]))(r)
    set_path(model, path)
    with torch.no_grad():
        model(ids)
    for r in routers(model):
        del r.select
    return chosen


def run(model, ids, rows, path, fixed, probe):
    set_path(model, path)
    for r in routers(model):
        if fixed is not None:
            r.select = (lambda t: lambda s, k: t)(fixed[r])
        elif "select" in r.__dict__:
            del r.select
    for p in probe.values():
        p.grad = None
    t0 = time.time()
    lp = model(ids, head_rows=rows, head_ids=ids[0, rows + 1, None]).logits[0, rows, 0]
    lp.sum().backward()
    torch.cuda.synchronize()
    return lp.detach().cpu(), {n: p.grad.float().cpu() for n, p in probe.items()}, time.time() - t0


def compare(tag, a, b):
    d = (a[0] - b[0]).abs()
    rel = {n: ((a[1][n] - b[1][n]).norm() / b[1][n].norm()).item() for n in b[1]}
    cos = {n: torch.nn.functional.cosine_similarity(a[1][n].flatten(), b[1][n].flatten(), dim=0).item() for n in b[1]}
    print(f"{tag}: log-probs mean |d| {d.mean():.4f} max {d.max():.3f} | grad relative L2 "
          + ", ".join(f"{n} {v:.3e}" for n, v in rel.items()) + " | cosine " + ", ".join(f"{n} {v:.4f}" for n, v in cos.items()),
          flush=True)


if __name__ == "__main__":
    seq = min(json.load(open(sys.argv[1])), key=lambda s: abs(len(s["ids"]) - 8192))
    cut = 8192
    ids = torch.tensor(seq["ids"][:cut])
    gen = torch.zeros(cut, dtype=torch.bool)
    for a, b in seq["spans"]:
        gen[a:min(b, cut)] = True
    rows = torch.nonzero(gen[1:]).flatten()
    model = load_flint(MODEL, DEVICES)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    probe = {"sinks.L2": model.model.layers[2].self_attn.softmax_offset,
             "q_proj.L10": model.model.layers[10].self_attn.q_proj.weight,
             "router.L10": model.model.layers[10].mlp.gate.weight,
             "expert0.down.L10": model.model.layers[10].mlp.experts[0].down_proj.weight,
             "lm_head": model.lm_head.weight}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in probe.values():
        p.requires_grad_(True)
    x = ids[None].to(DEVICES[0])
    ref_routing = record(model, x, "eager")
    train_routing = record(model, x, "training")
    flips = [(ref_routing[r].sort(-1)[0] != train_routing[r].sort(-1)[0]).any(-1).float().mean().item() for r in routers(model)]
    print(f"tokens whose expert set differs between the paths, per MoE layer: mean {sum(flips) / len(flips):.4f}, "
          f"max {max(flips):.4f}, first layers {[round(f, 4) for f in flips[:5]]}", flush=True)
    eager_own = run(model, x, rows, "eager", None, probe)
    train_own = run(model, x, rows, "training", None, probe)
    eager_fix = run(model, x, rows, "eager", ref_routing, probe)
    train_fix = run(model, x, rows, "training", ref_routing, probe)
    print(f"times: eager {eager_own[2]:.1f} s, training {train_own[2]:.1f} s", flush=True)
    compare("own routing   (training vs eager)", train_own, eager_own)
    compare("fixed routing (training vs eager)", train_fix, eager_fix)
    compare("eager, fixed vs own routing      ", eager_fix, eager_own)
    print("ROUTING_CHECK_DONE", flush=True)
