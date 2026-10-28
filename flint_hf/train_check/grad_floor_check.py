"""How far apart two correct computations of the same whole-model gradient fall, per layer: the floor for
routing_check.py's training-vs-eager gradient gap. All runs use the routing an fp32 eager no-grad forward chose, on the
same real 8k sequence (the sum of the generated tokens' log-probs). Runs: A eager; B eager with the attention's query
chunks at 64 rows (same math, another fp32 summation order); C the training path (te + grouped experts); D the training
path again (determinism). Per layer: cosine and relative L2 of the gradients of the sinks, q_norm and the router, for
B vs A (the floor), C vs A and D vs C. Usage: grad_floor_check.py p0_seqs.json"""
import json
import sys

import torch

import flint_hf.modeling_flint as M
from flint_hf import load_flint
from routing_check import DEVICES, MODEL, record, routers, set_path


def grads(model, x, rows, path, fixed, params):
    set_path(model, path)
    for r in routers(model):
        r.select = (lambda t: lambda s, k: t)(fixed[r])
    for p in params.values():
        p.grad = None
    lp = model(x, head_rows=rows, head_ids=x[0, rows + 1, None]).logits[0, rows, 0]
    lp.sum().backward()
    return lp.detach().cpu(), {n: p.grad.float().cpu().flatten() for n, p in params.items()}


def table(tag, a, b, names):
    cos = lambda n: torch.nn.functional.cosine_similarity(a[1][n], b[1][n], dim=0).item()
    rel = lambda n: ((a[1][n] - b[1][n]).norm() / b[1][n].norm()).item()
    print(f"{tag}: log-probs mean |d| {(a[0] - b[0]).abs().mean():.4f}", flush=True)
    for kind in ("sinks", "q_norm", "router"):
        ks = [n for n in names if n.startswith(kind)]
        print(f"  {kind:6s} cosine " + " ".join(f"{cos(n):.3f}" for n in ks), flush=True)
        print(f"  {kind:6s} relL2  " + " ".join(f"{rel(n):.3f}" for n in ks), flush=True)


if __name__ == "__main__":
    seq = min(json.load(open(sys.argv[1])), key=lambda s: abs(len(s["ids"]) - 8192))
    ids = torch.tensor(seq["ids"][:8192])
    gen = torch.zeros(8192, dtype=torch.bool)
    for a, b in seq["spans"]:
        gen[a:min(b, 8192)] = True
    rows = torch.nonzero(gen[1:]).flatten()
    model = load_flint(MODEL, DEVICES)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = {}
    for i, layer in enumerate(model.model.layers):
        params[f"sinks.L{i}"] = layer.self_attn.softmax_offset
        params[f"q_norm.L{i}"] = layer.self_attn.q_norm.weight
        if isinstance(layer.mlp, M.FlintSparseMoeBlock):
            params[f"router.L{i}"] = layer.mlp.gate.weight
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
    x = ids[None].to(DEVICES[0])
    fixed = record(model, x, "eager")
    A = grads(model, x, rows, "eager", fixed, params)
    old, M.SCORE_ELEMS = M.SCORE_ELEMS, 1          # 64-query chunks in the eager attention
    B = grads(model, x, rows, "eager", fixed, params)
    M.SCORE_ELEMS = old
    C = grads(model, x, rows, "training", fixed, params)
    D = grads(model, x, rows, "training", fixed, params)
    names = list(params)
    table("B vs A (eager, another summation order: the floor)", B, A, names)
    table("C vs A (training vs eager)", C, A, names)
    table("D vs C (training path run twice)", D, C, names)
    print("GRAD_FLOOR_DONE", flush=True)
