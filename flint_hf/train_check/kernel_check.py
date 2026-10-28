"""GPU checks of Flint's training kernels against the exact eager reference (S2 gate 1b/1c). On 4 GPUs:
1. attention: Transformer Engine's sink attention (``te``) vs ``eager`` on random packed documents (32/4 heads of 128,
   window 513 and full, Flint's real sinks), forward and gradients of q, k, v and the sinks; then one 65,536-token
   document through ``te`` forward + backward (time, memory);
2. experts: ``grouped_experts`` (torch._grouped_mm) vs the per-expert reference on a real layer's weights and
   router, forward and gradients of the input, the router and two experts;
3. head: ``head_log_probs`` on 65,536 rows at Flint's vocabulary (time, memory) and against log_softmax on 2,048;
4. model: the whole Flint on a real 8k-token sequence (build_p0.py's output), eager path vs training path (te +
   grouped experts), per-token log-probs of the generated tokens and gradients of a few parameters.
Prints one line per comparison: max and mean absolute difference, and the reference's scale.
Usage: kernel_check.py p0_seqs.json"""
import json
import sys
import time

import torch

import flint_hf.modeling_flint as M
from flint_hf import FlintConfig, load_flint

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
torch.manual_seed(0)


def diff(name, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    print(f"{name:42s} max {d.max().item():.3e} mean {d.mean().item():.3e} | ref max {b.abs().max().item():.3e} "
          f"mean {b.abs().mean().item():.3e}", flush=True)


def peak(dev):
    return f"{torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB"


def real_tensor(name):
    from safetensors import safe_open
    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{MODEL}/{index[name]}", "pt") as f:
        return f.get_tensor(name)


_FLEX = {}


def flex_sink_attention(q, k, v, sinks, scale, window, cu_seqlens):
    """Candidate fallback: PyTorch flex attention on packed documents (block mask: same document, causal, window),
    the sink applied through the log-sum-exp: o * sigmoid(lse - sink), exact for a value-less sink logit."""
    from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

    if "fn" not in _FLEX:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    T, H, D = q.shape
    cu = cu_seqlens.to(q.device)
    doc = torch.repeat_interleave(torch.arange(len(cu) - 1, device=q.device), cu[1:] - cu[:-1])

    def mask(b, h, qi, ki):
        m = (doc[qi] == doc[ki]) & (ki <= qi)
        return m & (qi - ki < window) if window else m

    bm = create_block_mask(mask, None, None, T, T, device=q.device)
    out, aux = _FLEX["fn"](q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=bm,
                           scale=scale, enable_gqa=True, return_aux=AuxRequest(lse=True))
    w = torch.sigmoid(aux.lse.float() - sinks.float()[None, :, None])           # [1, H, T]
    return (out.float() * w[..., None])[0].transpose(0, 1).to(q.dtype)


M.ATTENTION_FUNCTIONS["flex"] = flex_sink_attention
WORKING = []   # attention candidates that ran forward and backward on the random inputs


def check_attention(dev):
    H, KV, D = 32, 4, 128
    sinks = real_tensor("model.layers.2.self_attn.softmax_offset").float().to(dev)
    cu = torch.tensor([0, 1500, 4096], device=dev)
    for window in (513, None):
        q = (torch.randn(4096, H, D, device=dev) * 1.5).bfloat16().requires_grad_()
        k = (torch.randn(4096, KV, D, device=dev) * 1.5).bfloat16().requires_grad_()
        v = torch.randn(4096, KV, D, device=dev).bfloat16().requires_grad_()
        s = sinks.clone().requires_grad_()
        do = torch.randn(4096, H, D, device=dev).bfloat16()
        outs, grads = {}, {}
        for impl in ("eager", "te", "flex"):
            for t in (q, k, v, s):
                t.grad = None
            try:
                o = M.ATTENTION_FUNCTIONS[impl](q, k, v, s, D ** -0.5, window, cu)
                o.backward(do)
            except Exception as e:   # report and go on: the other candidate may still work
                print(f"attention {impl} (window {window}) FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            outs[impl], grads[impl] = o.detach(), [t.grad.clone() for t in (q, k, v, s)]
            if impl != "eager" and window is None and impl in outs:
                WORKING.append(impl)
        tag = f"window {window}"
        for impl in ("te", "flex"):
            if impl in outs:
                diff(f"attention {impl} out ({tag})", outs[impl], outs["eager"])
                for n, gt, ge in zip(("dq", "dk", "dv", "dsinks"), grads[impl], grads["eager"]):
                    diff(f"attention {impl} {n} ({tag})", gt, ge)
    # one long document through te, forward + backward
    T = 65536
    torch.cuda.reset_peak_memory_stats(dev)
    q = torch.randn(T, H, D, device=dev).bfloat16().requires_grad_()
    k = torch.randn(T, KV, D, device=dev).bfloat16().requires_grad_()
    v = torch.randn(T, KV, D, device=dev).bfloat16().requires_grad_()
    s = sinks.clone().requires_grad_()
    for impl in ("te", "flex"):
        for window in (513, None):
            for t in (q, k, v, s):
                t.grad = None
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            try:
                M.ATTENTION_FUNCTIONS[impl](q, k, v, s, D ** -0.5, window, torch.tensor([0, T], device=dev)).sum().backward()
            except Exception as e:
                print(f"attention {impl} {T} tokens window {window} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            torch.cuda.synchronize(dev)
            print(f"attention {impl} {T} tokens window {window}: fwd+bwd {time.time() - t0:.2f} s, finite "
                  f"{all(bool(torch.isfinite(t.grad).all()) for t in (q, k, v, s))}, peak {peak(dev)}", flush=True)


def check_experts(dev):
    cfg = FlintConfig.from_pretrained(MODEL)
    block = M.FlintSparseMoeBlock(cfg)
    layer = 10
    state = {n: real_tensor(f"model.layers.{layer}.mlp.{n}") for n, _ in block.named_parameters()}
    block.load_state_dict(state, strict=False)
    block = block.to(dev, torch.bfloat16)
    block.gate.expert_bias = torch.zeros(cfg.num_experts, device=dev)
    x = (torch.randn(8192, cfg.hidden_size, device=dev) * 0.5).bfloat16()
    gout = torch.randn(8192, cfg.hidden_size, device=dev).bfloat16()
    res = {}
    for name, fn in (("reference", M.experts_reference), ("grouped", M.grouped_experts)):
        block.experts_fn = fn
        block.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_()
        out = block(xi)
        out.backward(gout)
        e0 = int(block.gate(x)[2][0, 0])
        res[name] = (out.detach(), xi.grad, block.gate.weight.grad.clone(),
                     block.experts[e0].down_proj.weight.grad.clone(), block.experts[e0].gate_proj.weight.grad.clone())
    for n, a, b in zip(("out", "dx", "drouter", "dexpert.down", "dexpert.gate"), res["grouped"], res["reference"]):
        diff(f"experts {n} (layer {layer}, 8192 tokens)", a, b)


def check_head(dev):
    V, Hd = 150144, 2048
    w = (real_tensor("lm_head.weight")).to(dev).requires_grad_()
    h = (torch.randn(2048, Hd, device=dev) * 2).bfloat16().requires_grad_()
    ids = torch.randint(0, V, (2048, 33), device=dev)
    lp, _ = M.head_log_probs(h, w, ids, 1.0)
    ref = torch.log_softmax(h.float() @ w.float().t(), dim=-1).gather(1, ids)
    diff("head log-probs vs log_softmax (2048 rows)", lp, ref)
    g = torch.randn_like(lp)
    gh, gw = torch.autograd.grad((lp * g).sum(), (h, w))
    rh, rw = torch.autograd.grad((ref * g).sum(), (h, w))
    diff("head dhidden", gh, rh)
    diff("head dweight", gw, rw)
    T = 65536
    torch.cuda.reset_peak_memory_stats(dev)
    h = torch.randn(T, Hd, device=dev).bfloat16().requires_grad_()
    ids = torch.randint(0, V, (T, 33), device=dev)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    lp, _ = M.head_log_probs(h, w, ids, 1.0)
    lp.sum().backward()
    torch.cuda.synchronize(dev)
    print(f"head {T} rows x 33 ids: fwd+bwd {time.time() - t0:.2f} s, peak {peak(dev)} "
          f"(full fp32 logits alone would be {T * V * 4 / 2**30:.1f} GiB)", flush=True)


def check_model(seqs_path, devices):
    seq = min(json.load(open(seqs_path)), key=lambda s: abs(len(s["ids"]) - 8192))
    cut = 8192
    ids = torch.tensor(seq["ids"][:cut])
    gen = torch.zeros(cut, dtype=torch.bool)
    for a, b in seq["spans"]:
        gen[a:min(b, cut)] = True
    rows = torch.nonzero(gen[1:]).flatten()   # row r predicts token r + 1
    model = load_flint(MODEL, devices)
    model.gradient_checkpointing_enable()   # the eager reference keeps fp32 attention probabilities otherwise
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
    res = {}
    fast = WORKING[0] if WORKING else "eager"
    print(f"model training path uses attention {fast!r}", flush=True)
    for path in ("eager", "training"):
        for m in model.modules():
            if isinstance(m, M.FlintAttention):
                m.attn_impl = "eager" if path == "eager" else fast
            if isinstance(m, M.FlintSparseMoeBlock):
                m.experts_fn = M.experts_reference if path == "eager" else M.grouped_experts
        for p in probe.values():
            p.grad = None
        for d in devices:
            torch.cuda.reset_peak_memory_stats(d)
        t0 = time.time()
        out = model(ids[None].to(devices[0]), head_rows=rows, head_ids=ids[rows + 1, None]).logits[0, rows, 0]
        out.sum().backward()
        torch.cuda.synchronize()
        print(f"model {path}: {len(rows)} generated tokens of {cut}, fwd+bwd {time.time() - t0:.1f} s, peak "
              + ", ".join(peak(d) for d in devices), flush=True)
        res[path] = (out.detach().cpu(), {n: p.grad.float().cpu() for n, p in probe.items()})
    diff("model log-probs of generated tokens", res["training"][0], res["eager"][0])
    for n in probe:
        diff(f"model grad {n}", res["training"][1][n], res["eager"][1][n])
    rel = {n: ((res["training"][1][n] - res["eager"][1][n]).norm() / res["eager"][1][n].norm()).item() for n in probe}
    print("model grad relative L2 difference: " + ", ".join(f"{n} {v:.3e}" for n, v in rel.items()), flush=True)


if __name__ == "__main__":
    print("torch", torch.__version__, "GPUs", torch.cuda.device_count(), flush=True)
    check_attention("cuda:0")
    check_experts("cuda:1")
    check_head("cuda:2")
    torch.cuda.empty_cache()
    check_model(sys.argv[1], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    print("KERNEL_CHECK_DONE", flush=True)
