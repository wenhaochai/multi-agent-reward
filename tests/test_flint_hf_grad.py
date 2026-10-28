"""CPU gradient checks of flint_hf in float64: torch.autograd.gradcheck over every parameter of a tiny random Flint
(a dense layer with a sliding window and RoPE, then an MoE layer with full attention, no RoPE, the sink, the shared
expert and sigmoid top-k routing), scored through the chunked head (head_rows / head_ids, the trainer's path) on two
packed documents; head_log_probs against an explicit log_softmax (values, gradients, entropy, empty slots, repeated
ids, temperature); and grouped_experts against experts_reference. In miles-v0.1.3.sif: PYTHONPATH=. python3 tests/test_flint_hf_grad.py"""
import torch
from torch.func import functional_call

import flint_hf.modeling_flint as M
from flint_hf import FlintConfig, FlintForCausalLM


def tiny():
    cfg = FlintConfig(vocab_size=23, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=4, moe_intermediate_size=8, shared_expert_intermediate_size=12,
                      num_experts=4, num_experts_per_tok=2, sliding_window=3,
                      layer_types=["sliding_attention", "full_attention"], no_rope_layer_indices=[1],
                      mlp_only_layers=[0])
    torch.manual_seed(0)
    m = FlintForCausalLM(cfg).double()
    for mod in m.modules():
        if isinstance(mod, M.FlintAttention):
            mod.softmax_offset.data.normal_(0, 1)
        if isinstance(mod, M.FlintRMSNorm):
            mod.weight.data.uniform_(0.5, 1.5)
        if isinstance(mod, M.FlintTopKRouter):
            mod.weight.data.normal_(0, 1)   # well-separated scores: no top-k tie within the finite difference
    return m


def test_gradcheck_whole_model():
    m = tiny()
    ids = torch.tensor([3, 7, 1, 9, 2, 5, 11, 4, 8, 6, 2])           # two documents: 6 + 5 tokens
    pos = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4])
    rows = torch.tensor([1, 2, 3, 4, 7, 8, 9])
    head = torch.tensor([[ids[r + 1], 0, 5] for r in rows.tolist()])  # the next token and two other ids
    head[2, 2] = -1
    names = [n for n, _ in m.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for _, p in m.named_parameters())

    def f(*ps):
        out = functional_call(m, dict(zip(names, ps)), (ids[None], pos[None]),
                              {"head_rows": rows, "head_ids": head, "head_temperature": 0.8})
        return out.logits[0, rows]

    n = sum(p.numel() for p in params)
    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-7, rtol=1e-5, fast_mode=False)
    print(f"gradcheck: {len(params)} parameters, {n} elements, float64")


def test_head_log_probs_matches_log_softmax():
    torch.manual_seed(1)
    R, H, V, t = 37, 8, 29, 0.7
    h = torch.randn(R, H, dtype=torch.float64, requires_grad=True)
    w = torch.randn(V, H, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(0, V, (R, 5))
    ids[:, 1] = ids[:, 0]          # a repeated id
    ids[3, 2:] = -1                # empty slots
    g = torch.randn(R, 5, dtype=torch.float64)
    for chunk in (4, 64):
        lp, ent = M.head_log_probs(h, w, ids, t, chunk=chunk, with_entropy=True)
        ref_all = torch.log_softmax(h @ w.t() / t, dim=-1)
        ref = torch.where(ids >= 0, ref_all.gather(1, ids.clamp(min=0)), 0.0)
        assert torch.allclose(lp, ref, atol=1e-12)
        assert torch.allclose(ent, -(ref_all.exp() * ref_all).sum(-1), atol=1e-12)
        gh, gw = torch.autograd.grad((lp * g).sum(), (h, w))
        rh, rw = torch.autograd.grad((ref * g).sum(), (h, w))
        assert torch.allclose(gh, rh, atol=1e-12) and torch.allclose(gw, rw, atol=1e-12), chunk
    assert torch.autograd.gradcheck(lambda a, b: M.head_log_probs(a, b, ids, t, chunk=8)[0], (h, w))
    print("head_log_probs: values, entropy and gradients equal log_softmax's at chunks 4 and 64")


def test_head_rows_equal_full_logits():
    m = tiny()
    ids = torch.tensor([[3, 7, 1, 9, 2, 5, 11, 4]])
    rows = torch.tensor([0, 2, 5, 6])
    head = torch.tensor([[1, 2], [9, -1], [11, 0], [4, 4]])
    with torch.no_grad():
        full = torch.log_softmax(m(ids).logits[0], dim=-1)
        out = m(ids, head_rows=rows, head_ids=head, head_entropy=True).logits[0]
    ref = torch.where(head >= 0, full[rows].gather(1, head.clamp(min=0)), 0.0)
    assert torch.allclose(out[rows, :2], ref, atol=1e-12)
    assert torch.allclose(out[rows, 2], -(full[rows].exp() * full[rows]).sum(-1), atol=1e-12)
    others = torch.ones(ids.shape[1], dtype=torch.bool)
    others[rows] = False
    assert out[others].abs().max() == 0
    print("head_rows: the scored rows equal the full logits' log_softmax, the other rows are 0")


def test_grouped_experts_match_reference():
    cfg = FlintConfig(hidden_size=64, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2,
                      shared_expert_intermediate_size=16, num_hidden_layers=1, layer_types=["full_attention"])
    torch.manual_seed(2)
    block = M.FlintSparseMoeBlock(cfg)   # float32: torch._grouped_mm takes no float64
    for p in block.parameters():
        p.data.normal_(0, 0.1)
    x = torch.randn(50, 64, requires_grad=True)
    res = []
    for fn in (M.experts_reference, M.grouped_experts):
        block.experts_fn = fn
        block.zero_grad(set_to_none=True)
        x.grad = None
        out = block(x)
        out.pow(2).sum().backward()
        res.append([out.detach(), x.grad.clone()] + [p.grad.clone() for p in block.parameters()])
    for a, b in zip(*res):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    print("grouped_experts: output and every gradient equal experts_reference's (float32, 1e-6)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
