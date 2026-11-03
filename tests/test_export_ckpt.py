"""rl/export_ckpt.py on a small synthetic checkpoint saved through torch.distributed.checkpoint: every expert, its row split, bf16, the files beside the weights; names or shapes that
differ from the base checkpoint fail. Run in miles-v0.1.3.sif: python3 tests/test_export_ckpt.py"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file

spec = importlib.util.spec_from_file_location("export_ckpt", os.path.join(os.path.dirname(__file__), "..", "rl", "export_ckpt.py"))
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)
E, INTER, H = 3, 4, 8   # experts, inter size, hidden size
torch.manual_seed(0)


def ckpt(d, extra=None):
    """The fp32 master as flint_hf stacks it, in miles' checkpoint form."""
    state = {"layers.0.gate_up": torch.randn(E, 2 * INTER, H), "layers.0.down": torch.randn(E, H, INTER), "layers.1.q": torch.randn(H, H),
             "head": torch.randn(5, H), "embed": torch.randn(5, H)}
    named = {"model.layers.0.mlp.experts.gate_up_proj": state["layers.0.gate_up"], "model.layers.0.mlp.experts.down_proj": state["layers.0.down"],
             "model.layers.1.self_attn.q_proj.weight": state["layers.1.q"], "lm_head.weight": state["head"], "model.embed_tokens.weight": state["embed"]}
    named |= extra or {}
    dcp.save({X.PREFIX + k: v for k, v in named.items()}, checkpoint_id=d, no_dist=True)
    return named


def base(d, named):
    """The base checkpoint: the same tensors in its form (bf16, one per expert)."""
    os.makedirs(d)
    tensors = {}
    for k, t in named.items():
        tensors |= X.unstack(k, t)
    save_file({k: torch.zeros_like(v) for k, v in tensors.items()}, os.path.join(d, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    for f in ("config.json", "tokenizer.json"):
        open(os.path.join(d, f), "w").write("{}")
    os.makedirs(os.path.join(d, "sglang"))
    open(os.path.join(d, "README.md"), "w").write("not linked")
    return tensors


with tempfile.TemporaryDirectory() as td:
    named = ckpt(f"{td}/model")
    want = base(f"{td}/base", named)
    r = X.export(f"{td}/model", f"{td}/base", f"{td}/out")
    assert r["tensors"] == len(want) == 2 * E + E + 3, r
    idx = json.load(open(f"{td}/out/model.safetensors.index.json"))["weight_map"]
    assert set(idx) == set(want)
    got = {}
    for f in set(idx.values()):
        got |= load_file(f"{td}/out/{f}")
    assert set(got) == set(want)
    for k, v in got.items():
        assert v.dtype == torch.bfloat16 and torch.equal(v, want[k]), k
    g, u, d = named["model.layers.0.mlp.experts.gate_up_proj"], named["model.layers.0.mlp.experts.down_proj"], None
    for e in range(E):
        assert torch.equal(got[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"], g[e, :INTER].to(torch.bfloat16)), "gate is the first half of the rows"
        assert torch.equal(got[f"model.layers.0.mlp.experts.{e}.up_proj.weight"], g[e, INTER:].to(torch.bfloat16)), "up is the second half"
        assert torch.equal(got[f"model.layers.0.mlp.experts.{e}.down_proj.weight"], u[e].to(torch.bfloat16))
    assert torch.equal(got["lm_head.weight"], named["lm_head.weight"].to(torch.bfloat16))
    assert os.path.islink(f"{td}/out/config.json") and os.path.islink(f"{td}/out/sglang") and not os.path.lexists(f"{td}/out/README.md")
    assert len(set(idx.values())) == 3, "one file per layer and one for the rest"
    # a tensor the base lacks; a base tensor the checkpoint lacks; another shape
    extra = {"model.layers.1.self_attn.k_proj.weight": torch.randn(H, H)}
    ckpt(f"{td}/model2", extra)
    try:
        X.export(f"{td}/model2", f"{td}/base", f"{td}/out2")
        sys.exit("no error for a tensor the base lacks")
    except ValueError as e:
        assert "not in the base" in str(e), e
    os.remove(f"{td}/base/model-00001-of-00001.safetensors")
    more = dict(want) | {"model.norm.weight": torch.zeros(H)}
    save_file({k: torch.zeros_like(v) for k, v in more.items()}, f"{td}/base/model-00001-of-00001.safetensors", metadata={"format": "pt"})
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in more}}, open(f"{td}/base/model.safetensors.index.json", "w"))
    try:
        X.export(f"{td}/model", f"{td}/base", f"{td}/out3")
        sys.exit("no error for a base tensor the checkpoint lacks")
    except ValueError as e:
        assert "not in the checkpoint" in str(e), e
print("ALL OK")
