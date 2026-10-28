"""transformers' from_pretrained path (miles' FSDP backend loads on CPU this way) for flint_hf: load info, sampled
tensors equal to the checkpoint's, the expert_bias buffers zero. CPU, ~85 GB RAM: PYTHONPATH=. python3 <this>"""
import json
import time

import torch
from safetensors import safe_open

from flint_hf import FlintForCausalLM

M = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
t0 = time.time()
model, info = FlintForCausalLM.from_pretrained(M, dtype=torch.bfloat16, output_loading_info=True)
print(f"from_pretrained {time.time() - t0:.0f}s; missing {len(info['missing_keys'])}; "
      f"unexpected {len(info['unexpected_keys'])}; mismatched {len(info.get('mismatched_keys', []))}")
idx = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
params = dict(model.named_parameters())
names = ["model.embed_tokens.weight", "model.layers.0.self_attn.softmax_offset", "model.layers.7.mlp.gate.weight",
         "model.layers.7.mlp.experts.200.down_proj.weight", "model.layers.48.mlp.shared_expert.up_proj.weight",
         "model.layers.30.self_attn.q_norm.weight", "model.norm.weight", "lm_head.weight"]
for n in names:
    with safe_open(f"{M}/{idx[n]}", "pt") as f:
        print(n, torch.equal(f.get_tensor(n), params[n].data))
print("expert_bias buffers all zero:",
      all(float(b.abs().sum()) == 0 for n, b in model.named_buffers() if n.endswith("expert_bias")))
print("dtype", {p.dtype for p in model.parameters()})
