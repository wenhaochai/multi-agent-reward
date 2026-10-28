"""A tiny random Flint checkpoint with Flint's real vocabulary, tokenizer and every structural feature (a dense and two
MoE layers, sliding and full attention, no-RoPE layers, sinks, a shared expert), for an integration run of miles' FSDP
path on 4 GPUs before the 40B step. Usage: make_tiny.py out_dir"""
import shutil
import sys

import torch

from flint_hf import FlintConfig, FlintForCausalLM

SRC = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
out = sys.argv[1]
cfg = FlintConfig.from_pretrained(SRC)
cfg.update(dict(hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                head_dim=64, moe_intermediate_size=128, shared_expert_intermediate_size=128, num_experts=8,
                num_experts_per_tok=2, layer_types=["full_attention", "sliding_attention", "full_attention"],
                no_rope_layer_indices=[0, 2], mlp_only_layers=[0]))
torch.manual_seed(0)
model = FlintForCausalLM(cfg).to(torch.bfloat16)
model.save_pretrained(out)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(f"{SRC}/{f}", f"{out}/{f}")
print("tiny Flint", sum(p.numel() for p in model.parameters()) / 1e6, "M params ->", out)
