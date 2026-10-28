"""HF side of the parity check, on every GPU in CUDA_VISIBLE_DEVICES (load_flint's manual pipeline). Loads the model,
waits for SGLang's results, then for every sequence (and each sampled continuation) the log-probability of each next
token, the HF top-32 head, and the HF logprobs of SGLang's head ids. Then one forward+backward on 8k tokens with
gradient checkpointing: finite gradients, peak memory. Then the transformers loading path that miles' FSDP backend
uses (from_pretrained on CPU) against load_flint. Usage: hf_score.py seqs.json sglang.pt out.pt"""
import json
import os
import sys
import time

import torch

from flint_hf import FlintForCausalLM, load_flint

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
SEQS, SG, OUT = sys.argv[1:4]
seqs = json.load(open(SEQS))
devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
t0 = time.time()
model = load_flint(MODEL, devs).eval()
print(f"loaded on {devs} in {time.time() - t0:.0f}s", flush=True)
while not os.path.exists(SG):
    time.sleep(20)
time.sleep(5)
sg = torch.load(SG)


@torch.no_grad()
def score(ids, head_ids):
    x = torch.tensor(ids, device=devs[0])
    h = model.model(x[:-1])
    lp, tv, ti = model.token_logprobs(h, x[1:], topk=32)
    at = []
    for s in range(0, h.shape[0], 2048):
        lps = torch.log_softmax(model.logits(h[s:s + 2048]), -1)
        at.append(lps.gather(1, head_ids[s:s + 2048].to(lps.device)))
    return {"lp": lp.double().cpu(), "top_v": tv.double().cpu(), "top_i": ti.cpu(), "at_head": torch.cat(at).double().cpu()}


res = {"seqs": [], "samples": []}
for s, a in zip(seqs, sg["solo"]):
    t0 = time.time()
    res["seqs"].append(score(s["ids"], a["top_i"]))
    print(f"scored {len(s['ids'])} tokens in {time.time() - t0:.0f}s", flush=True)
for smp in sg["samples"]:
    n = len(smp["ids"])
    head = torch.cat([torch.zeros(len(smp["prefix"]) - 1, 32, dtype=torch.long), smp["top_i"]])
    r = score(smp["prefix"] + smp["ids"], head)
    res["samples"].append({k: v[-n:] for k, v in r.items()})
torch.save(res, OUT)
print("HF_SCORED", OUT, flush=True)

model.train()
model.gradient_checkpointing_enable()
for d in devs:
    torch.cuda.reset_peak_memory_stats(d)
x = torch.tensor(seqs[2]["ids"][:8192], device=devs[0])
t0 = time.time()
lp = model.token_logprobs(model.model(x[:-1]), x[1:])
(-lp.mean()).backward()
for d in devs:
    torch.cuda.synchronize(d)
counts = {"finite": 0, "nonfinite": 0, "none_nonexpert": 0, "none_expert": 0}
for name, p in model.named_parameters():
    if p.grad is None:
        counts["none_expert" if ".experts." in name else "none_nonexpert"] += 1
    else:
        counts["finite" if torch.isfinite(p.grad).all() else "nonfinite"] += 1
peak = [round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in devs]
print(f"BACKWARD {x.numel() - 1} tokens in {time.time() - t0:.0f}s: {counts}; loss {(-lp.mean()).item():.4f}; "
      f"peak GiB per GPU {peak}", flush=True)
ref = {n: p.detach() for n, p in model.named_parameters()}
del model
torch.cuda.empty_cache()

t0 = time.time()
hf = FlintForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
names = ["model.embed_tokens.weight", "model.layers.0.self_attn.softmax_offset", "model.layers.7.mlp.gate.weight",
         "model.layers.7.mlp.experts.200.down_proj.weight", "model.layers.48.mlp.shared_expert.up_proj.weight",
         "lm_head.weight"]
same = all(torch.equal(dict(hf.named_parameters())[n], ref[n].cpu()) for n in names)
print(f"FROM_PRETRAINED in {time.time() - t0:.0f}s: {sum(1 for _ in hf.parameters())} parameters, sampled tensors "
      f"equal load_flint's: {same}", flush=True)
