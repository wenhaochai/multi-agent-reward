"""Gates for the stacked experts (coordinator 2026-10-28), on CPU.
sync: FlintForCausalLM.from_pretrained (the path miles' FSDP actor takes) and, for every parameter, what weight sync
  sends: revert_weight_conversion of a stacked tensor (miles' unfuse_batched_experts), else the tensor itself. Every
  produced name must be a checkpoint tensor and equal it bit for bit, and every checkpoint tensor must be produced.
logprobs: generated-token log-probs of a 2,048-token prefix (the prompt is 1,106) of a real packing-problem sequence from load_flint
  (eager attention, reference experts, bf16, CPU); run once with this code and once with the per-expert code
  (PYTHONPATH of an older flint_hf), then `compare` checks they are bit-identical.
Usage: stack_check.py sync | logprobs OUT.pt | compare A.pt B.pt"""
import json
import sys

import torch

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
SEQS = "/scratch/gpfs/GROUP/USER/tmp/flint_s2/p0_seqs.json"

if sys.argv[1] == "sync":
    from safetensors import safe_open
    from transformers.core_model_loading import revert_weight_conversion

    from flint_hf import FlintForCausalLM
    model = FlintForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    handles = {f: safe_open(f"{MODEL}/{f}", "pt") for f in set(index.values())}
    seen, bad = set(), []
    stacked = 0
    for name, t in model.state_dict().items():
        if t.dim() == 3 and name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            outs, stacked = revert_weight_conversion(model, {name: t}), stacked + 1
        else:
            outs = {name: t}
        for n, v in outs.items():
            ref = handles[index[n]].get_tensor(n) if n in index else None
            if ref is None or ref.dtype != v.dtype or not torch.equal(ref, v.contiguous()):
                bad.append(n)
            seen.add(n)
    missing = set(index) - seen
    print(f"sync: {len(model.state_dict())} model tensors ({stacked} stacked) -> {len(seen)} synced names; "
          f"checkpoint {len(index)}; not bit-identical {len(bad)} {bad[:3]}; never sent {len(missing)} {sorted(missing)[:3]}")
    print("SYNC_OK" if not bad and not missing else "SYNC_FAIL")
elif sys.argv[1] == "logprobs":
    from flint_hf import load_flint
    seq = json.load(open(SEQS))[0]
    ids = torch.tensor(seq["ids"][:2048])
    gen = torch.zeros(2048, dtype=torch.bool)
    for a, b in seq["spans"]:
        gen[a:min(b, 2048)] = True
    rows = torch.nonzero(gen[1:]).flatten()
    model = load_flint(MODEL, ["cpu"])
    with torch.no_grad():
        lp = model(ids[None], head_rows=rows, head_ids=ids[rows + 1, None]).logits[0, rows, 0]
    torch.save({"rows": rows, "lp": lp}, sys.argv[2])
    print("logprobs", len(rows), "tokens, mean", lp.mean().item())
else:
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    same = torch.equal(a["rows"], b["rows"]) and torch.equal(a["lp"], b["lp"])
    d = (a["lp"] - b["lp"]).abs()
    print(f"logprobs: {len(a['lp'])} generated tokens, bit-identical {same}, max |d| {d.max().item():.3e}")
