"""miles debug rollout data for the S2 training step: one GRPO group of 8 from build_p0.py's sequences (one prompt,
Pack the Polyominoes), each a Sample with the loss on Flint's generated tokens, reward = the episode's S, and the
sampler's next-token log-probs and top-32 heads from rescore.py. Written as rollout_<id>.pt for every id given, so
miles' --load-debug-rollout-data trains on the same batch at each step. Usage:
make_rollout.py p0_seqs.json rescore_dir out_dir id [id ...]"""
import json
import sys

import numpy as np
import torch

from miles.utils.types import Sample

seqs = json.load(open(sys.argv[1]))
RESCORE, OUT, IDS = sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]]
samples = []
for i, s in enumerate(seqs):
    ids, p = s["ids"], s["prompt_len"]
    r = torch.load(f"{RESCORE}/seq{i}.pt")
    assert len(r["lp"]) == len(ids) - 1
    mask = np.zeros(len(ids), dtype=np.int64)
    for a, b in s["spans"]:
        mask[a:b] = 1
    sl = slice(p - 1, len(ids) - 1)       # the rows that predict the response tokens p .. end
    samples.append(Sample(
        group_index=0, index=i, prompt=f"Pack the Polyominoes ({s['run']})", tokens=list(ids),
        response_length=len(ids) - p, reward=float(s["S"]), loss_mask=mask[p:].tolist(),
        rollout_log_probs=r["lp"][sl].tolist(), rollout_topk_token_ids=r["top_i"][sl].numpy().astype(np.int32),
        rollout_topk_log_probs=r["top_v"][sl].numpy().astype(np.float32), status=Sample.Status.COMPLETED,
        metadata={"run": s["run"]}))
    print(s["run"], "tokens", len(ids), "response", len(ids) - p, "loss tokens", int(mask[p:].sum()), "reward",
          round(s["S"], 3), flush=True)
for rid in IDS:
    torch.save({"rollout_id": rid, "metadata": {}, "samples": [x.to_dict() for x in samples]}, f"{OUT}/rollout_{rid}.pt")
print("wrote", [f"{OUT}/rollout_{rid}.pt" for rid in IDS])
