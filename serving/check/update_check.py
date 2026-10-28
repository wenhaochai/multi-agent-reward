"""Weight updates into the ported Flint server through miles' own non-colocated transfer path: NCCL broadcast from
one trainer rank (miles/backends/training_utils/weight_update/protocols/broadcast.py) in buckets packed by
hf_weight_iterator.bucketing, HF-named tensors from flint_hf (S1, c0b0cef) loaded on this process's GPUs, as the FSDP
trainer would send them. On the first three S1 sequences (2k, 4k, 8k tokens):
(a) every parameter sent back unchanged: the served logprobs stay bit-identical;
(b) three updates through the three load paths, a plain tensor (layer 10 o_proj x1.25), a shard of the fused qkv
    (layer 3 v_proj x1.25) and every expert's gate_proj of layer 20 in the fused experts (x1.1): the served logprobs
    move as flint_hf's move under the same change; sending the saved originals then restores them bit for bit;
(c) the wall time of a full update (all 39.8B parameters).
Usage: update_check.py seqs.json url out.json"""
import argparse
import json
import sys
import time

import requests
import torch

from flint_hf import load_flint
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import pack_units_by_size
from miles.backends.training_utils.weight_update.protocols.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)
from miles.utils import async_utils

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
USER, ASSISTANT = 150003, 150004
BUCKET = 512 * 1024**2   # miles' --update-weight-buffer-size default
GROUP = "miles-s3"
seqs = json.load(open(sys.argv[1]))[:3]
URL, OUT = sys.argv[2], sys.argv[3]
client = SGLangApiClient(server_url=URL)


def served():
    """Logprob of each next token and the top-32 head, from a prefill of every sequence."""
    res = []
    for s in seqs:
        body = {"input_ids": s["ids"], "sampling_params": {"temperature": 1.0, "max_new_tokens": 1},
                "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 32}
        m = requests.post(f"{URL}/generate", json=body, timeout=3600).json()["meta_info"]
        res.append({"lp": torch.tensor([x[0] for x in m["input_token_logprobs"][1:]], dtype=torch.float64),
                    "top_v": torch.tensor([[x[0] for x in r] for r in m["input_top_logprobs"][1:]], dtype=torch.float64),
                    "top_i": torch.tensor([[x[1] for x in r] for r in m["input_top_logprobs"][1:]])})
    return {k: torch.cat([r[k] for r in res]) for k in res[0]}


@torch.no_grad()
def hf_scores(model):
    lps = []
    for s in seqs:
        x = torch.tensor(s["ids"], device="cuda:0")
        lp, _, _ = model.token_logprobs(model.model(x[:-1]), x[1:], topk=32)
        lps.append(lp.double().cpu())
    return torch.cat(lps)


def assistant_mask():
    out = []
    for s in seqs:
        cur = False
        for t in s["ids"][1:]:
            out.append(cur)
            cur = True if t == ASSISTANT else False if t == USER else cur
    return torch.tensor(out)


def send(named):
    """One update: buckets of HF-named tensors broadcast from this rank, then the server's radix cache flushed."""
    units = ([(n, t.detach().to("cuda:0").contiguous())] for n, t in named)
    for bucket in pack_units_by_size(units, BUCKET):
        for r in async_utils.wait_futures(update_weights_from_distributed(GROUP, group, [client], bucket)):
            assert r["success"], r
    async_utils.wait_futures([async_utils.submit(client.flush_cache())])


torch.cuda.set_device(0)
base = served()
t0 = time.time()
model = load_flint(MODEL, ["cuda:0", "cuda:1"]).eval()
print(f"flint_hf loaded in {time.time() - t0:.0f}s", flush=True)
hf_base = hf_scores(model)
asst = assistant_mask()
params = dict(model.named_parameters())
args = argparse.Namespace(rollout_num_gpus_per_engine=2, update_weight_transfer_mode="broadcast")
group = connect_rollout_engines_from_distributed(args, GROUP, [client], engine_gpu_counts=[2])
res = {"n_params": sum(p.numel() for p in params.values()), "n_tensors": len(params)}

# (a) + (c): every parameter, unchanged
t0 = time.time()
send(params.items())
res["full_update_s"] = round(time.time() - t0, 1)
ident = served()
res["identity"] = {k: float((ident[k] != base[k]).double().sum()) for k in ("lp", "top_v", "top_i")}

# (b): three load paths, then the originals back
changes = {"model.layers.10.self_attn.o_proj.weight": 1.25, "model.layers.3.self_attn.v_proj.weight": 1.25,
           **{f"model.layers.20.mlp.experts.{e}.gate_proj.weight": 1.1 for e in range(256)}}
orig = {n: params[n].detach().clone() for n in changes}
with torch.no_grad():
    for n, f in changes.items():
        params[n].mul_(f)
send((n, params[n]) for n in changes)
pert, hf_pert = served(), hf_scores(model)
with torch.no_grad():
    for n in changes:
        params[n].copy_(orig[n])
send(orig.items())
restored = served()
res["restore"] = {k: float((restored[k] != base[k]).double().sum()) for k in ("lp", "top_v", "top_i")}
disconnect_rollout_engines_from_distributed(args, GROUP, group, [client])


def mean_abs(x, mask):
    return float(x.abs()[mask].mean())


d_srv, d_hf = pert["lp"] - base["lp"], hf_pert - hf_base
res["perturb_assistant"] = {
    "served_change": mean_abs(d_srv, asst), "hf_change": mean_abs(d_hf, asst),
    "change_gap": mean_abs(d_srv - d_hf, asst), "corr": float(torch.corrcoef(torch.stack([d_srv[asst], d_hf[asst]]))[0, 1]),
    "served_vs_hf_base": mean_abs(base["lp"] - hf_base, asst), "served_vs_hf_perturbed": mean_abs(pert["lp"] - hf_pert, asst),
    "n": int(asst.sum())}
json.dump(res, open(OUT, "w"), indent=1)
print("UPDATE_CHECK", json.dumps(res), flush=True)
