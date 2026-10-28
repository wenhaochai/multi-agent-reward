"""Weight updates into the ported Flint server through miles' own non-colocated transfer path, in miles' session frame
(training_utils/weight_update/session.py: pause, flush, begin_weight_update, buckets, end_weight_update, resume), with
miles' SGLangApiClient, its NCCL group (protocols/broadcast.py connect/disconnect) and its bucketing
(hf_weight_iterator.bucketing, 512 MiB). Each bucket is sent as broadcast.update_weights_from_distributed sends it (the
HTTP request, then the NCCL broadcast from rank 0), but the group setup and the broadcast run in a thread, so a
request the server refuses, or an NCCL setup that never completes, ends this process instead of blocking rank 0. HF-named tensors come
from flint_hf (S1, c0b0cef) loaded on this process's GPUs, as the FSDP trainer would send them.
On the first three S1 sequences (2k, 4k, 8k tokens):
(a) every parameter sent back unchanged: the served logprobs stay bit-identical;
(b) three changes through the three load paths, a plain tensor (layer 10 o_proj x1.25), a shard of the fused qkv
    (layer 3 v_proj x1.25) and every expert's gate_proj of layer 20 in the fused experts (x1.1): the served logprobs
    move as flint_hf's move under the same change; sending the saved originals then restores them bit for bit;
(c) the wall time of a full update (all 39.8B parameters).
Every server call that fails prints the server's reply and ends the process; so does a run past LIMIT_S.
Usage: update_check.py seqs.json url out.json"""
import argparse
import json
import os
import sys
import threading
import time
import traceback

import requests
import torch
import torch.distributed as dist

from flint_hf import load_flint
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import pack_units_by_size
from miles.backends.training_utils.weight_update.protocols.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
)
from miles.utils import async_utils

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
USER, ASSISTANT = 150003, 150004
BUCKET = 512 * 1024**2   # miles' --update-weight-buffer-size default
GROUP = "miles-s3"
BUCKET_LIMIT_S = 300     # one bucket: request, broadcast and reply
LIMIT_S = 40 * 60        # the whole check
seqs = json.load(open(sys.argv[1]))[:3]
URL, OUT = sys.argv[2], sys.argv[3]
client = SGLangApiClient(server_url=URL)


def die(msg, exc=None):
    print("UPDATE_CHECK_ABORT", msg, flush=True)
    if exc is not None:
        print("".join(traceback.format_exception(exc)), flush=True)   # includes the server's reply (an exception note)
    os._exit(1)


threading.Timer(LIMIT_S, die, args=(f"past {LIMIT_S} s",)).start()


def rpc(name, **kw):
    """One SGLangApiClient call; a failed call or a reply with success false ends the process."""
    try:
        r = async_utils.wait_futures([async_utils.submit(getattr(client, name)(**kw))])[0]
    except Exception as e:   # noqa: BLE001
        die(f"{name} failed", e)
    if isinstance(r, dict) and r.get("success") is False:
        die(f"{name} refused: {r}")
    return r


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


def in_thread(fn, poll=None):
    """Run fn in a thread, so a blocked NCCL call (the communicator is built inside the first collective) cannot block
    this process: poll() may end the process meanwhile, and the limit timer ends it in any case."""
    box = {}

    def run():
        try:
            box["r"] = fn()
        except Exception as e:   # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    deadline = time.time() + BUCKET_LIMIT_S
    while th.is_alive():
        if poll is not None:
            poll()
        if time.time() > deadline:
            die(f"an NCCL step took over {BUCKET_LIMIT_S} s")
        time.sleep(0.02)
    if "e" in box:
        die("an NCCL step failed", box["e"])
    return box.get("r")


def send_bucket(bucket):
    tensors = [t for _, t in bucket]
    fut = async_utils.submit(client.update_weights_from_distributed(
        names=[n for n, _ in bucket], dtypes=[t.dtype for t in tensors], shapes=[t.shape for t in tensors],
        group_name=GROUP, selector="all"))

    def refused():
        if fut.done() and (fut.exception() is not None or not fut.result().get("success")):
            die("the server refused a bucket", fut.exception() or RuntimeError(str(fut.result())))

    def broadcast():
        for h in [dist.broadcast(t, 0, group=group, async_op=True) for t in tensors]:
            h.wait()

    in_thread(broadcast, refused)
    in_thread(lambda: fut.result(), refused)
    refused()


def send(named):
    """One weight update in miles' session frame; the radix cache is flushed after it."""
    rpc("pause_generation", mode="retract")
    rpc("flush_cache")
    rpc("begin_weight_update", selector="all", sync_base=True)
    units = ([(n, t.detach().to("cuda:0").contiguous())] for n, t in named)
    for bucket in pack_units_by_size(units, BUCKET):
        send_bucket(bucket)
    rpc("end_weight_update")
    rpc("continue_generation")
    rpc("flush_cache")


torch.cuda.set_device(0)
base = served()
t0 = time.time()
model = load_flint(MODEL, ["cuda:0", "cuda:1"]).eval()
print(f"flint_hf loaded in {time.time() - t0:.0f}s", flush=True)
hf_base = hf_scores(model)
asst = assistant_mask()
params = dict(model.named_parameters())
args = argparse.Namespace(rollout_num_gpus_per_engine=2, update_weight_transfer_mode="broadcast")
group = in_thread(lambda: connect_rollout_engines_from_distributed(args, GROUP, [client], engine_gpu_counts=[2]))
res = {"n_params": sum(p.numel() for p in params.values()), "n_tensors": len(params)}

# (a) + (c): every parameter, unchanged
t0 = time.time()
send(params.items())
res["full_update_s"] = round(time.time() - t0, 1)
ident = served()
res["identity"] = {k: float((ident[k] != base[k]).double().sum()) for k in ("lp", "top_v", "top_i")}
print("identity", res["identity"], "full update", res["full_update_s"], "s", flush=True)

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
os._exit(0)   # the limit timer is a non-daemon thread
