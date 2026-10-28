"""The smallest miles weight update into the ported Flint server, to find why the server ranks' NCCL setup fails
(w541: "Cuda failure 'invalid argument'" at the receive). Prints this side's CUDA device (visible devices, PCI bus id,
UUID) and torch / NCCL versions, then sends ONE small tensor (the final norm, read from the checkpoint) in miles'
session frame through miles' NCCL group (protocols/broadcast.py connect), and checks that a 1000-token prefill is
bit-identical before and after. The group setup and the broadcast run in a thread, so a server that fails or never
joins ends this process (exit 2) instead of blocking it; a limit timer ends it after LIMIT_S (exit 3).
Prints NCCL_DIAG_OK or NCCL_DIAG_FAIL. Usage: nccl_diag.py seqs.json url"""
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
from safetensors import safe_open

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.broadcast import connect_rollout_engines_from_distributed
from miles.utils import async_utils

MODEL = "/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint"
NAME = "model.norm.weight"
GROUP = "miles-diag"
LIMIT_S = 75
ids = json.load(open(sys.argv[1]))[0]["ids"][:1000]
URL = sys.argv[2]
client = SGLangApiClient(server_url=URL)


def end(code, msg, exc=None):
    print("NCCL_DIAG_OK" if code == 0 else "NCCL_DIAG_FAIL", msg, flush=True)
    if exc is not None:
        print("".join(traceback.format_exception(exc)), flush=True)
    os._exit(code)


threading.Timer(LIMIT_S, end, args=(3, f"past {LIMIT_S} s")).start()


def rpc(name, **kw):
    try:
        r = async_utils.wait_futures([async_utils.submit(getattr(client, name)(**kw))])[0]
    except Exception as e:   # noqa: BLE001
        end(2, f"{name} failed", e)
    if isinstance(r, dict) and r.get("success") is False:
        end(2, f"{name} refused: {r}")
    return r


def prefill():
    body = {"input_ids": ids, "sampling_params": {"temperature": 1.0, "max_new_tokens": 1}, "return_logprob": True,
            "logprob_start_len": 0}
    m = requests.post(f"{URL}/generate", json=body, timeout=60).json()["meta_info"]
    return [x[0] for x in m["input_token_logprobs"][1:]]


def in_thread(fn, poll=None):
    """Run fn in a thread, so a blocked NCCL call cannot block this process: poll() may end the process meanwhile,
    and the limit timer ends it in any case. An exception in fn ends the process."""
    box = {}

    def run():
        try:
            box["r"] = fn()
        except Exception as e:   # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    while th.is_alive():
        if poll is not None:
            poll()
        time.sleep(0.02)
    if "e" in box:
        end(2, "a call in the thread failed", box["e"])
    return box.get("r")


props = torch.cuda.get_device_properties(0)
print("trainer CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), "device 0", props.name,
      "pci", f"{props.pci_bus_id:02x}" if hasattr(props, "pci_bus_id") else "?", "uuid", getattr(props, "uuid", "?"),
      "torch", torch.__version__, "nccl", ".".join(map(str, torch.cuda.nccl.version())), flush=True)
print("NCCL env", {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, flush=True)
torch.cuda.set_device(0)
index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{MODEL}/{index[NAME]}", "pt") as fh:
    tensor = fh.get_tensor(NAME).to("cuda:0").contiguous()
print("tensor", NAME, tuple(tensor.shape), tensor.dtype, flush=True)
before = prefill()
args = argparse.Namespace(rollout_num_gpus_per_engine=2, update_weight_transfer_mode="broadcast")
group = in_thread(lambda: connect_rollout_engines_from_distributed(args, GROUP, [client], engine_gpu_counts=[2]))
print("group connected", flush=True)
rpc("pause_generation", mode="retract")
rpc("flush_cache")
rpc("begin_weight_update", selector="all", sync_base=True)
fut = async_utils.submit(client.update_weights_from_distributed(
    names=[NAME], dtypes=[tensor.dtype], shapes=[tensor.shape], group_name=GROUP, selector="all"))


def poll():
    if fut.done() and (fut.exception() is not None or not fut.result().get("success")):
        end(2, "the server refused the update", fut.exception() or RuntimeError(str(fut.result())))


in_thread(lambda: dist.broadcast(tensor, 0, group=group), poll)
while not fut.done():
    time.sleep(0.02)
poll()
rpc("end_weight_update")
rpc("continue_generation")
rpc("flush_cache")
after = prefill()
end(0 if after == before else 1, f"one tensor sent; prefill bit-identical: {after == before}")
