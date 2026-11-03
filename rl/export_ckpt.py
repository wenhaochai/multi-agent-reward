#!/usr/bin/env python3
"""A miles FSDP checkpoint's weights as a checkpoint an engine can load: the warm-up engine of an async run's resume (rl/elastic_engine.sh EL_WARM_CKPT) serves from it while
the actor waits for its first data. The checkpoint (torch.distributed.checkpoint, written by miles/backends/fsdp_utils/checkpoint.py) holds the fp32 master as flint_hf
stacks it; the engine reads what the weight push sends and the base checkpoint holds: bf16, one tensor per expert. So each layer is read alone, cast to bf16, unstacked as
flint_hf.modeling_flint.load_flint stacks it, and written to a file of its own (at most a few GiB in memory at a time). Non-weight files are linked from the base. The
names and shapes must equal the base checkpoint's, or the export fails. Whether the weights are right is the run's own check: the producer compares every tensor with the
run's own engine (/weights_checker) before it routes to this engine. Usage: export_ckpt.py MODEL_DIR BASE_CHECKPOINT OUT_DIR. Runs in miles-v0.1.3.sif, no GPU."""
import json
import os
import re
import struct
import sys
import time

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

PREFIX = "model_state.model."
LAYER = re.compile(r"^model\.layers\.(\d+)\.")
SKIP = re.compile(r"^(model-.*\.safetensors|model\.safetensors\.index\.json|\.cache|\.gitattributes|assets|LICENSE|README\.md)$")
_DTYPE = {"BF16": torch.bfloat16, "F32": torch.float32}


def base_header(base: str) -> dict[str, tuple[str, list[int]]]:
    """Tensor name -> (dtype, shape) of the base checkpoint, from its safetensors headers."""
    index = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(index.values())):
        with open(os.path.join(base, f), "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        out |= {k: (v["dtype"], v["shape"]) for k, v in header.items() if k != "__metadata__"}
    return out


def unstack(name: str, t: torch.Tensor) -> dict[str, torch.Tensor]:
    """What one checkpoint tensor (fp32) is on the engine's side (bf16): the stacked experts of a layer one per expert, anything else as it is."""
    t = t.to(torch.bfloat16)
    if name.endswith(".experts.gate_up_proj"):
        stem, inter = name[: -len(".gate_up_proj")], t.shape[1] // 2
        out = {}
        for e in range(t.shape[0]):
            out[f"{stem}.{e}.gate_proj.weight"] = t[e, :inter].clone()
            out[f"{stem}.{e}.up_proj.weight"] = t[e, inter:].clone()
        return out
    if name.endswith(".experts.down_proj"):
        stem = name[: -len(".down_proj")]
        return {f"{stem}.{e}.down_proj.weight": t[e].clone() for e in range(t.shape[0])}
    return {name: t}


def export(model_dir: str, base: str, out_dir: str) -> dict:
    t0 = time.time()
    meta = FileSystemReader(model_dir).read_metadata().state_dict_metadata
    bad = [k for k in meta if not k.startswith(PREFIX)]
    if bad:
        raise ValueError(f"checkpoint keys outside {PREFIX}: {bad[:3]}")
    groups: dict[str, list[str]] = {}   # one file per layer, one for the rest
    for k in meta:
        m = LAYER.match(k[len(PREFIX):])
        groups.setdefault(m.group(1) if m else "rest", []).append(k)
    os.makedirs(out_dir, exist_ok=True)
    want = base_header(base)
    weight_map, total = {}, 0
    names = sorted(groups, key=lambda g: (g == "rest", int(g) if g != "rest" else 0))
    for i, g in enumerate(names, 1):
        state = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) for k in groups[g]}
        dcp.load(state, checkpoint_id=model_dir, no_dist=True)
        tensors: dict[str, torch.Tensor] = {}
        for k, t in state.items():
            tensors |= unstack(k[len(PREFIX):], t)
        del state
        for name, t in tensors.items():
            if name not in want:
                raise ValueError(f"{name}: not in the base checkpoint")
            if list(t.shape) != want[name][1] or t.dtype != _DTYPE[want[name][0]]:
                raise ValueError(f"{name}: {t.dtype} {list(t.shape)}, base {want[name]}")
        fname = f"model-{i:05d}-of-{len(names):05d}.safetensors"
        save_file(tensors, os.path.join(out_dir, fname), metadata={"format": "pt"})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        weight_map |= {name: fname for name in tensors}
    missing = sorted(set(want) - set(weight_map))
    if missing:
        raise ValueError(f"{len(missing)} base tensors not in the checkpoint, e.g. {missing[:3]}")
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    for f in os.listdir(base):
        if not SKIP.match(f) and not os.path.lexists(os.path.join(out_dir, f)):
            os.symlink(os.path.join(base, f), os.path.join(out_dir, f))
    return {"tensors": len(weight_map), "files": len(names), "gib": round(total / 2**30, 1), "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    print("[export_ckpt]", json.dumps(export(*sys.argv[1:4])), flush=True)
