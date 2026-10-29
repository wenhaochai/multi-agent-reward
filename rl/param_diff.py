"""Every trained tensor of a saved actor (miles' FSDP dcp checkpoint, flint_hf names) against the original Flint
checkpoint (per-expert names; flint_hf stacks each layer's experts as gate_up_proj [E, 2I, H] and down_proj [E, H, I]).
Per tensor: max|d|, ||d||/||w||, max|d|/(max|w|+eps), max|d| in units of the lr, the fraction of elements changed,
dtype. Writes <out>/param_diff.json, prints the ranking. Usage: param_diff.py CKPT_MODEL_DIR HF_DIR OUT LR (on CPU)."""
import json
import os
import sys

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

ckpt, hf, out, lr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
reader = dcp.FileSystemReader(ckpt)
meta = reader.read_metadata().state_dict_metadata
index = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
handles = {}


def hf_tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(hf, f), "pt")
    return handles[f].get_tensor(name).float()


def original(name):
    if name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
        base, kind = name.rsplit(".", 1)
        e = 0
        rows = []
        while f"{base}.{e}.down_proj.weight" in index:
            if kind == "gate_up_proj":
                rows.append(torch.cat([hf_tensor(f"{base}.{e}.gate_proj.weight"), hf_tensor(f"{base}.{e}.up_proj.weight")]))
            else:
                rows.append(hf_tensor(f"{base}.{e}.down_proj.weight"))
            e += 1
        return torch.stack(rows)
    return hf_tensor(name)


rows, missing = [], []
for key, m in meta.items():
    if not hasattr(m, "size"):
        continue
    name = key.removeprefix("model.")   # ModelState saves {"model": state_dict}
    t = torch.empty(m.size, dtype=m.properties.dtype)
    dcp.load({key: t}, storage_reader=reader, no_dist=True)
    try:
        w = original(name)
    except KeyError:
        missing.append(name)
        continue
    t = t.float()
    assert t.shape == w.shape, (name, t.shape, w.shape)
    d = t - w
    eps = 1e-12
    rows.append({"name": name, "dtype": str(m.properties.dtype), "numel": d.numel(),
                 "max_abs_d": d.abs().max().item(), "max_abs_w": w.abs().max().item(),
                 "rel_norm": (d.norm() / (w.norm() + eps)).item(),
                 "rel_max": (d.abs().max() / (w.abs().max() + eps)).item(),
                 "max_d_in_lr": d.abs().max().item() / lr, "frac_changed": (d != 0).float().mean().item()})
hf_names = set(index)
json.dump({"rows": rows, "missing_in_hf": missing}, open(os.path.join(out, "param_diff.json"), "w"), indent=1)
print(len(rows), "tensors compared;", len(missing), "not in the HF checkpoint:", missing[:10])
for k in ("rel_norm", "max_d_in_lr"):
    print(f"--- top 25 by {k}")
    for r in sorted(rows, key=lambda r: -r[k])[:25]:
        print(f"{r['name']:70} {r['dtype']:14} max|d| {r['max_abs_d']:.3e} max|w| {r['max_abs_w']:.3e} "
              f"rel_norm {r['rel_norm']:.3e} max_d/lr {r['max_d_in_lr']:.1f} changed {r['frac_changed']:.3f}")
by = {}
for r in rows:
    kind = r["name"].split(".")[-2] + "." + r["name"].split(".")[-1]
    by.setdefault(kind, []).append(r)
print("--- per kind: median rel_norm, max max_d/lr, median frac_changed")
for kind, rs in sorted(by.items()):
    rs2 = sorted(x["rel_norm"] for x in rs)
    print(f"{kind:40} n={len(rs):3} rel_norm_med {rs2[len(rs2)//2]:.2e} max_d/lr {max(x['max_d_in_lr'] for x in rs):.1f} "
          f"changed_med {sorted(x['frac_changed'] for x in rs)[len(rs)//2]:.3f}")
