#!/usr/bin/env python3
"""The order an async run (rl/rl_job.sh async=1 place=2numa) hands its 8 GPUs out in: miles' actor takes the first 4 (miles MILES_GPU_ORDER), so 2 GPUs of each NUMA
node come first and the run's own engine gets the rest. A rank holds 157 GiB on the host in training (weights, Adam moments, gradients): 4 ranks on one NUMA node
need 628 GiB against the node's 515 GiB, and fall into 4 KB page faults and reclaim on every micro-batch (tmp/elastic_rl/async.md); 2 per node fit, as the 8-rank
sync actor does. Prints the GPU ids, comma-separated; exit 1 with the reason on stderr when the GPUs are not 8, 4 on each of two NUMA nodes.
Usage: gpu_order.py [CUDA_VISIBLE_DEVICES]"""
import subprocess
import sys


def order(numa_of: dict[str, int]) -> list[str]:
    """numa_of: GPU id -> NUMA node, in the job's order."""
    nodes: dict[int, list[str]] = {}
    for gpu, node in numa_of.items():
        nodes.setdefault(node, []).append(gpu)
    if len(numa_of) != 8 or len(nodes) != 2 or any(len(v) != 4 for v in nodes.values()):
        raise ValueError(f"needs 8 GPUs, 4 on each of 2 NUMA nodes: {numa_of}")
    a, b = (nodes[k] for k in sorted(nodes))
    return a[:2] + b[:2] + a[2:] + b[2:]


def numa_of_gpus(ids: list[str]) -> dict[str, int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True, timeout=60, check=True).stdout
    bus = {i.strip(): b.strip() for i, b in (x.split(",") for x in out.strip().splitlines())}
    res = {}
    for g in ids:
        pci = bus[g].lower()[-12:]   # 00000000:19:00.0 -> 0000:19:00.0, as sysfs names it
        res[g] = int(open(f"/sys/bus/pci/devices/{pci}/numa_node").read())
    return res


if __name__ == "__main__":
    try:
        ids = [x.strip() for x in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if x.strip()]
        print(",".join(order(numa_of_gpus(ids))))
    except Exception as e:
        print(f"gpu_order: {e}", file=sys.stderr)
        sys.exit(1)
