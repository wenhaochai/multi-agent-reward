"""rl/gpu_order.py: 2 GPUs of each NUMA node first (the actor's), the others after (the run's own engine). Run: python3 tests/test_gpu_order.py"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("gpu_order", os.path.join(os.path.dirname(__file__), "..", "rl", "gpu_order.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
assert G.order({str(i): i // 4 for i in range(8)}) == list("01452367"), G.order({str(i): i // 4 for i in range(8)})
assert G.order({str(i): 1 - i // 4 for i in range(8)}) == list("45016723"), "the lower NUMA node first, each node's GPUs in the job's order"
assert G.order({str(i): i % 2 for i in range(8)}) == list("02134657"), "GPUs of a node need not be neighbours"
for bad in ({str(i): 0 for i in range(8)}, {str(i): i // 4 for i in range(4)}, {str(i): (0 if i < 5 else 1) for i in range(8)}, {str(i): i // 3 for i in range(8)}):
    try:
        G.order(bad)
    except ValueError:
        continue
    sys.exit(f"no error for {bad}")
print("ALL OK")
