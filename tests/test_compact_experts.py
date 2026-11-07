"""The buffer's routed experts as uint8 (flint.elastic_rollout compact_routed_experts / expand_routed_experts; host memory
for kl=B's reference server, 2026-11-06). (1) A group's int32 ids within 0-255 become uint8 with the same values, the
bytes saved are reported, and an array shared by two samples is compacted once. (2) An array with an id outside 0-255,
a sample without routed experts and an empty array stay as they are. (3) The drain's expansion gives int32 arrays equal
to the originals, and samples that shared an array (the sample filter's split copies) share the expanded one.
(4) On w1400's dumped step-0 samples (real ids, 49 layers x top 6): compaction then expansion gives back every array
exactly. python3 tests/test_compact_experts.py in miles-v0.1.3.sif with the run's miles on PYTHONPATH (no GPU)."""
import glob
import os
from types import SimpleNamespace

import numpy as np

from helpers import check, done, env

env()
from flint import elastic_rollout as ER  # noqa: E402

rng = np.random.default_rng(0)


def smp(arr):
    return SimpleNamespace(rollout_routed_experts=arr)


a = rng.integers(0, 256, size=(50, 49, 6), dtype=np.int32)
b = rng.integers(0, 256, size=(20, 49, 6), dtype=np.int32)
wide = a.copy(); wide[3, 7, 2] = 300
s1, s2, s3 = smp(a.copy()), smp(b.copy()), smp(None)
twin = smp(s1.rollout_routed_experts)   # a split copy shares its source's array
s_wide, s_empty = smp(wide.copy()), smp(np.zeros((0, 49, 6), dtype=np.int32))
group = [[s1, twin], s2, [s3, s_wide, s_empty]]
saved = ER.compact_routed_experts(group)
check("compact: int32 ids within 0-255 become uint8 with the same values",
      s1.rollout_routed_experts.dtype == np.uint8 and np.array_equal(s1.rollout_routed_experts, a)
      and s2.rollout_routed_experts.dtype == np.uint8 and np.array_equal(s2.rollout_routed_experts, b))
check("compact: the bytes saved are 3/4 of the compacted arrays", saved == (a.nbytes + b.nbytes) * 3 // 4, saved)
check("compact: a shared array is compacted once and stays shared", twin.rollout_routed_experts is s1.rollout_routed_experts)
check("compact: an id outside 0-255, no array, an empty array stay as they are",
      s_wide.rollout_routed_experts.dtype == np.int32 and np.array_equal(s_wide.rollout_routed_experts, wide)
      and s3.rollout_routed_experts is None and s_empty.rollout_routed_experts.dtype == np.int32)
check("compact: a second call changes nothing", ER.compact_routed_experts(group) == 0)
ER.expand_routed_experts([group])
check("expand: int32 again, equal to the originals",
      all(x.rollout_routed_experts.dtype == np.int32 for x in (s1, twin, s2, s_wide))
      and np.array_equal(s1.rollout_routed_experts, a) and np.array_equal(s2.rollout_routed_experts, b))
check("expand: samples that shared an array share the expanded one", twin.rollout_routed_experts is s1.rollout_routed_experts)

R = "/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/train67_signal33_kgate-graphs-tp4_dupend50_e_a2_p4g8/r0/samples"
files = sorted(glob.glob(f"{R}/*.pt"))[:2]
if files:
    import torch

    real, orig = [], []
    for f in files:
        for d in torch.load(f, weights_only=False)["samples"]:
            orig.append(np.array(d["rollout_routed_experts"], copy=True))
            real.append(smp(d["rollout_routed_experts"]))
    saved = ER.compact_routed_experts([real])
    after_bytes = sum(x.rollout_routed_experts.nbytes for x in real)
    ER.expand_routed_experts([real])
    check(f"real samples ({len(real)} of w1400's step 0): every array back exactly, {saved / 2**30:.2f} GiB saved while compact",
          all(x.rollout_routed_experts.dtype == np.int32 and np.array_equal(x.rollout_routed_experts, o) for x, o in zip(real, orig))
          and saved == 3 * after_bytes, (saved, after_bytes))
else:
    check("real samples: w1400's step-0 dump is readable", False, R)
done()
