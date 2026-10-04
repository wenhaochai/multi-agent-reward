"""Copy a run's training rollout dumps (rollout_data/<rollout_id>.pt from --save-debug-rollout-data) without the masked
padding samples (metadata.fo_pad), for replay under miles --variable-rollout-samples (the no-pad equivalence test).
  python3 tools/strip_pad_samples.py <src rollout_data dir> <dst dir>"""
import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("*.pt")):
    if not f.stem.isdigit():  # eval dumps keep their own names
        continue
    d = torch.load(f, weights_only=False)
    keep = [s for s in d["samples"] if not (s.get("metadata") or {}).get("fo_pad")]
    torch.save({**d, "samples": keep}, dst / f.name)
    print(f"{f.name}: {len(d['samples'])} -> {len(keep)} samples")
