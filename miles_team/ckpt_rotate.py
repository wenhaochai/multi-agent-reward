"""miles --custom-megatron-post-save-hook-path: keep only the newest Megatron checkpoint of this role (actor or critic).
Runs on the role's rank 0 after its save is finalized; deletes sibling iter_* dirs older than the one just written.
Why: a full-parameter 9B actor + critic with optimizer state is ~250 GB per save and the group disk is near full;
val eval runs in-training (every --eval-interval), so old checkpoints serve no model selection, only resume.
Use: --custom-megatron-post-save-hook-path miles_team.ckpt_rotate.keep_latest
"""
import re
import shutil
import sys
from pathlib import Path


def keep_latest(args, rollout_id, checkpoint_dir, hf_checkpoint_dir=None):
    cur = Path(checkpoint_dir)
    m = re.fullmatch(r"iter_(\d{7})", cur.name)
    if m is None or not cur.is_dir():
        print(f"[ckpt_rotate] skip: unexpected checkpoint dir {cur}", file=sys.stderr, flush=True)
        return
    for d in cur.parent.glob("iter_*"):
        dm = re.fullmatch(r"iter_(\d{7})", d.name)
        if dm and int(dm.group(1)) < int(m.group(1)):
            shutil.rmtree(d, ignore_errors=True)
            print(f"[ckpt_rotate] removed {d}", flush=True)
