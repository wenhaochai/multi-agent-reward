"""miles --custom-megatron-post-save-hook-path: keep only the newest checkpoint that the actor AND the critic both have.
Runs on a role's rank 0 after its save is finalized. Deletes iter_* dirs older than the newest iteration both roles hold,
in both role dirs (<save> and <save>_critic, the launchers' naming). Before 2026-10-04 each role kept only its own
newest save, so a wall-clock kill between the actor save and the critic save (train.py saves actor, critic, then the
data state) left actor N with critic N-k and no common pair: every chained segment then failed miles' restored-rollout
assert. Now the older pair survives until both roles hold the newer one (one extra actor save on disk for ~1 minute).
In the critic-only phase the actor dir is empty and the critic keeps only its newest save, as before.
Why rotate at all: a full-parameter 9B actor + critic with optimizer state is ~220-250 GB per save and the group disk
is near full; val eval runs in-training, so old checkpoints serve no model selection, only resume.
Use: --custom-megatron-post-save-hook-path miles_team.ckpt_rotate.keep_latest
"""
import re
import shutil
import sys
from pathlib import Path

_ITER = re.compile(r"iter_(\d{7})")


def iterations(role_dir: Path) -> list[int]:
    """Iterations with a finished Megatron checkpoint (its metadata.json); an iter dir holding only debug_events (the
    actor writes one per save point even when it saves no weights) does not count."""
    return sorted(int(m.group(1)) for d in role_dir.glob("iter_*")
                  if (m := _ITER.fullmatch(d.name)) and (d / "metadata.json").is_file())


def sibling(role_dir: Path) -> Path:
    """The other role's checkpoint dir: <save> <-> <save>_critic."""
    n = role_dir.name
    return role_dir.with_name(n[: -len("_critic")]) if n.endswith("_critic") else role_dir.with_name(n + "_critic")


def keep_latest(args, rollout_id, checkpoint_dir, hf_checkpoint_dir=None):
    cur = Path(checkpoint_dir)
    m = _ITER.fullmatch(cur.name)
    if m is None or not cur.is_dir():
        print(f"[ckpt_rotate] skip: unexpected checkpoint dir {cur}", file=sys.stderr, flush=True)
        return
    mine, other = cur.parent, sibling(cur.parent)
    newest = int(m.group(1))
    theirs = iterations(other) if other.is_dir() else []
    # the newest iteration both roles hold; when the other role has nothing (critic-only phase, or a run without a
    # critic) only this role's newest save is kept
    common = max((i for i in theirs if i in set(iterations(mine))), default=None)
    keep = newest if not theirs else (common if common is not None else min(newest, max(theirs)))
    for role_dir in ([mine, other] if theirs else [mine]):
        for i in iterations(role_dir):
            if i < keep:
                shutil.rmtree(role_dir / f"iter_{i:07d}", ignore_errors=True)
                print(f"[ckpt_rotate] removed {role_dir / f'iter_{i:07d}'} (keeping >= {keep})", flush=True)
