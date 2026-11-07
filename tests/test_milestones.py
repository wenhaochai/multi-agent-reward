"""Milestone checkpoints of the train67 run (MA_RL_KEEP_EVERY=20, rl/rl_inner.sh; flint.miles_agent's pruning): a fake
actor dir written as miles writes it (iter_k = the checkpoint after rollout k - 1, every 2 steps: model/, optimizer/,
lr_scheduler/, rng.pt, meta.json, then the tracker), the pruning called as log_rollout calls it (after each rollout,
before its training step) for a 44-step run, then once more as rl_inner.sh does after train.py: the newest 2 full
checkpoints (iter_42, iter_44) stay untouched and milestones/iter_20, iter_40 hold the model part only (hard links: the
same bytes, no optimizer); a resume reads the tracker's checkpoint (miles' FSDP load path: root/iter_<tracker>/model);
a checkpoint miles is still writing (model/ there, no tracker yet) gets no milestone; MA_RL_KEEP_EVERY=0 (the other
modes) keeps none; a second pass changes nothing. Runs in miles-v0.1.3.sif (tests/run_all.sh) or any python 3.8+."""
import json
import os
import subprocess
import sys
import tempfile

from helpers import ROOT, check, done, env

env(MA_LG_N=1, MA_LG_TOKENS=90, MA_AG_TOOL_STRICT=0)
from flint import miles_agent as MA  # noqa: E402

FILES = {"model": ["__0_0.distcp", "__1_0.distcp", ".metadata"], "optimizer": ["__0_0.distcp", ".metadata"],
         "lr_scheduler": ["__0_0.distcp", ".metadata"]}


def save(root, k, tracker=True):
    """miles' FSDP save of iter_k: model, optimizer, scheduler, rng, meta, then the tracker."""
    d = os.path.join(root, f"iter_{k:07d}")
    for part, names in FILES.items():
        os.makedirs(os.path.join(d, part), exist_ok=True)
        for n in names:
            open(os.path.join(d, part, n), "w").write(f"{part} {n} iter {k}\n")
    open(os.path.join(d, "rng.pt"), "w").write(f"rng {k}\n")
    json.dump({"iteration": k, "rollout_id": k - 1, "next_rollout_id": k}, open(os.path.join(d, "meta.json"), "w"))
    if tracker:
        open(os.path.join(root, "latest_checkpointed_iteration.txt"), "w").write(str(k))


def listing(d):
    return sorted(os.path.relpath(os.path.join(a, f), d) for a, _, fs in os.walk(d) for f in fs)


def run(every, steps=44, keep=2):
    out = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
    root = os.path.join(out, "actor")
    os.makedirs(root)
    for r in range(steps):   # train.py: rollout r (log_rollout prunes), training step r, save iter_{r+1} if even
        MA._keep_newest_checkpoints(root, keep, every)
        if (r + 1) % 2 == 0:
            save(root, r + 1)
    MA._keep_newest_checkpoints(root, keep, every)   # rl_inner.sh after train.py
    return out, root


out, root = run(20)
ms = os.path.join(out, "milestones")
check("prune: the newest 2 full checkpoints stay, nothing else in actor/",
      sorted(os.listdir(root)) == ["iter_0000042", "iter_0000044", "latest_checkpointed_iteration.txt"], os.listdir(root))
full = ["lr_scheduler/.metadata", "lr_scheduler/__0_0.distcp", "meta.json", "model/.metadata", "model/__0_0.distcp",
        "model/__1_0.distcp", "optimizer/.metadata", "optimizer/__0_0.distcp", "rng.pt"]
check("prune: iter_42 and iter_44 whole (model, optimizer, scheduler, rng, meta)",
      all(listing(os.path.join(root, f"iter_{k:07d}")) == full for k in (42, 44)))
check("milestones: iter_20 and iter_40 only", os.path.isdir(ms) and sorted(os.listdir(ms)) == ["iter_0000020", "iter_0000040"],
      os.listdir(ms) if os.path.isdir(ms) else None)
mfiles = ["meta.json", "model/.metadata", "model/__0_0.distcp", "model/__1_0.distcp"]
check("milestones: the model part and meta.json only (no optimizer, scheduler, rng)",
      all(listing(os.path.join(ms, f"iter_{k:07d}")) == mfiles for k in (20, 40)))
check("milestones: the bytes of their own iteration",
      all(open(os.path.join(ms, f"iter_{k:07d}", "model", "__1_0.distcp")).read() == f"model __1_0.distcp iter {k}\n"
          and json.load(open(os.path.join(ms, f"iter_{k:07d}", "meta.json")))["iteration"] == k for k in (20, 40)))
check("milestones: no .partial left", not any(x.endswith(".partial") for x in os.listdir(ms)))
# the resume: miles' FSDP load (miles/backends/fsdp_utils/checkpoint.py) reads root/iter_<tracker>/{model,optimizer,...}
k = int(open(os.path.join(root, "latest_checkpointed_iteration.txt")).read())
check("resume: the tracker names the newest full checkpoint, iter_44, with model and optimizer",
      k == 44 and all(os.path.isdir(os.path.join(root, f"iter_{k:07d}", p)) for p in ("model", "optimizer", "lr_scheduler")))
try:   # miles' own loader on the fake dir, up to the DCP read (which needs a real model)
    from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration
    check("resume: miles' tracker reader picks iter 44", read_checkpoint_tracker_iteration(root) == 44)
except ImportError as e:
    print("skip miles' tracker reader:", e)
before = {x: listing(os.path.join(out, x)) for x in ("actor", "milestones")}
MA._keep_newest_checkpoints(root, 2, 20)
check("a second pass changes nothing", before == {x: listing(os.path.join(out, x)) for x in ("actor", "milestones")})
# a 60-step run: iter_60 comes after the last rollout's hook; only the pass after train.py keeps it
out60, root60 = run(20, 60)
check("60 steps: milestones 20, 40, 60 (60 from the pass after train.py)",
      sorted(os.listdir(os.path.join(out60, "milestones"))) == ["iter_0000020", "iter_0000040", "iter_0000060"])
# a checkpoint miles is still writing: model/ there, tracker not yet moved to it
out2 = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
root2 = os.path.join(out2, "actor")
os.makedirs(root2)
save(root2, 18)
save(root2, 20, tracker=False)
MA._keep_newest_checkpoints(root2, 2, 20)
check("iter_20 in writing (tracker still at 18): no milestone yet, nothing deleted",
      not os.path.exists(os.path.join(out2, "milestones", "iter_0000020"))
      and sorted(os.listdir(root2)) == ["iter_0000018", "iter_0000020", "latest_checkpointed_iteration.txt"])
open(os.path.join(root2, "latest_checkpointed_iteration.txt"), "w").write("20")
MA._keep_newest_checkpoints(root2, 2, 20)
check("once the tracker names iter_20: its milestone", os.path.isdir(os.path.join(out2, "milestones", "iter_0000020")))
# off: the other modes
out0, root0 = run(0)
check("MA_RL_KEEP_EVERY=0: no milestones, the newest 2 as before",
      not os.path.exists(os.path.join(out0, "milestones")) and sorted(os.listdir(root0))[:2] == ["iter_0000042", "iter_0000044"])
# rl_inner.sh: MA_RL_KEEP_EVERY=20 in the train67 mode only, and its pass after train.py
src = open(os.path.join(ROOT, "rl", "rl_inner.sh")).read()
check("rl_inner.sh: MA_RL_KEEP_EVERY=20 for train67 only", '[ "$MODE" = train67 ] && export MA_RL_KEEP_EVERY=20' in src
      and src.count("MA_RL_KEEP_EVERY=") == 1)
check("log_rollout passes MA_RL_KEEP_EVERY to the pruning",
      'int(os.environ.get("MA_RL_KEEP_EVERY", "0") or 0)' in open(os.path.join(ROOT, "flint", "miles_agent.py")).read())
# KEEP_CHECKPOINTS = 1 (2026-11-07): the newest only, the tracker's one too while miles writes the next
check("KEEP_CHECKPOINTS is 1, used by log_rollout and by rl_inner.sh's pass after train.py",
      MA.KEEP_CHECKPOINTS == 1 and "KEEP_CHECKPOINTS, int(os.environ.get(\"MA_RL_KEEP_EVERY\"" in
      open(os.path.join(ROOT, "flint", "miles_agent.py")).read() and "KEEP_CHECKPOINTS as n; k('$OUT/actor', n," in src)
out1, root1 = run(20, keep=1)
check("keep 1: iter_44 alone in actor/, whole; milestones 20 and 40",
      sorted(os.listdir(root1)) == ["iter_0000044", "latest_checkpointed_iteration.txt"]
      and listing(os.path.join(root1, "iter_0000044")) == full
      and sorted(os.listdir(os.path.join(out1, "milestones"))) == ["iter_0000020", "iter_0000040"], os.listdir(root1))
out3 = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
root3 = os.path.join(out3, "actor")
os.makedirs(root3)
save(root3, 16)
save(root3, 18)
save(root3, 20, tracker=False)
MA._keep_newest_checkpoints(root3, 1, 20)
check("keep 1, iter_20 in writing (tracker at 18): iter_18 and iter_20 stay, iter_16 goes, no milestone yet",
      sorted(os.listdir(root3)) == ["iter_0000018", "iter_0000020", "latest_checkpointed_iteration.txt"]
      and not os.path.exists(os.path.join(out3, "milestones", "iter_0000020")), os.listdir(root3))
open(os.path.join(root3, "latest_checkpointed_iteration.txt"), "w").write("20")
MA._keep_newest_checkpoints(root3, 1, 20)
check("keep 1, once the tracker names iter_20: iter_20 alone, and its milestone",
      sorted(os.listdir(root3)) == ["iter_0000020", "latest_checkpointed_iteration.txt"]
      and os.path.isdir(os.path.join(out3, "milestones", "iter_0000020")), os.listdir(root3))
r = subprocess.run(["bash", "-n", os.path.join(ROOT, "rl", "rl_inner.sh")], capture_output=True, text=True)
check("rl_inner.sh parses", r.returncode == 0, r.stderr)
done()
