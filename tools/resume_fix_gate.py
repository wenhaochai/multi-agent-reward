"""Gate for the actor resume fix (miles resume-fix bc4948ad1): smoke B's run resumed from its iteration-1 checkpoint,
whose actor model weights are all zero (saved before the fix), and ran rollout 2 live. On exit 0 the sbatch wrapper
fast-forwards the miles worktree the Frontier-CS jobs import (miles-easyppo) to the fix; the resumed chains depend
afterok on that job, so a failure makes Slurm cancel them (kill_invalid_depend).
  python3 tools/resume_fix_gate.py SMOKE_LOG CKPT_DIR FIX_SHA [STEP]  (inside miles.sif; tools/resume_fix_gate.sbatch)
STEP: the first actor step after the resume (2 for the smoke).

Without the fix the first actor step after a resume runs on an all-zero model: the trainer and the engine both give
uniform log-probs (train-vs-rollout gap exactly 0), the gradient is 0 and the reference KL is ~1.1. With it they look
like any later step: gap ~0.01, KL to the reference ~0, gradient > 0. The checkpoint this run saves must hold a nonzero
actor model."""
import ast
import math
import re
import subprocess
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

log, ckpt, fix = sys.argv[1], sys.argv[2], sys.argv[3]
first = int(sys.argv[4]) if len(sys.argv) > 4 else 2
STEP = re.compile(r"log_utils\.py:\d+ - (critic-step|step) (\d+): (\{.*\})")
fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


def parse(s):
    d = ast.literal_eval(re.sub(r"(?<![\w'\"])(-?)(nan|inf)\b", lambda m: f"'{m.group(1)}{m.group(2)}'", s))
    return {k: float(v) if isinstance(v, str) and v.lstrip("-") in ("nan", "inf") else v for k, v in d.items()}


steps = {}
code = ""
for line in open(log, errors="replace"):
    m = STEP.search(line)
    if m and (m.group(1), int(m.group(2))) not in steps:
        steps[(m.group(1), int(m.group(2)))] = parse(m.group(3))
    if "[miles-run] code:" in line and not code:
        code = line.strip()
check(f"miles {fix[:9]}" in code, f"the smoke ran the fix: {code[-90:]}")
check(("step", first) in steps, f"actor step {first} (the first step after the resume) logged")
if ("step", first) in steps:
    a = steps[("step", first)]
    gap, kl, gn = a["train/train_rollout_logprob_abs_diff"], a["train/kl_loss"], a["train/grad_norm"]
    check(math.isfinite(gap) and 0.002 < gap < 0.05, f"train-vs-rollout log-prob gap {gap:.5f} in (0.002, 0.05) (all-zero model: 0)")
    check(math.isfinite(kl) and kl < 0.05, f"reference KL {kl:.5f} < 0.05 (all-zero model: ~1.1)")
    check(math.isfinite(gn) and gn > 0.01, f"actor grad norm {gn:.4f} > 0.01 (all-zero model: 0)")
crit = [v for (k, i), v in steps.items() if k == "critic-step" and i >= 2 * first]
check(len(crit) >= 2 and all(math.isfinite(c["train/critic-value_loss"]) for c in crit), f"critic steps after the resume logged and finite ({len(crit)})")

md = FileSystemReader(ckpt).read_metadata()
names = [k for k in md.state_dict_metadata if k in ("decoder.final_layernorm.weight", "decoder.layers.0.mlp.linear_fc2.weight", "output_layer.weight")]
check(len(names) == 3, f"saved actor checkpoint {ckpt} has the model tensors")
for k in names:
    m = md.state_dict_metadata[k]
    sd = {k: torch.empty(m.size, dtype=m.properties.dtype)}
    dcp.load(sd, storage_reader=FileSystemReader(ckpt), no_dist=True)
    t = sd[k].float()
    check(t.abs().mean().item() > 1e-4 and (t == 0).float().mean().item() < 0.5,
          f"{k}: absmean {t.abs().mean():.4g}, zero fraction {(t == 0).float().mean():.3f}")

print("GATE", "FAIL: " + "; ".join(fails) if fails else "PASS", flush=True)
sys.exit(1 if fails else 0)
