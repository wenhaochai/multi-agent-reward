"""Mutation check of the miles fork's masked fused head tests: each purpose-made bug must fail tests/fast/backends/
fsdp_utils/test_flint_fused_head.py; the source is restored after each. Run from the miles worktree, in
miles-v0.1.3.sif with this repository on PYTHONPATH (flint_hf): python <this file>; prints MUTATIONS_OK."""
import subprocess, sys
LP = "miles/backends/training_utils/loss/hub/logit_processors.py"
SC = "miles/backends/training_utils/loss/hub/score_centering_loss.py"
AC = "miles/backends/fsdp_utils/actor.py"
M = [
    ("no renormalization in the score-centering head", SC,
     "result[\"selected\"].append(support_log_probs_from_head(chunk.float(), ids[:, 1:] >= 0)[0])",
     "result[\"selected\"].append(torch.where(ids >= 0, chunk, 0.0))"),
    ("support width off by one in the log-prob pass", LP,
     "valid = torch.arange(width, device=chunk.device) < lengths[:, None]",
     "valid = torch.arange(width, device=chunk.device) <= lengths[:, None]"),
    ("normalizer over the sampled-token column too", LP,
     "support = head[:, 1:].masked_fill(~valid, -torch.inf)",
     "support = torch.cat((head[:, :1], head[:, 1:].masked_fill(~valid, -torch.inf)), -1)"),
    ("certain rows keep autograd's rounding", LP,
     "& ~certain\n", "\n"),
    ("support rows in reverse order", AC,
     "start, lengths = offsets[indices], offsets[indices + 1] - offsets[indices]",
     "indices = indices.flip(0)\n    start, lengths = offsets[indices], offsets[indices + 1] - offsets[indices]"),
    ("entropy over the full head row", LP,
     "return logp, -(p * logp[:, 1:]).sum(-1)",
     "return logp, -(head[:, 1:].exp() * head[:, 1:]).sum(-1)"),
]
bad = 0
for name, path, a, b in M:
    src = open(path).read()
    assert src.count(a) == 1, name
    open(path, "w").write(src.replace(a, b))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/fast/backends/fsdp_utils/test_flint_fused_head.py"], capture_output=True, text=True)
    finally:
        open(path, "w").write(src)
    caught = r.returncode != 0
    bad += not caught
    print(("CAUGHT " if caught else "MISSED ") + name, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
print("MUTATIONS_OK" if not bad else f"MUTATIONS_MISSED {bad}")
