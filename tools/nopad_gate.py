"""Gate for the no-pad framework (--variable-rollout-samples, --bshd-pad-per-sample): compares smoke B (A's rollout
dumps without the masked pad samples, replayed on the no-pad framework) with smoke A (the same two rollouts on the old
framework, with pads). Exit 0 lets the held Frontier-CS chains start (each depends afterok on this job); exit 1 makes
Slurm cancel them (kill_invalid_depend).
  python3 tools/nopad_gate.py A_LOG B_LOG          (tools/nopad_gate.sbatch runs it as a CPU job after smoke B)

A and B split a rollout's real samples into optimizer steps differently (A balanced 36 samples with 24 pads over the
ranks; B balances the 12 real ones), so single critic steps see different samples and are compared only coarsely.
The checks that hold whatever the split:
  1. B ran both replayed rollouts on the no-pad path: critic-steps 0-3 and actor step 1, every metric finite, both
     flags on its command line, and every visible "variable rollout" line planning 1 actor step or 2 critic steps.
  2. Actor step 1 sees every real token of rollout 1 under the initial weights in both runs (rollout 0 trains only the
     critic). Exact, whatever the padding layout: ESS ratio = 6646/6655 (A's 6646/6679 without the 24 pads), PPO KL
     and reference KL 0, 6 samples per rank. Within bands that absorb bf16 layout noise: the trainer-vs-rollout log-prob
     gap (abs diff within 15%, KL within 50% of A's; data from another run differs by 32%).
  3. Coarse scale checks, which catch a wrong token normalization (pads counted as tokens or samples gives ~3x): the
     mean critic value loss of the 4 critic steps within 0.5-2x of A's, the actor grad norm within 0.33-3x, and no
     critic grad norm above 10x A's largest."""
import ast
import math
import re
import sys

STEP = re.compile(r"log_utils\.py:\d+ - (critic-step|step) (\d+): (\{.*\})")
VAR = re.compile(r"(actor|critic)_cell\d+_rank\d+\][^\n]*variable rollout: (\d+) local samples -> (\d+) steps")


def metrics(path):
    """{(kind, step): {metric: value}}, first occurrence of each step (model.py repeats the log_utils line)."""
    out = {}
    for line in open(path, errors="replace"):
        m = STEP.search(line)
        if m and (m.group(1), int(m.group(2))) not in out:
            out[(m.group(1), int(m.group(2)))] = parse(m.group(3))
    return out


def parse(s):
    """A logged metrics dict; bare nan/inf (not Python literals) become floats."""
    d = ast.literal_eval(re.sub(r"(?<![\w'\"])(-?)(nan|inf)\b", lambda m: f"'{m.group(1)}{m.group(2)}'", s))
    return {k: float(v) if isinstance(v, str) and v.lstrip("-") in ("nan", "inf") else v for k, v in d.items()}


def rel(b, a):
    return abs(b - a) / max(abs(a), 1e-12)


a_log, b_log = sys.argv[1], sys.argv[2]
A, B = metrics(a_log), metrics(b_log)
fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


need = [("critic-step", k) for k in range(4)] + [("step", 1)]
for key in need:
    check(key in A, f"A has {key[0]} {key[1]}")
    check(key in B, f"B has {key[0]} {key[1]}")
if not fails:
    bad = [f"{k[0]} {k[1]} {n}" for k in need for n, v in B[k].items() if isinstance(v, float) and not math.isfinite(v)]
    check(not bad, f"B metrics finite {bad or ''}")
    # the no-pad path: B's launcher printed both flags, and every visible "variable rollout" line has its role's step
    # count. Ray folds lines of one pattern (words with digits ignored, so actor and critic lines match) from several
    # processes into one "[repeated Nx]" line, so a role may have no visible line; at least one line must show
    train = next((l for l in open(b_log, errors="replace") if "[miles-run] train args:" in l), "")
    extra = next((l for l in open(b_log, errors="replace") if "[miles-run] extra args:" in l), "")
    flags = [f for f in ("--variable-rollout-samples", "--bshd-pad-per-sample") if f not in train + extra]
    check(not flags, f"B launched with both no-pad flags {flags or ''}")
    var = []
    for line in open(b_log, errors="replace"):
        m = VAR.search(line)
        if m:
            var.append((m.group(1), int(m.group(2)), int(m.group(3))))
    want = {"actor": 1, "critic": 2}
    check(var and all(st == want[role] for role, _, st in var), f"B no-pad step plans (role, local samples, steps): {sorted(set(var))}")
    # each replayed rollout has 12 real samples, balanced 6 and 6 over the 2 DP ranks (TP 4 x DP 2)
    check(var and all(n == 6 for _, n, _ in var), f"B ranks hold 6 samples each: {sorted({n for _, n, _ in var})}")
    a1, b1 = A[("step", 1)], B[("step", 1)]
    # B pads each sequence to its own length (A: all to 6144), which moves bf16 sums by an unmeasured amount: wide
    # bands here, exact layout-free checks below
    for name, tol in (("train/train_rollout_logprob_abs_diff", 0.15), ("train/train_rollout_kl", 0.5)):
        check(rel(b1[name], a1[name]) <= tol, f"actor step 1 {name}: A {a1[name]:.6g} B {b1[name]:.6g} rel {rel(b1[name], a1[name]):.2e} <= {tol}")
    # actor step 1 has 6646 loss tokens (3 sequences; --actor-only-overlong-filter masks the 9 truncated ones), and each
    # fully masked row adds 1 to the token denominator: A 6646 / (6646 + 24 pads + 9) = 0.9950591 as logged, so B,
    # without the pads, must log 6646 / 6655 (independent review, 2026-10-04, from A's dumps)
    d = abs(b1["train/ess_ratio"] - 6646 / 6655)
    check(d <= 5e-4, f"actor step 1 ess_ratio: A {a1['train/ess_ratio']:.7f} B {b1['train/ess_ratio']:.7f}, want 6646/6655 = {6646 / 6655:.7f} +- 5e-4")
    # the first step is on-policy against the initial weights: PPO KL and reference KL are exactly 0 in A
    for name in ("train/ppo_kl", "train/kl_loss"):
        check(abs(b1[name]) <= 1e-6, f"actor step 1 {name}: A {a1[name]:.3g} B {b1[name]:.3g}, want 0")
    r = b1["train/grad_norm"] / a1["train/grad_norm"]
    check(1 / 3 <= r <= 3, f"actor step 1 grad_norm: A {a1['train/grad_norm']:.4g} B {b1['train/grad_norm']:.4g} ratio {r:.3f} in [0.33, 3]")
    va = sum(A[("critic-step", k)]["train/critic-value_loss"] for k in range(4)) / 4
    vb = sum(B[("critic-step", k)]["train/critic-value_loss"] for k in range(4)) / 4
    check(0.5 <= vb / va <= 2.0, f"mean critic value loss: A {va:.4g} B {vb:.4g} ratio {vb / va:.3f} in [0.5, 2]")
    ga = max(A[("critic-step", k)]["train/critic-grad_norm"] for k in range(4))
    gb = max(B[("critic-step", k)]["train/critic-grad_norm"] for k in range(4))
    check(gb <= 10 * ga, f"largest critic grad norm: A {ga:.4g} B {gb:.4g} <= 10x A")
    print("per step (A -> B):")
    for k in need:
        for n in ("train/critic-value_loss", "train/critic-grad_norm", "train/pg_loss", "train/grad_norm"):
            if n in A[k]:
                print(f"  {k[0]} {k[1]} {n}: {A[k][n]:.6g} -> {B[k][n]:.6g} (rel {rel(B[k][n], A[k][n]):.2e})")
print("GATE", "FAIL: " + "; ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
