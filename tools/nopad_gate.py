"""Gate for the no-pad framework (--variable-rollout-samples, --bshd-pad-per-sample): compares smoke B (A's rollout
dumps without the masked pad samples, replayed on the no-pad framework) with smoke A (the same two rollouts on the old
framework, with pads). Exit 0 lets the held Frontier-CS chains start (each depends afterok on this job); exit 1 makes
Slurm cancel them (kill_invalid_depend).
  python3 tools/nopad_gate.py A_LOG B_LOG          (tools/nopad_gate.sbatch runs it as a CPU job after smoke B)

A and B split a rollout's real samples into optimizer steps differently (A balanced 36 samples with 24 pads over the
ranks; B balances the 12 real ones), so single critic steps see different samples and are compared only coarsely.
The checks that hold whatever the split:
  1. B ran both replayed rollouts on the no-pad path: critic-steps 0-3 and actor step 1, every metric finite, and the
     "variable rollout" lines with 1 actor step and 2 critic steps per rollout.
  2. Actor step 1 sees every real token of rollout 1 under the initial weights in both runs (rollout 0 trains only the
     critic), so the trainer-vs-rollout log-prob gap must agree: relative difference <= 5% (abs diff), <= 25% (KL),
     ESS ratio within 0.01. Broken per-sample padding or masks would move these far more.
  3. Coarse scale checks, which catch a wrong token normalization (pads counted as tokens or samples gives ~3x): the
     mean critic value loss of the 4 critic steps within 0.67-1.5x of A's, the actor grad norm within 0.33-3x, and
     no critic grad norm above 10x A's largest."""
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
    var = {"actor": [], "critic": []}
    for line in open(b_log, errors="replace"):
        m = VAR.search(line)
        if m:
            var[m.group(1)].append((int(m.group(2)), int(m.group(3))))
    check(len(var["actor"]) >= 1 and {s for _, s in var["actor"]} == {1}, f"B actor on the no-pad path, 1 step per rollout: {sorted(set(var['actor']))}")
    check(len(var["critic"]) >= 2 and {s for _, s in var["critic"]} == {2}, f"B critic on the no-pad path, 2 steps per rollout: {sorted(set(var['critic']))}")
    a1, b1 = A[("step", 1)], B[("step", 1)]
    for name, tol in (("train/train_rollout_logprob_abs_diff", 0.05), ("train/train_rollout_kl", 0.25)):
        check(rel(b1[name], a1[name]) <= tol, f"actor step 1 {name}: A {a1[name]:.6g} B {b1[name]:.6g} rel {rel(b1[name], a1[name]):.2e} <= {tol}")
    d = abs(b1["train/ess_ratio"] - a1["train/ess_ratio"])
    check(d <= 0.01, f"actor step 1 ess_ratio: A {a1['train/ess_ratio']:.5f} B {b1['train/ess_ratio']:.5f} |diff| {d:.2e} <= 0.01")
    r = b1["train/grad_norm"] / a1["train/grad_norm"]
    check(1 / 3 <= r <= 3, f"actor step 1 grad_norm: A {a1['train/grad_norm']:.4g} B {b1['train/grad_norm']:.4g} ratio {r:.3f} in [0.33, 3]")
    va = sum(A[("critic-step", k)]["train/critic-value_loss"] for k in range(4)) / 4
    vb = sum(B[("critic-step", k)]["train/critic-value_loss"] for k in range(4)) / 4
    check(0.67 <= vb / va <= 1.5, f"mean critic value loss: A {va:.4g} B {vb:.4g} ratio {vb / va:.3f} in [0.67, 1.5]")
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
