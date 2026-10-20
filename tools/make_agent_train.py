"""EMIT dirs (run, env, args) for RL training of one agent in mini-swe-agent (owner step 2, 2026-10-16: the agentic
single-agent baseline; reward R = S, MA_AG_MU=0 by popup). The base is the agentic probe's config
(tmp/agentic/probe_sft, itself the par_indiv run's: same model, parallelism, context and the paper's Frontier-CS EasyPPO
values), which loads the SFT weights from an empty --load dir. Changed for training:
  main   --num-rollout 60 (the first 10 train only the critic, as every oracle arm), --prompt-data fcs_train200v2_team.jsonl
         (user 2026-10-17: the 200 problems with FrontierSmith PR #4's fixed checkers; none is interactive), 16 problems x 6
         episodes per rollout (the probe's), validation every 20 rollouts on the 98 classic val problems x 2 episodes, --skip-eval-before-train,
         --save-interval 3 (the post-save hook keeps the latest), own run dir, W&B ids and trace dirs, every episode
         traced, and the probe's runs/<run>/extra_env (NVLS off, judge pool) in the new run dir.
  smoke  the same code path at toy size: 3 rollouts (1 critic-only), 2 problems x 2 episodes, 8 steps of 12288 tokens (32768 per episode),
         eval on 8 classic problems x 1 after rollout 1 and at the end, save every rollout (a resume is exercised by
         the main task's first segment only when it loads the smoke's dirs, which it does not: the smoke has its own).
  swarm (user 2026-10-20 popup): N=4 plays miles_team.fcs_agent_team (N agents in one shared repo, shared reward R = S),
  SCORE=last_submit judges the last submitter's snapshot, RB=4 problems per rollout (4 x 6 x 4 agents = the 96 agent
  sessions of the lone agent's 16 x 6); the eval plays the same swarm.
  python3 tools/make_agent_train.py NAME OUT_DIR [SEED=S] [STEPS=K] [SAVE=K] [N=4 SCORE=last_submit RB=4]    e.g. fcs_agent_solo_q9b-fcs-sft_s42 /scratch/.../tmp/agentic/solo_s42"""
import os
import shutil
import sys

B = "/scratch/gpfs/GROUP/USER/project/miles-q38-build"
D = f"{B}/data"
PROBE = "/scratch/gpfs/GROUP/USER/tmp/agentic/probe_sft"
OVR = {}


def setv(a, flag, *vals):
    """Set every occurrence of flag to vals (the launcher reads the last one; keep them equal)."""
    idx = [i for i, x in enumerate(a) if x == flag]
    if not idx:
        return a + [flag, *vals]
    for i in idx:
        a[i + 1:i + 1 + len(vals)] = list(vals)
    return a


def drop(a, flag, n=0):
    while flag in a:
        i = a.index(flag)
        del a[i:i + 1 + n]
    return a


def emit(part, name, out):
    src = f"{PROBE}/{'smoke' if part == 'smoke' else 'main'}"
    a = [x for x in open(f"{src}/args").read().split("\n") if x != ""]
    env = [x for x in open(f"{src}/env").read().split("\n") if x != ""]
    old = open(f"{src}/run").read().strip()
    rdir = f"{B}/runs/{name}"
    a = [x.replace(f"{B}/runs/{old}", rdir) for x in a]
    env = [x.replace(f"{B}/runs/{old}", rdir) for x in env]
    a = drop(a, "--override-opt_param-scheduler")   # eval-only fix: a training run's scheduler must match its ckpt
    a = setv(a, "--wandb-group", name)
    a = setv(a, "--wandb-run-id", name)
    a = setv(a, "--prompt-data", f"{D}/fcs_train200v2_team.jsonl")   # user 2026-10-17: FrontierSmith PR #4 data
    if part == "smoke":
        a = setv(a, "--num-rollout", "3")
        a = setv(a, "--num-critic-only-steps", "1")
        a = setv(a, "--rollout-batch-size", "2")
        a = setv(a, "--n-samples-per-prompt", "2")
        a = setv(a, "--eval-interval", "1")
        a = setv(a, "--save-interval", "1")
        a = setv(a, "--eval-prompt-data", "fo_agent", f"{D}/fcs_val_classic_agent_smoke8.jsonl")
        a = setv(a, "--n-samples-per-eval-prompt", "1")
    else:
        a = setv(a, "--num-rollout", "60")
        a = setv(a, "--num-critic-only-steps", "10")
        a = setv(a, "--rollout-batch-size", "16")
        a = setv(a, "--n-samples-per-prompt", "6")
        a = setv(a, "--eval-interval", "20")
        a = setv(a, "--save-interval", "3")       # review 2026-10-16: a wall kill loses at most 3 rollouts
        if "--skip-eval-before-train" not in a:     # the probe dropped it (eval-only); without it every segment re-evaluates
            a.append("--skip-eval-before-train")
        a = setv(a, "--eval-prompt-data", "fo_agent", f"{D}/fcs_val_classic_agent.jsonl")
        a = setv(a, "--n-samples-per-eval-prompt", "2")
    keep = [e for e in env if not e.startswith(("RUN_NAME=", "MA_AG_MU=", "MA_FO_TRACE_EVERY="))]
    env = [f"RUN_NAME={name}"] + keep + ["MA_AG_MU=0", "MA_FO_TRACE_EVERY=1"]
    if "SEED" in OVR:
        a = setv(a, "--seed", OVR["SEED"])
        a = setv(a, "--rollout-seed", OVR["SEED"])
    if "SAVE" in OVR and part != "smoke":
        a = setv(a, "--save-interval", OVR["SAVE"])
    if "N" in OVR:   # the swarm: same log functions (fcs_agent_team re-exports fcs_agent's)
        a = setv(a, "--custom-generate-function-path", "miles_team.fcs_agent_team.generate")
        env = [e for e in env if not e.startswith(("MA_AG_N=", "MA_AG_REWARD=", "MA_AG_SCORE="))] + [
            f"MA_AG_N={OVR['N']}", "MA_AG_REWARD=shared", f"MA_AG_SCORE={OVR.get('SCORE', 'repo')}"]
    if "RB" in OVR and part != "smoke":
        a = setv(a, "--rollout-batch-size", OVR["RB"])
    if "STEPS" in OVR:
        env = [e for e in env if not e.startswith("MA_AG_STEPS=")] + [f"MA_AG_STEPS={OVR['STEPS']}"]
    if part == "smoke":   # at 4096 tokens per step the probe smoke never submitted: the training episodes must reach the judge
        env = [e for e in env if not e.startswith(("MA_AG_TURN=", "MA_AG_TOKENS="))] + ["MA_AG_TURN=12288", "MA_AG_TOKENS=32768"]
    os.makedirs(out, exist_ok=False)
    # runs/<run>/extra_env (exported before ray start): NVLS off as in every run since 2026-10-08 (CUDA 999 after wake_up)
    # and the judge's shared pool of 56 slots (eval tails); the probe never trained, so it never hit the first case
    os.makedirs(rdir, exist_ok=True)
    shutil.copy(f"{B}/runs/{old}/extra_env", f"{rdir}/extra_env")
    open(f"{out}/args", "w").write("\n".join(a) + "\n")
    open(f"{out}/env", "w").write("\n".join(env) + "\n")
    open(f"{out}/run", "w").write(name + "\n")
    assert not any(old in x for x in a + env), "a probe path is left"
    print(part, name, len(a), "args ->", out)


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    for kv in sys.argv[3:]:   # SEED=S (--seed/--rollout-seed; engines take seed+i, so space seeds >= 8), STEPS=K (MA_AG_STEPS),
        # SAVE=K (main --save-interval; 1 for fetchable-partial segments, user 2026-10-19)
        k, v = kv.split("=", 1)
        assert k in ("SEED", "STEPS", "SAVE", "N", "SCORE", "RB"), kv
        OVR[k] = v
    if os.path.exists(out):
        sys.exit(f"{out} exists: remove it by hand first")
    smoke = name.replace("fcs_agent_", "fcs_agent_smoke_", 1)
    assert smoke != name
    emit("smoke", smoke, f"{out}/smoke")
    emit("main", name, f"{out}/main")
