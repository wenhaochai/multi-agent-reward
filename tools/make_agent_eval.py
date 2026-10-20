"""EMIT dirs (run, env, args) for eval-only copies of the agentic probe (user 2026-10-19 by popup: the agent rows of the
agentic figure on the ev4 footing). The base is tmp/agentic/probe_sft (the untrained probe: 98 classic val problems,
mini-swe-agent, fcs_agent), changed to:
  --load       CKPT (a run's ckpt dir with latest_checkpointed_iteration.txt) or, with "sft", the new run's empty dir
               (the SFT weights from --megatron-hf-checkpoint)
  --critic-load the new run's empty dir: the eval-only path builds and loads the critic but never reads it, and the
               finished runs' critics are deleted (owner 2026-10-18); it starts from the HF checkpoint (15794440)
  N episodes per problem (main; the smoke keeps 8 problems x 1), own run dir, W&B ids, trace dirs, MA_AG_STEPS=0 (user
  2026-10-18: no step limit; the agent run trains without one from rollout 21), runs/<run>/extra_env from the probe.
  python3 tools/make_agent_eval.py NAME CKPT|sft N OUT_DIR [--module M] [KEY=VALUE ...]   (env set last)
  e.g. python3 tools/make_agent_eval.py fcs_evalonly_agentev4_sft sft 4 /scratch/.../tmp/agentic/ev4_sft"""
import os
import shutil
import sys

B = "/scratch/gpfs/GROUP/USER/project/miles-q38-build"
PROBE = "/scratch/gpfs/GROUP/USER/tmp/agentic/probe_sft"
EVAL_MILES = "/scratch/gpfs/GROUP/USER/project/miles-easyppo-evalonly"   # miles evalonly-critic a8f2c0cef


def setv(a, flag, *vals):
    """Set every occurrence of flag to vals (the launcher reads the last one; keep them equal)."""
    idx = [i for i, x in enumerate(a) if x == flag]
    assert idx, flag
    for i in idx:
        a[i + 1:i + 1 + len(vals)] = list(vals)
    return a


def emit(part, name, ckpt, n, out, module="fcs_agent", sets=()):
    src = f"{PROBE}/{part}"
    a = [x for x in open(f"{src}/args").read().split("\n") if x != ""]
    env = [x for x in open(f"{src}/env").read().split("\n") if x != ""]
    old = open(f"{src}/run").read().strip()
    rdir = f"{B}/runs/{name}"
    a = [x.replace(f"{B}/runs/{old}", rdir) for x in a]
    env = [x.replace(f"{B}/runs/{old}", rdir) for x in env]
    if ckpt != "sft":
        assert os.path.isdir(f"{ckpt}/iter_{int(open(f'{ckpt}/latest_checkpointed_iteration.txt').read()):07d}"), ckpt
        a = setv(a, "--load", ckpt)
        # a trained checkpoint holds its run's LR/WD schedule (15841612/13 died on Megatron's match assert); its critic
        # checkpoint is gone, so the eval runs miles evalonly-critic (no actor/critic restored-rollout assert at
        # --num-rollout 0; the shared miles-easyppo stays untouched for the running chains)
        if "--override-opt_param-scheduler" not in a:
            a.append("--override-opt_param-scheduler")
        env = [e for e in env if not e.startswith("MILES_SRC=")] + [f"MILES_SRC={EVAL_MILES}"]
    a = setv(a, "--critic-load", f"{rdir}/ckpt_critic")
    a = setv(a, "--wandb-group", name)
    a = setv(a, "--wandb-run-id", name)
    if part == "main":
        a = setv(a, "--n-samples-per-eval-prompt", str(n))
    env = [e for e in env if not e.startswith(("RUN_NAME=", "MA_AG_STEPS="))]
    env = [f"RUN_NAME={name}"] + env + ["MA_AG_STEPS=0"]
    if module != "fcs_agent":   # e.g. fcs_agent_team: N agents in one shared repo (log functions are fcs_agent's)
        a = setv(a, "--custom-generate-function-path", f"miles_team.{module}.generate")
    keys = [kv.split("=", 1)[0] + "=" for kv in sets]
    env = [e for e in env if not e.startswith(tuple(keys))] + list(sets)
    assert not any(old in x for x in a + env), "a probe path is left"
    os.makedirs(out, exist_ok=False)
    os.makedirs(rdir, exist_ok=True)
    shutil.copy(f"{B}/runs/{old}/extra_env", f"{rdir}/extra_env")
    open(f"{out}/args", "w").write("\n".join(a) + "\n")
    open(f"{out}/env", "w").write("\n".join(env) + "\n")
    open(f"{out}/run", "w").write(name + "\n")
    print(part, name, "load", a[a.index("--load") + 1], "->", out)


if __name__ == "__main__":
    name, ckpt, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    rest = sys.argv[5:]
    module = rest[rest.index("--module") + 1] if "--module" in rest else "fcs_agent"
    sets = [x for x in rest if "=" in x and x.split("=", 1)[0].isupper()]
    if os.path.exists(out):
        sys.exit(f"{out} exists: remove it by hand first")
    smoke = name.replace("fcs_evalonly_", "fcs_evalonly_smoke_", 1)
    assert smoke != name
    emit("smoke", smoke, ckpt, 1, f"{out}/smoke", module, sets)
    emit("main", name, ckpt, n, f"{out}/main", module, sets)
