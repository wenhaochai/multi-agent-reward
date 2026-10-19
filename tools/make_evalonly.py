"""Eval-only copies of a finished oracle-game run (owner 2026-10-13: re-score the final checkpoints with more episodes per
problem and every episode traced, before any new training). Writes the EMIT dirs (run, env, args) that
tools/packed_oracle.sbatch runs: a smoke copy (16 val problems, 1 episode, short turns) and the eval itself.

The eval run loads the source run's newest checkpoint (--load / --critic-load stay the source's, read only), saves
nothing (--save points at its own run dir; miles' eval-only path, --num-rollout 0, evaluates once with the loaded
weights and exits), and evaluates one game set with N episodes per problem. Its config is the source's last training
segment: the '[miles-run] train args:' line and the '[sbatch] env' lines of the source's newest log that has them,
plus the source's runs/<run>/extra_args (folded into the args) and extra_env (copied). Changed: run name, --save,
--critic-save, wandb dir/group/id, --num-rollout 0, --eval-prompt-data, --n-samples-per-eval-prompt, the replay flags
dropped, MA_FO_TRACE_EVERY=1 (every episode traced), MA_FO_TRACE_DIR, and MA_FO_ROUNDS when --rounds is given.
Logs before 2026-10-11 (2e75d3b) do not print the job's MA_/FCS_ env: give it with --env K=V, read off the
submit_fcs_oracle.sh version that submitted the source run (its fo=(...) array); the source's extra_env still applies.

  python3 tools/make_evalonly.py SRC_RUN GAME N TAG OUT_DIR [--rounds R] [--env K=V ...]
      [--fresh] [--module M] [--data PATH --smoke-data PATH] [--set K=V ...] [--smoke-set K=V ...]
  --fresh: the untrained initialization (--load/--critic-load point at the new, empty run dir); --module M: the game
  module's generate/log_eval/log_rollout (e.g. fcs_agent); --data/--smoke-data: the eval sets; --set/--smoke-set: env
  added after the source's (both copies / the smoke only).
  e.g. python3 tools/make_evalonly.py fcs_oracle_team_bonus_q9b-fcs-sft_s42 team 4 ev4 /scratch/.../evalonly/bonus
"""
import argparse
import glob
import os
import re
import shlex
import shutil

B = "/scratch/gpfs/GROUP/USER/project/miles-q38-build"
DATA = {g: f"{B}/data/fcs_val172_fo_{g}.jsonl" for g in ("team", "seq", "par")}
SMOKE_DATA = {g: f"{B}/data/fcs_val16_fo_{g}.jsonl" for g in ("team", "seq", "par")}
SMOKE_ENV = {"MA_FO_BUDGET": "4096", "MA_FO_PLAN_BUDGET": "2048", "MA_FO_REPORT_BUDGET": "4096", "MA_FO_MAX_LEN": "36864"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def source_config(run):
    """(train args, MA_/FCS_ env, miles worktree) of the run's newest log with a train-args line."""
    logs = sorted(glob.glob(f"{B}/logs/miles-{run}-*.out"), key=lambda p: int(re.search(r"-(\d+)\.out$", p).group(1)))
    for log in reversed(logs):
        args = env = miles = None
        env = {}
        for line in open(log, errors="replace"):
            line = ANSI.sub("", line).rstrip("\n")
            if line.startswith("[sbatch] env "):
                k, v = line[len("[sbatch] env "):].split("=", 1)
                env[k] = v
            elif line.startswith("[miles-run] train args: "):
                args = shlex.split(line[len("[miles-run] train args: "):])
            elif line.startswith("[miles-run] ") and " miles=" in line and miles is None:
                miles = re.search(r" miles=(\S+)", line).group(1)
        if args:
            return args, env, miles, log
    raise SystemExit(f"no log of {run} has a train-args line")


def drop(args, flag, nvals):
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 1 + nvals
        else:
            out.append(args[i]); i += 1
    return out


def setf(args, flag, *vals):
    """The flag once, with these values (nargs taken from the values given)."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                i += 1
        else:
            out.append(args[i]); i += 1
    return out + [flag, *vals]


def emit(src, game, n, name, out, rounds, smoke, given, o=None):
    args, env, miles, log = source_config(src)
    assert not (env and given), "the log prints the env: do not give it"
    env = env or given
    assert env.get("MA_FO_GAME") and env.get("MA_FO_BUDGET"), "no MA_FO_* env: pass --env"
    sdir = f"{B}/runs/{src}"
    extra = []
    if os.path.exists(f"{sdir}/extra_args"):
        for line in open(f"{sdir}/extra_args"):
            line = line.split(" #")[0].strip()
            if line and not line.startswith("#"):
                extra += shlex.split(line)
    a = args + extra
    rdir = f"{B}/runs/{name}"
    for f in ("--load", "--critic-load"):
        assert f in a and a[a.index(f) + 1].startswith(sdir), (f, "must load the source run")
    if o is not None and o.fresh:
        a = setf(a, "--load", f"{rdir}/ckpt")
        a = setf(a, "--critic-load", f"{rdir}/ckpt_critic")
    # the eval-only path still builds and loads the critic, though eval never reads it; the critics of the finished runs
    # were deleted (owner 2026-10-18), and a missing iter dir kills init (15794440): then the critic starts from the HF
    # checkpoint, as in --fresh
    cl = a[a.index("--critic-load") + 1]
    tracker = f"{cl}/latest_checkpointed_iteration.txt"
    if cl.startswith(sdir) and (not os.path.exists(tracker) or not os.path.isdir(f"{cl}/iter_{int(open(tracker).read()):07d}")):
        a = setf(a, "--critic-load", f"{rdir}/ckpt_critic")
    if o is not None and o.module:
        for f, fn in (("--custom-generate-function-path", "generate"), ("--custom-eval-rollout-log-function-path", "log_eval"),
                      ("--custom-rollout-log-function-path", "log_rollout")):
            a = setf(a, f, f"miles_team.{o.module}.{fn}")
    for f, nv in (("--replay-rollout-data", 1), ("--replay-rollout-until", 1), ("--skip-eval-before-train", 0)):
        a = drop(a, f, nv)
    a = setf(a, "--save", f"{rdir}/ckpt")
    a = setf(a, "--critic-save", f"{rdir}/ckpt_critic")
    a = setf(a, "--num-rollout", "0")
    # the source checkpoint holds the training run's LR/WD schedule; --num-rollout 0 gives 0 decay iterations, and Megatron
    # asserts the two match on load (15522985). Eval-only takes no optimizer step, so the class values win.
    if "--override-opt_param-scheduler" not in a:
        a.append("--override-opt_param-scheduler")
    data = (o.smoke_data if smoke else o.data) if (o is not None and o.data) else (SMOKE_DATA if smoke else DATA)[game]
    a = setf(a, "--eval-prompt-data", f"fo_{game}", data)
    a = setf(a, "--n-samples-per-eval-prompt", str(n))
    a = setf(a, "--wandb-dir", rdir)
    a = setf(a, "--wandb-group", name)
    a = setf(a, "--wandb-run-id", name)
    e = {k: v for k, v in env.items() if k.startswith(("MA_", "FCS_"))}
    e.update({"MA_FO_TRACE_DIR": f"{rdir}/traces", "MA_FO_TRACE_EVERY": "1"})
    if rounds:
        e["MA_FO_ROUNDS"] = str(rounds)
    if smoke:
        e.update(SMOKE_ENV)
    if o is not None:
        e.update(dict(kv.split("=", 1) for kv in o.set))
        if smoke:
            e.update(dict(kv.split("=", 1) for kv in o.smoke_set))
    base = {"MILES_SRC": miles, "RUN_NAME": name, "GPUS": "8", "MILES_MODEL_TYPE": "qwen3.5-9B"}
    os.makedirs(out, exist_ok=True); os.makedirs(rdir, exist_ok=True)
    if os.path.exists(f"{sdir}/extra_env"):
        shutil.copy(f"{sdir}/extra_env", f"{rdir}/extra_env")
    with open(f"{rdir}/extra_env", "a") as f:
        f.write("# eval-only copy (tools/make_evalonly.py): NVLS off as in every run since 2026-10-08\nNCCL_NVLS_ENABLE=0\n")
    open(f"{out}/run", "w").write(name + "\n")
    open(f"{out}/env", "w").write("".join(f"{k}={v}\n" for k, v in {**base, **e}.items()))
    open(f"{out}/args", "w").write("".join(x + "\n" for x in a))
    print(f"{name}: from {os.path.basename(log)}, {game} x{n}{' rounds ' + str(rounds) if rounds else ''}"
          f"{' (smoke)' if smoke else ''} -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("game", choices=("team", "seq", "par", "agent")); p.add_argument("n", type=int)
    p.add_argument("tag"); p.add_argument("out"); p.add_argument("--rounds", type=int)
    p.add_argument("--env", nargs="*", default=[])
    p.add_argument("--fresh", action="store_true"); p.add_argument("--module")
    p.add_argument("--data"); p.add_argument("--smoke-data")
    p.add_argument("--set", nargs="*", default=[]); p.add_argument("--smoke-set", nargs="*", default=[])
    o = p.parse_args()
    given = dict(kv.split("=", 1) for kv in o.env)
    assert not o.data or o.smoke_data, "--data needs --smoke-data"
    stem = o.src.replace("fcs_oracle_", "") if not o.fresh else "sft"
    emit(o.src, o.game, o.n, f"fcs_evalonly_{o.tag}_{stem}", f"{o.out}/main", o.rounds, False, given, o)
    emit(o.src, o.game, 1, f"fcs_evalonly_smoke_{o.tag}_{stem}", f"{o.out}/smoke", o.rounds, True, given, o)


if __name__ == "__main__":
    main()
