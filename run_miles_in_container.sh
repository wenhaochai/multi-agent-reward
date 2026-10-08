#!/bin/bash
# Runs INSIDE miles.sif (radixark/miles:latest, built 2026-09-30; its /root/miles is 874b3e275 and MILES_SRC is checked out there) (sbatch_miles.sh). miles' own launcher (command_utils) starts with `pkill -9 sglang; ray stop
# --force; pkill -9 ray`, which on a shared Della node kills other jobs' processes, so this script does its job by hand:
# a job-private Ray head on job-derived ports (Della memory reference_slime_della / reference_della_ray_shared_node),
# every env var the Ray workers need exported BEFORE `ray start` (workers inherit the raylet's env), then
# `python3 train.py <model args> <train args>` as the driver. Args: MILES_MODEL_TYPE (scripts/models/<type>.py) and
# the train args ("$@").
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
C=${MQB_CODE:-$B}   # the code (miles_team, tools); runs/ and logs/ stay under $B. A candidate worktree for a smoke
MILES=${MILES_SRC:-/scratch/gpfs/GROUP/USER/project/miles-q38}
: "${RUN_NAME:?}" "${SLURM_JOB_ID:?}" "${MILES_MODEL_TYPE:?}"
RUN_DIR=$B/runs/$RUN_NAME
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
# /tmp on a compute node is small and shared by every job there (owner rule 2026-10-08): temp files, compiles, the judge
# and the caches go to a job dir on /scratch (sbatch_miles.sh makes it and binds it; jobs queued before that change get
# it here), removed when this script exits. Ray alone stays on /tmp (below): its unix socket paths must stay < 108 bytes
JT=/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID${QTASK:+_q$QTASK}
export TMPDIR=$JT HOME=$JT/home TRITON_CACHE_DIR=$JT/home/.cache/triton SGLANG_CACHE_DIR=$JT/home/.cache/sglang
export TORCHINDUCTOR_CACHE_DIR=$JT/torchinductor FCS_JUDGE_TMP=$JT FCS_JUDGE_CACHE=$JT/fcs_judge_cache
mkdir -p "$HOME" "$FCS_JUDGE_CACHE"
echo "[miles-run] TMPDIR=$TMPDIR (judge, compiles, caches); Ray on /tmp"
MEGATRON=/root/Megatron-LM   # the image's patched Megatron (a namespace package: megatron.__file__ is None)
# pyfix/ first: the image (radixark/miles:latest, 2026-09-30) ships opentelemetry-api 1.45.0 with sdk/exporters 1.44.0, so
# Ray's dashboard agent dies at import ('_ExtendedAttributes') and `ray start` times out (smoke 14791336). pyfix holds
# opentelemetry-api 1.44.0 (tools: pip download + pip install --target, see README), matching the rest.
export PYTHONPATH=$B/pyfix:$MEGATRON:$MILES:$C${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 MASTER_ADDR=127.0.0.1 no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_USAGE_STATS_ENABLED=0
export NCCL_NVLS_ENABLE=$(nvidia-smi topo -m 2>/dev/null | grep -c 'NV[0-9]' | awk '{print ($1 > 0) ? 1 : 0}')
# runs/<run>/extra_env: KEY=VALUE lines ('#' ignored), exported before `ray start` so every Ray worker sees them (e.g.
# FCS_EXTRACT=longest pins a run that started under the old code-block rule)
if [ -f "$RUN_DIR/extra_env" ]; then
  while IFS= read -r kv; do case "$kv" in ''|'#'*) ;; *) export "$kv"; echo "[miles-run] extra env: $kv" ;; esac; done < "$RUN_DIR/extra_env"
fi
echo "[miles-run] $(date -u +%FT%TZ) host=$(hostname) job=$SLURM_JOB_ID run=$RUN_NAME miles=$MILES sha=$(git -C "$MILES" rev-parse --short HEAD) megatron=$MEGATRON"
# every segment imports the live trees at start: record exactly which code it ran (uncommitted files counted)
echo "[miles-run] code: miles $(git -C "$MILES" rev-parse --short HEAD) dirty=$(git -C "$MILES" status --porcelain 2>/dev/null | wc -l)" \
     "repo $(git -C "$C" rev-parse --short HEAD) dirty=$(git -C "$C" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
python3 -c "import torch, sglang; print('torch', torch.__version__, 'sglang', sglang.__version__)"

# job-private Ray: a 10-port block below the ephemeral range, keyed by the job id
# QTASK (queue worker: several tasks in one job, one after another) gives each task its own port block and Ray dir
P=$((20000 + ((SLURM_JOB_ID + 97 * ${QTASK:-0}) % 1270) * 10))
RAY_TMP=/tmp/ray_$SLURM_JOB_ID${QTASK:+_q$QTASK}
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${GPUS:-8}" --port=$P --dashboard-port=$((P + 1)) \
  --dashboard-agent-listen-port=$((P + 2)) --dashboard-agent-grpc-port=$((P + 3)) --metrics-export-port=$((P + 4)) \
  --ray-client-server-port=$((P + 5)) --runtime-env-agent-port=$((P + 6)) --node-manager-port=$((P + 7)) \
  --object-manager-port=$((P + 8)) --include-dashboard=false --disable-usage-stats --temp-dir="$RAY_TMP"
export RAY_ADDRESS=127.0.0.1:$P
# the job dir is a bind mount point in here (rm of it fails: smoke 15214649 exited 1 after a good run, since a failing
# command in an EXIT trap under set -e sets the exit status): empty it; sbatch_miles.sh removes the dir
trap 'pkill -9 -u "$(id -u)" -f -- "$RAY_TMP" 2>/dev/null || true; echo "[miles-run] TMPDIR at exit: $(du -sh "$JT" 2>/dev/null | cut -f1); /tmp: $(du -sh /tmp 2>/dev/null | cut -f1)"; find "/scratch/gpfs/GROUP/USER/tmp/job/$SLURM_JOB_ID${QTASK:+_q$QTASK}" -mindepth 1 -delete 2>/dev/null || true' EXIT

# a wall-clock kill between the actor save and the critic save leaves the two trackers at different iterations, which
# fails miles' restored-rollout assert in every chained segment: point both at their newest common iteration first
[ -d "$RUN_DIR/ckpt" ] && python3 -m miles_team.ckpt_consistency "$RUN_DIR/ckpt" "$RUN_DIR/ckpt_critic" || true

cd "$MILES"
MODEL_ARGS=$(python3 -c "from miles.utils.external_utils.model_args_utils import shell_safe_model_args; print(shell_safe_model_args('$MILES_MODEL_TYPE'))")
echo "[miles-run] model args: $MODEL_ARGS"
echo "[miles-run] train args: $*"
# per-run overrides for jobs already queued (their args are fixed at submit; this script is read at job start):
# runs/<run>/extra_args, whitespace-separated flags, '#' lines ignored, appended last (argparse: the last flag wins)
EXTRA=()
[ -f "$RUN_DIR/extra_args" ] && read -r -a EXTRA <<< "$(grep -v '^#' "$RUN_DIR/extra_args" | tr '\n' ' ')"
# the last value of a flag over the train args and the extra args (argparse: the last one wins)
ALL=("$@" "${EXTRA[@]}")
argval() { local v="" a prev=""; for a in "${ALL[@]}"; do [ "$prev" = "$1" ] && v=$a; prev=$a; done; echo "$v"; }
# A segment that died in an eval (wall time, CSES idle kill, OOM) leaves a checkpoint K at an eval step with no
# "eval K" line in any log of the run; the next segment resumes after it and would never redo it. Check that here, in
# every segment (audit 2026-10-07: an afternotok check job covered one run only and read only the actor tracker;
# ckpt_consistency above has already put both trackers on their common iteration).
TRK=$RUN_DIR/ckpt/latest_checkpointed_iteration.txt
if [ -f "$TRK" ]; then
  K=$(cat "$TRK"); EVERY=$(argval --eval-interval); NR=$(argval --num-rollout)
  LOGS=$(ls "$B"/logs/miles-"$RUN_NAME"-[0-9]*.out 2>/dev/null | grep -v -- "-$SLURM_JOB_ID.out" || true)
  LOST=0
  # an eval whose replies were saved for judging after training resumed (fcs_rm.log_eval) is not lost: it is judged
  # from runs/<run>/eval_pending/eval_K.pt, without the GPUs
  if [ -n "$EVERY" ] && [ $(( (K + 1) % EVERY )) -eq 0 ] && [ ! -f "$RUN_DIR/eval_pending/eval_$K.pt" ] \
     && { [ -z "$LOGS" ] || ! grep -aq "metrics.py:[0-9]* - eval $K:" $LOGS; }; then
    LOST=1
  fi
  # the run already reached its last rollout: nothing to train, and miles evaluates before training only when a rollout
  # is left (train.py: num_rollout > start_rollout_id), so a lost final eval cannot be redone here: exit, say so
  if [ -n "$NR" ] && [ $((K + 1)) -ge "$NR" ]; then
    [ $LOST -eq 1 ] && echo "[miles-run] WARNING: final eval $K was lost; evaluate checkpoint $K offline"
    if ls "$RUN_DIR"/eval_pending/eval_*.pt >/dev/null 2>&1; then
      echo "[miles-run] judging the evals left pending: $(cd "$RUN_DIR/eval_pending" && ls eval_*.pt | tr '\n' ' ')"
      # the GPUs idle meanwhile (Della cancels after 90 min at 0%): bounded, and a failure only warns
      timeout 80m python3 -m miles_team.fcs_rm --score-pending "$RUN_DIR" || echo "[miles-run] WARNING: judging the pending evals failed or timed out"
    fi
    echo "[miles-run] run complete at checkpoint $K of $NR rollouts: nothing to do"; exit 0
  fi
  if [ $LOST -eq 1 ] && [ ! -f "$RUN_DIR/eval_before_train_once" ]; then
    touch "$RUN_DIR/eval_before_train_once"; echo "[miles-run] eval $K was lost by an earlier segment: redo it before training"
  fi
fi
# runs/<run>/eval_before_train_once: this segment evaluates before training (miles labels it with the resumed checkpoint's
# rollout, so it redoes an eval a cancelled segment lost) despite --skip-eval-before-train; the file is used up
if [ -f "$RUN_DIR/eval_before_train_once" ]; then
  KEEP=(); for a in "$@"; do [ "$a" = "--skip-eval-before-train" ] || KEEP+=("$a"); done; set -- "${KEEP[@]}"
  KEEP=(); for a in "${EXTRA[@]}"; do [ "$a" = "--skip-eval-before-train" ] || KEEP+=("$a"); done; EXTRA=("${KEEP[@]}")
  mv "$RUN_DIR/eval_before_train_once" "$RUN_DIR/eval_before_train_once.used.$SLURM_JOB_ID"
  echo "[miles-run] eval before training (one shot, $RUN_DIR/eval_before_train_once)"
fi
echo "[miles-run] extra args: ${EXTRA[*]:-none}"
eval "python3 train.py $MODEL_ARGS \"\$@\" \"\${EXTRA[@]}\""
