#!/bin/bash
# Runs INSIDE miles.sif (radixark/miles:latest, built 2026-09-30; its /root/miles is 874b3e275 and MILES_SRC is checked out there) (sbatch_miles.sh). miles' own launcher (command_utils) starts with `pkill -9 sglang; ray stop
# --force; pkill -9 ray`, which on a shared Della node kills other jobs' processes, so this script does its job by hand:
# a job-private Ray head on job-derived ports (Della memory reference_slime_della / reference_della_ray_shared_node),
# every env var the Ray workers need exported BEFORE `ray start` (workers inherit the raylet's env), then
# `python3 train.py <model args> <train args>` as the driver. Args: MILES_MODEL_TYPE (scripts/models/<type>.py) and
# the train args ("$@").
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
MILES=${MILES_SRC:-/scratch/gpfs/GROUP/USER/project/miles-q38}
: "${RUN_NAME:?}" "${SLURM_JOB_ID:?}" "${MILES_MODEL_TYPE:?}"
RUN_DIR=$B/runs/$RUN_NAME
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
export HOME=/tmp/home TRITON_CACHE_DIR=/tmp/home/.cache/triton
mkdir -p "$HOME"
MEGATRON=/root/Megatron-LM   # the image's patched Megatron (a namespace package: megatron.__file__ is None)
# pyfix/ first: the image (radixark/miles:latest, 2026-09-30) ships opentelemetry-api 1.45.0 with sdk/exporters 1.44.0, so
# Ray's dashboard agent dies at import ('_ExtendedAttributes') and `ray start` times out (smoke 14791336). pyfix holds
# opentelemetry-api 1.44.0 (tools: pip download + pip install --target, see README), matching the rest.
export PYTHONPATH=$B/pyfix:$MEGATRON:$MILES:$B${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 MASTER_ADDR=127.0.0.1 no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_USAGE_STATS_ENABLED=0
export NCCL_NVLS_ENABLE=$(nvidia-smi topo -m 2>/dev/null | grep -c 'NV[0-9]' | awk '{print ($1 > 0) ? 1 : 0}')
echo "[miles-run] $(date -u +%FT%TZ) host=$(hostname) job=$SLURM_JOB_ID run=$RUN_NAME miles=$MILES sha=$(git -C "$MILES" rev-parse --short HEAD) megatron=$MEGATRON"
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
trap 'pkill -9 -u "$(id -u)" -f -- "$RAY_TMP" 2>/dev/null || true' EXIT

cd "$MILES"
MODEL_ARGS=$(python3 -c "from miles.utils.external_utils.model_args_utils import shell_safe_model_args; print(shell_safe_model_args('$MILES_MODEL_TYPE'))")
echo "[miles-run] model args: $MODEL_ARGS"
echo "[miles-run] train args: $*"
# per-run overrides for jobs already queued (their args are fixed at submit; this script is read at job start):
# runs/<run>/extra_args, whitespace-separated flags, '#' lines ignored, appended last (argparse: the last flag wins)
EXTRA=()
[ -f "$RUN_DIR/extra_args" ] && read -r -a EXTRA <<< "$(grep -v '^#' "$RUN_DIR/extra_args" | tr '\n' ' ')"
echo "[miles-run] extra args: ${EXTRA[*]:-none}"
eval "python3 train.py $MODEL_ARGS \"\$@\" \"\${EXTRA[@]}\""
