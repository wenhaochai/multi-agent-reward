#!/bin/bash
# Inside miles-v0.1.3.sif (rl/engine_job.sh): one external Flint SGLang engine over the job's GPUs, with run20's
# serving flags. The argv comes from miles' own _compute_server_args on the trainer's arguments (rl/ext_args.sh,
# rl/engine_argv.py), so the engine serves exactly what the trainer's miles would have launched, minus colocation.
# Env: IP (the node's routable address), PORT, BASE (40 free ports from BASE), PKG (the built Flint plugin), TMPDIR,
# REPO, MILES, FLINT. Usage: engine_inner.sh serve | argv (print the launch line and exit).
set -uo pipefail
: "${IP:?}" "${PORT:?}" "${BASE:?}" "${PKG:?}" "${TMPDIR:?}" "${REPO:?}" "${MILES:?}" "${FLINT:?}"
export HOME=$TMPDIR/home TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export SGLANG_CACHE_DIR=$TMPDIR/sglang FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer
mkdir -p "$HOME"
export PYTHONPATH=$PKG:$REPO:$MILES PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export no_proxy=127.0.0.1,localhost,$IP
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1 SGLANG_RETURN_ORIGINAL_LOGPROB=0
# the weight push from a trainer in another job: NCCL between separate apptainer instances fails with cuMem on
# (w545: "unhandled cuda error"; NCCL_CUMEM_ENABLE=0 on both sides passed); bootstrap over eno8303, IB found by NCCL
export NCCL_CUMEM_ENABLE=0 NCCL_SOCKET_IFNAME=eno8303
# run20's engines were launched colocated, where miles turns the custom all-reduce v2 off; keep that numerics here
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
NG=8 ENGINE_TP=8 ENGINE_NGPUS=8 ENGINE_MEM=0.85
source "$REPO/rl/ext_args.sh"
out=$(python3 "$REPO/rl/engine_argv.py" "$IP" "$PORT" "$BASE" -- "${TRAIN[@]}" "${SERVE[@]}" \
  --rollout-external-engine-addrs "$IP:$PORT" --update-weight-transfer-mode broadcast --num-rollout 1) || { echo "[engine] argv failed"; exit 1; }
env_line=$(grep '^ENV ' <<< "$out" | cut -d' ' -f2-)
argv_line=$(grep '^ARGV ' <<< "$out" | cut -d' ' -f2-)
[ -n "$argv_line" ] || { echo "[engine] no argv: $out"; exit 1; }
# miles' engine environment (SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false, SGLANG_TIMEOUT_KEEP_ALIVE=60, ...); a
# value set above wins, as in miles (os.environ.get(key, default))
for kv in $env_line; do k=${kv%%=*}; [ -n "${!k+x}" ] || export "$kv"; done
echo "[engine] env: $env_line"
echo "[engine] $(date -u +%FT%TZ) launch: $argv_line"
[ "${1:-serve}" = argv ] && exit 0
eval "exec $argv_line"
