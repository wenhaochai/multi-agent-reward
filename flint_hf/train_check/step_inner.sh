#!/bin/bash
# Runs inside miles-v0.1.3.sif (step_job.sh): a job-private Ray head on job-derived ports, then `python3 train.py`.
set -uo pipefail
: "${JID:?}"
DATA=${1:?} OUT=${2:?}
shift 2
P=/scratch/gpfs/GROUP/USER/project
W=$P/miles-flint
F=$P/mas-rl-flintrl
MODEL=$P/labs-molt/_workspace/models/flint
export HOME=$TMPDIR/home TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
mkdir -p "$HOME"
export PYTHONPATH=/root/Megatron-LM:$W:$F PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_USAGE_STATS_ENABLED=0 no_proxy=127.0.0.1,localhost MASTER_ADDR=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ray starts workers single-threaded; the CPU-offloaded Adam step needs the job's cores (96 / 8 ranks)
export OMP_NUM_THREADS=11 MKL_NUM_THREADS=11
echo "[s2] miles $(git -C "$W" rev-parse --short HEAD) dirty=$(git -C "$W" status --porcelain | wc -l); flint_hf $(git -C "$F" rev-parse --short HEAD) dirty=$(git -C "$F" status --porcelain | wc -l)"
NG=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JNUM=${JID//[!0-9]/}
PORT=$((20000 + (JNUM % 1270) * 10))
RAY_TMP=/tmp/ray_$JID
ray start --head --node-ip-address 127.0.0.1 --num-gpus "$NG" --port=$PORT --dashboard-port=$((PORT + 1)) \
  --dashboard-agent-listen-port=$((PORT + 2)) --dashboard-agent-grpc-port=$((PORT + 3)) --metrics-export-port=$((PORT + 4)) \
  --ray-client-server-port=$((PORT + 5)) --runtime-env-agent-port=$((PORT + 6)) --node-manager-port=$((PORT + 7)) \
  --object-manager-port=$((PORT + 8)) --include-dashboard=false --disable-usage-stats --temp-dir="$RAY_TMP"
export RAY_ADDRESS=127.0.0.1:$PORT
trap 'ray stop --force >/dev/null 2>&1; pkill -9 -u "$(id -u)" -f -- "$RAY_TMP" 2>/dev/null || true' EXIT
cd "$W"
# miles builds a prompt data source even when it replays --load-debug-rollout-data; one unused prompt satisfies it
[ -f "$OUT/unused_prompt.jsonl" ] || echo '{"prompt": "unused", "label": ""}' > "$OUT/unused_prompt.jsonl"
python3 "$W/train.py" \
  --train-backend fsdp --hf-checkpoint "$MODEL" --attn-implementation eager \
  --prompt-data "$OUT/unused_prompt.jsonl" --input-key prompt --label-key label \
  --load-debug-rollout-data "$DATA/rollout_{rollout_id}.pt" --save-debug-train-data "$OUT/train_{rollout_id}_{rank}.pt" \
  --num-rollout 2 --rollout-batch-size 1 --n-samples-per-prompt 8 --global-batch-size 8 \
  --rollout-temperature 1 --rollout-top-logprobs-num 32 \
  --advantage-estimator grpo --disable-grpo-std-normalization --kl-coef 0 --entropy-coef 0 \
  --loss-type score_centering --score-centering-is none \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98 \
  --gradient-checkpointing --use-dynamic-batch-size --max-tokens-per-gpu 66560 \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NG" --seed 1 --fsdp-cpu-offload "$@" \
  > "$OUT/train.log" 2>&1
r=$?
grep -E -i "Traceback|Error|memory|timer|loss|grad_norm|train_rollout|step" "$OUT/train.log" | grep -v -i warning | tail -60
exit $r
