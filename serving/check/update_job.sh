#!/bin/bash
#SBATCH --job-name=flint-sglang023-update
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# miles' weight update into the ported Flint server (update_check.py): the port serves at TP2 on GPUs 0-1
# (miles-v0.1.3.sif, the eval servers' flags), and the trainer side, flint_hf on GPUs 2-3, sends HF-named tensors in
# miles' session frame with miles' non-colocated transfer code (the fork's branch flint-grpo). Guards: the check ends
# itself on any refused server call or after 40 min, timeout kills it at 45 min, and gpu_watchdog stops the job when
# one of its GPUs stays at 0% for 10 min. Usage: update_job.sh
# Results: /scratch/gpfs/GROUP/USER/tmp/flint_s3/update_<job>/update.json
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-sglang023
MODEL=$P/labs-molt/_workspace/models/flint
MSIF=$P/miles-q38-build/miles-v0.1.3.sif
MILES=$P/miles-flint                                                   # the fork, branch flint-grpo
HF=/scratch/gpfs/GROUP/USER/tmp/flint_s3/flint_hf_c0b0cef         # S1's flint_hf at c0b0cef
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s3/update_$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR" "$OUT"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMPDIR"' EXIT
trap 'exit 1' TERM
source $W/serving/check/gpu_watchdog.sh
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 4 ] || { echo "need 4 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) miles $(git -C $MILES rev-parse HEAD) GPUs ${G[*]}"
bash $W/serving/build_flint_sglang.sh "$TMPDIR/flint_pkg" || exit 1
PORT=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
APPTAINERENV_CUDA_VISIBLE_DEVICES=${G[0]},${G[1]} APPTAINERENV_MODEL=$MODEL APPTAINERENV_PORT=$PORT \
APPTAINERENV_TMPDIR=$TMPDIR APPTAINERENV_PYTHONPATH=$TMPDIR/flint_pkg APPTAINERENV_FLINT_TRITON_CACHE_BASE=$TMPDIR/triton \
  apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$MSIF" \
  bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port $((PORT + 1)) --host 127.0.0.1 \
  > "$OUT/server.log" 2>&1 &
SRV=$!
gpu_watchdog $$ $SRV &
for _ in $(seq 1 80); do
  curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || break
  sleep 15
done
curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null || { echo "server not ready"; tail -30 "$OUT/server.log"; exit 1; }
echo "server ready $(date -Is)"
APPTAINERENV_CUDA_VISIBLE_DEVICES=${G[2]},${G[3]} APPTAINERENV_PYTHONPATH=$HF:$MILES APPTAINERENV_TMPDIR=$TMPDIR \
  timeout -s KILL 45m apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$MSIF" \
  python3 -u $W/serving/check/update_check.py "$SEQS" "http://127.0.0.1:$PORT" "$OUT/update.json" \
  > "$OUT/update.log" 2>&1 &
wait $!
grep -v "model of type" "$OUT/update.log" | tail -40
grep -E "Failed to update|Traceback|RuntimeError" "$OUT/server.log" | head -5
[ -f "$OUT/update.json" ] && echo UPDATE_JOB_DONE || { echo UPDATE_JOB_FAIL; exit 1; }
