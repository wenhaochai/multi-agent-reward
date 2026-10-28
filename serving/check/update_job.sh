#!/bin/bash
#SBATCH --job-name=flint-sglang023-update
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# miles' weight update into the ported Flint server (update_check.py): the port serves at TP2 on GPUs 0-1
# (miles-v0.1.3.sif, the eval servers' flags), and the trainer side, flint_hf on GPUs 2-3, sends HF-named tensors in
# miles' session frame with miles' non-colocated transfer code (the fork's branch flint-grpo). Limits: the whole job is
# killed at 75 min; the check ends itself on any refused server call or after 40 min; gpu_watchdog ends the job when
# one of its GPUs stays at 0% for 10 min. The job stops only the process groups it started (procs.sh).
# Usage: update_job.sh. Results: /scratch/gpfs/GROUP/USER/tmp/flint_s3/update_<job>/update.json
set -uo pipefail
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-sglang023
MODEL=$P/labs-molt/_workspace/models/flint
MSIF=$P/miles-q38-build/miles-v0.1.3.sif
MILES=$P/miles-flint                                                   # the fork, branch flint-grpo
HF=/scratch/gpfs/GROUP/USER/tmp/flint_s3/flint_hf_c0b0cef         # S1's flint_hf at c0b0cef
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s3/update_$JID
JOBTMP=/scratch/gpfs/GROUP/USER/tmp/job/$JID
source $W/serving/check/procs.sh
source $W/serving/check/gpu_watchdog.sh
export STARTED_FILE=$OUT/started
if [ -z "${CHECK_INNER:-}" ]; then
  mkdir -p "$OUT"
  : > "$STARTED_FILE"
  CHECK_INNER=1 timeout -s KILL 75m bash "$0" "$@"
  rc=$?
  stop $(cat "$STARTED_FILE")
  rm -rf "$JOBTMP"
  echo "update job exit $rc (137 = the 75 min limit)"
  exit $rc
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export TMPDIR=$JOBTMP
mkdir -p "$TMPDIR"
trap 'stop "${STARTED[@]}"; kill $(jobs -p) 2>/dev/null' EXIT
trap 'exit 1' TERM
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 4 ] || { echo "need 4 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) miles $(git -C $MILES rev-parse HEAD) GPUs ${G[*]}"
bash $W/serving/build_flint_sglang.sh "$TMPDIR/flint_pkg" || exit 1
PORT=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
start apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP --env "CUDA_VISIBLE_DEVICES=${G[0]},${G[1]}" \
  --env "MODEL=$MODEL" --env "PORT=$PORT" --env "TMPDIR=$TMPDIR" --env "PYTHONPATH=$TMPDIR/flint_pkg" \
  --env "FLINT_TRITON_CACHE_BASE=$TMPDIR/triton" "$MSIF" \
  bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port $((PORT + 1)) --host 127.0.0.1 \
  > "$OUT/server.log" 2>&1
SRV=${STARTED[0]}
gpu_watchdog $$ &
for _ in $(seq 1 80); do
  curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || break
  sleep 15
done
curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null || { echo "server not ready"; tail -30 "$OUT/server.log"; exit 1; }
echo "server ready $(date -Is)"
# NCCL_CUMEM_ENABLE=0 on the trainer, as miles/ray/specs/train.py sets it for its actors: SGLang sets 0 in its own
# processes, and a communicator whose ranks disagree fails to set up (nccl_diag_job.sh w545: "invalid argument")
start apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP --env "CUDA_VISIBLE_DEVICES=${G[2]},${G[3]}" \
  --env "PYTHONPATH=$HF:$MILES" --env "TMPDIR=$TMPDIR" --env NCCL_CUMEM_ENABLE=0 "$MSIF" \
  timeout -s KILL 45m python3 -u $W/serving/check/update_check.py "$SEQS" "http://127.0.0.1:$PORT" "$OUT/update.json" \
  > "$OUT/update.log" 2>&1
wait "${STARTED[1]}"
grep -v "model of type" "$OUT/update.log" | tail -40
grep -E "Failed to update|Failed to receive|Traceback|RuntimeError" "$OUT/server.log" | head -5
[ -f "$OUT/update.json" ] && echo UPDATE_JOB_DONE || { echo UPDATE_JOB_FAIL; exit 1; }
