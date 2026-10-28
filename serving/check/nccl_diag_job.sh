#!/bin/bash
#SBATCH --job-name=flint-nccl-diag
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# Why do the ported server's ranks fail to set up NCCL with miles' trainer rank (w541: "Cuda failure 'invalid
# argument'")? Three trials, each a fresh port at TP2 on GPUs 0-1 (no CUDA graphs, to start fast) and nccl_diag.py
# on GPU 2 sending one small tensor through miles' update path, NCCL_DEBUG=INFO on every rank:
#   two     server and trainer in separate apptainer exec instances (as w541), default NCCL settings;
#   one     server and trainer in ONE apptainer exec (as miles runs them), default NCCL settings;
#   cumem0  separate instances with NCCL_CUMEM_ENABLE=0 on both sides.
# The whole job is killed at 600 s; each trainer at 90 s; afterwards anything of mine left on this job's GPUs is
# killed. Usage: nccl_diag_job.sh. Results: /scratch/gpfs/GROUP/USER/tmp/flint_s3/nccl_<job>/
set -uo pipefail
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-sglang023
MODEL=$P/labs-molt/_workspace/models/flint
MSIF=$P/miles-q38-build/miles-v0.1.3.sif
MILES=$P/miles-flint
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s3/nccl_$JID
JOBTMP=/scratch/gpfs/GROUP/USER/tmp/job/$JID
ME=$(id -un)
reap() {   # kill whatever of mine still holds one of this job's GPUs
  local pid
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null); do
    [ "$(stat -c %U /proc/$pid 2>/dev/null)" = "$ME" ] && kill -KILL "$pid" 2>/dev/null
  done
}
if [ -z "${DIAG_INNER:-}" ]; then
  mkdir -p "$OUT"
  DIAG_INNER=1 timeout -s KILL 600 bash "$0" "$@"
  rc=$?
  reap; sleep 3; reap
  rm -rf "$JOBTMP"
  echo "nccl diag exit $rc (137 = the 600 s limit)"
  exit $rc
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export TMPDIR=$JOBTMP
mkdir -p "$TMPDIR"
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 4 ] || { echo "need 4 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs ${G[*]}"
nvidia-smi -L | sed -n "$((G[0] + 1))p;$((G[1] + 1))p;$((G[2] + 1))p"
nvidia-smi topo -m | head -12
bash $W/serving/build_flint_sglang.sh "$TMPDIR/flint_pkg" || exit 1
PORT=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
SERVE="bash $MODEL/sglang/serve_h200.sh --tp-size 2 --mem-fraction-static 0.85 --nccl-port $((PORT + 1)) --host 127.0.0.1 --disable-cuda-graph"
TRAIN="timeout -s KILL 90 python3 -u $W/serving/check/nccl_diag.py $SEQS http://127.0.0.1:$PORT"
COMMON=(--nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP --env "MODEL=$MODEL" --env "PORT=$PORT" --env "TMPDIR=$TMPDIR"
        --env "FLINT_TRITON_CACHE_BASE=$TMPDIR/triton")
ready() {   # $1 server pid
  for _ in $(seq 1 40); do
    curl -sf "127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
trial() {   # $1 name, $2 mode (two|one), then NCCL settings as K=V
  local name=$1 mode=$2 srv kv
  shift 2
  local envs=(--env NCCL_DEBUG=INFO)
  for kv in "$@"; do envs+=(--env "$kv"); done
  echo "== $name ($mode instances; $*) $(date -Is)"
  if [ "$mode" = two ]; then
    apptainer exec "${COMMON[@]}" "${envs[@]}" --env "CUDA_VISIBLE_DEVICES=${G[0]},${G[1]}" \
      --env "PYTHONPATH=$TMPDIR/flint_pkg" "$MSIF" $SERVE > "$OUT/server_$name.log" 2>&1 &
    srv=$!
    if ready $srv; then
      apptainer exec "${COMMON[@]}" "${envs[@]}" --env "CUDA_VISIBLE_DEVICES=${G[2]}" --env "PYTHONPATH=$MILES" \
        "$MSIF" $TRAIN > "$OUT/trainer_$name.log" 2>&1
    else
      echo "server not ready"; tail -5 "$OUT/server_$name.log"
    fi
  else
    apptainer exec "${COMMON[@]}" "${envs[@]}" "$MSIF" bash -c "
      CUDA_VISIBLE_DEVICES=${G[0]},${G[1]} PYTHONPATH=$TMPDIR/flint_pkg $SERVE > $OUT/server_$name.log 2>&1 &
      S=\$!
      for _ in \$(seq 1 40); do
        python3 -c 'import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:$PORT/health_generate\", timeout=3)' \
          2>/dev/null && break
        sleep 5
      done
      CUDA_VISIBLE_DEVICES=${G[2]} PYTHONPATH=$MILES $TRAIN > $OUT/trainer_$name.log 2>&1
      kill \$S" &
    srv=$!
    wait $srv
  fi
  grep -h -E "NCCL_DIAG|trainer CUDA|group connected" "$OUT/trainer_$name.log"
  grep -h -m6 -E "Failed to receive|WARN|failure|Cuda failure" "$OUT/server_$name.log" "$OUT/trainer_$name.log"
  kill $srv 2>/dev/null
  sleep 5
  reap
  sleep 2
}
trial two two
trial one one
trial cumem0 two NCCL_CUMEM_ENABLE=0
echo NCCL_DIAG_JOB_DONE
