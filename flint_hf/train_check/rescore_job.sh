#!/bin/bash
#SBATCH --job-name=flint-s2-rescore
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# S2: SGLang's next-token log-probs and top-32 heads for the 8 packing-problem sequences (rescore.py), from the
# kit's SGLang 0.5.16 (flint-sglang.sif, the eval servers' flags), two TP2 servers side by side.
# Usage: rescore_job.sh SEQS_JSON OUT_DIR
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
MODEL=$P/labs-molt/_workspace/models/flint
FSIF=$P/flint-eval/flint-sglang.sif
SEQS=${1:?seqs} OUT=${2:?out dir}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR" "$OUT"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMPDIR"' EXIT
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 4 ] || { echo "need 4 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs ${G[*]}"
BASEP=$((21000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 400) * 20))
PIDS=()
for s in 0 1; do
  APPTAINERENV_CUDA_VISIBLE_DEVICES=${G[$((2 * s))]},${G[$((2 * s + 1))]} APPTAINERENV_MODEL=$MODEL \
  APPTAINERENV_PORT=$((BASEP + 2 * s)) APPTAINERENV_TMPDIR=$TMPDIR APPTAINERENV_FLINT_TRITON_CACHE_BASE=$P/flint-eval/triton_cache \
    apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
    bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port $((BASEP + 2 * s + 1)) \
    --host 127.0.0.1 > "$OUT/server_$s.log" 2>&1 &
  PIDS+=($!)
done
ok=0
for s in 0 1; do
  for _ in $(seq 1 90); do
    curl -sf "127.0.0.1:$((BASEP + 2 * s))/health_generate" >/dev/null 2>&1 && break
    kill -0 "${PIDS[$s]}" 2>/dev/null || break
    sleep 20
  done
  curl -sf "127.0.0.1:$((BASEP + 2 * s))/health_generate" >/dev/null && ok=$((ok + 1)) || tail -20 "$OUT/server_$s.log"
done
[ "$ok" -eq 2 ] || { echo "RESCORE_FAIL: servers not ready"; exit 1; }
Q=()
apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" python3 $W/flint_hf/train_check/rescore.py \
  "$SEQS" "http://127.0.0.1:$BASEP" "$OUT" 0 2 4 6 > "$OUT/query_0.log" 2>&1 & Q+=($!)
apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" python3 $W/flint_hf/train_check/rescore.py \
  "$SEQS" "http://127.0.0.1:$((BASEP + 2))" "$OUT" 1 3 5 7 > "$OUT/query_1.log" 2>&1 & Q+=($!)
rc=0
for q in "${Q[@]}"; do wait "$q" || rc=1; done
kill "${PIDS[@]}" 2>/dev/null
tail -4 "$OUT"/query_*.log
n=$(ls "$OUT"/seq*.pt 2>/dev/null | wc -l)
[ "$rc" -eq 0 ] && [ "$n" -eq 8 ] && echo "RESCORE_DONE 8 sequences" || { echo "RESCORE_FAIL rc $rc files $n"; exit 1; }
