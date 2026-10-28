#!/bin/bash
#SBATCH --job-name=flint-hf-parity
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# HF Flint (flint_hf) against SGLang, all eight GPUs at once: HF in a manual pipeline on GPUs 0-3; SGLang 0.5.16 with
# the Flint kit (flint-sglang.sif, the eval servers' flags) at TP2 on GPUs 4-5 (A, the reference); SGLang 0.5.21 of
# miles.sif with the same wheel on PYTHONPATH at TP2 on GPUs 6-7 (C: can miles' own SGLang serve Flint?). Sequences:
# flint_hf/parity/build_seqs.py output. Results and the table in tmp/flint_parity/<job>.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
MODEL=$P/labs-molt/_workspace/models/flint
FSIF=$P/flint-eval/flint-sglang.sif
MSIF=$P/miles-q38-build/miles.sif
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_parity/$JID
mkdir -p "$TMPDIR" "$OUT"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMPDIR"' EXIT
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 8 ] || { echo "need 8 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs ${G[*]}"
BASEP=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
serve() {   # $1 name, $2 sif, $3 gpus, $4 port, $5 nccl port, $6 extra PYTHONPATH, $7 triton cache
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$3 APPTAINERENV_MODEL=$MODEL APPTAINERENV_PORT=$4 APPTAINERENV_TMPDIR=$TMPDIR \
  APPTAINERENV_PYTHONPATH=$6 APPTAINERENV_FLINT_TRITON_CACHE_BASE=$7 \
    apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$2" \
    bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port "$5" --host 127.0.0.1 \
    > "$OUT/server_$1.log" 2>&1 &
}
serve A "$FSIF" "${G[4]},${G[5]}" $BASEP $((BASEP + 1)) "" $P/flint-eval/triton_cache
serve C "$MSIF" "${G[6]},${G[7]}" $((BASEP + 2)) $((BASEP + 3)) /scratch/gpfs/GROUP/USER/tmp/flint_sglang_miles \
  $TMPDIR/triton_C
APPTAINERENV_CUDA_VISIBLE_DEVICES=${G[0]},${G[1]},${G[2]},${G[3]} APPTAINERENV_PYTHONPATH=$W APPTAINERENV_TMPDIR=$TMPDIR \
  apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
  python3 $W/flint_hf/parity/hf_score.py "$SEQS" "$OUT/sglang_A.pt" "$OUT/hf.pt" > "$OUT/hf.log" 2>&1 &
HF=$!
query() {   # $1 name, $2 port, $3 extra flag
  for _ in $(seq 1 90); do curl -sf "127.0.0.1:$2/health_generate" >/dev/null 2>&1 && break; sleep 20; done
  curl -sf "127.0.0.1:$2/health_generate" >/dev/null || { echo "server $1 not ready"; tail -40 "$OUT/server_$1.log"; return 1; }
  echo "server $1 ready $(date -Is)"
  apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
    python3 $W/flint_hf/parity/query_sglang.py "$SEQS" "http://127.0.0.1:$2" "$OUT/sglang_$1.pt.tmp" $3 > "$OUT/query_$1.log" 2>&1 \
    && mv "$OUT/sglang_$1.pt.tmp" "$OUT/sglang_$1.pt"
  tail -2 "$OUT/query_$1.log"
}
query A $BASEP --floor &
QA=$!
query C $((BASEP + 2)) ""
grep -m3 -E "Traceback|Error|error" "$OUT/server_C.log" | head -3
wait $QA
[ -f "$OUT/sglang_A.pt" ] || { echo "PARITY_FAIL: no results from server A"; kill $HF; exit 1; }
pkill -f "sglang.launch_server.*--port $BASEP" ; pkill -f "sglang.launch_server.*--port $((BASEP + 2))"
wait $HF
tail -4 "$OUT/hf.log"
if [ -f "$OUT/sglang_C.pt" ]; then CP=$OUT/sglang_C.pt; else CP=$OUT/sglang_A.pt; echo "no C results: the C row repeats A"; fi
apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
  python3 $W/flint_hf/parity/compare.py "$SEQS" "$OUT/sglang_A.pt" "$CP" "$OUT/hf.pt" | tee "$OUT/table.txt"
