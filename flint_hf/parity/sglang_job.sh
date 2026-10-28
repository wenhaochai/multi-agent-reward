#!/bin/bash
#SBATCH --job-name=flint-parity-sglang
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# SGLang half of the HF-vs-SGLang parity check, two TP2 servers side by side: A = SGLang 0.5.16 with the Flint kit
# (flint-sglang.sif, the eval servers' flags), the reference, on GPUs 0-1; C = the SGLang of a miles image ($2) with
# the same wheel on PYTHONPATH, on GPUs 2-3 (can miles' own SGLang serve Flint?). Usage: sglang_job.sh OUTDIR MILES_SIF
# Writes OUTDIR/sglang_A.pt and sglang_C.pt (query_sglang.py); hf_job.sh reads them.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
MODEL=$P/labs-molt/_workspace/models/flint
FSIF=$P/flint-eval/flint-sglang.sif
OUT=${1:?outdir} MSIF=${2:?miles sif}
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR" "$OUT"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMPDIR"' EXIT
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 4 ] || { echo "need 4 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs ${G[*]} miles sif $MSIF"
declare -A SRV
BASEP=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
serve() {   # $1 name, $2 sif, $3 gpus, $4 port, $5 nccl port, $6 extra PYTHONPATH, $7 triton cache
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$3 APPTAINERENV_MODEL=$MODEL APPTAINERENV_PORT=$4 APPTAINERENV_TMPDIR=$TMPDIR \
  APPTAINERENV_PYTHONPATH=$6 APPTAINERENV_FLINT_TRITON_CACHE_BASE=$7 \
    apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$2" \
    bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port "$5" --host 127.0.0.1 \
    > "$OUT/server_$1.log" 2>&1 &
  SRV[$1]=$!
}
query() {   # $1 name, $2 port, $3 extra flag
  for _ in $(seq 1 90); do
    curl -sf "127.0.0.1:$2/health_generate" >/dev/null 2>&1 && break
    kill -0 "${SRV[$1]}" 2>/dev/null || break   # the server exited
    sleep 20
  done
  curl -sf "127.0.0.1:$2/health_generate" >/dev/null \
    || { echo "server $1 not ready"; tail -30 "$OUT/server_$1.log"; return 1; }
  echo "server $1 ready $(date -Is)"
  apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
    python3 $W/flint_hf/parity/query_sglang.py "$SEQS" "http://127.0.0.1:$2" "$OUT/sglang_$1.pt.tmp" $3 \
    > "$OUT/query_$1.log" 2>&1 && mv "$OUT/sglang_$1.pt.tmp" "$OUT/sglang_$1.pt"
  tail -2 "$OUT/query_$1.log"
}
serve A "$FSIF" "${G[0]},${G[1]}" $BASEP $((BASEP + 1)) "" $P/flint-eval/triton_cache
serve C "$MSIF" "${G[2]},${G[3]}" $((BASEP + 2)) $((BASEP + 3)) /scratch/gpfs/GROUP/USER/tmp/flint_sglang_miles \
  $TMPDIR/triton_C
query C $((BASEP + 2)) "" &
QC=$!
query A $BASEP --floor
wait $QC
kill "${SRV[@]}" 2>/dev/null
grep -m3 -E "Traceback|Error" "$OUT/server_C.log"
[ -f "$OUT/sglang_A.pt" ] || { echo "PARITY_SGLANG_FAIL: no results from A"; exit 1; }
[ -f "$OUT/sglang_C.pt" ] && echo "C_SERVES: miles SGLang served Flint" || echo "C_FAILS: see $OUT/server_C.log"
echo PARITY_SGLANG_DONE
