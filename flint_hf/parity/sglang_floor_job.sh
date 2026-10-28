#!/bin/bash
#SBATCH --job-name=flint-parity-floor
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# SGLang's own noise floor for the parity check: the kit server (flint-sglang.sif, the eval flags) twice more at TP2,
# with chunked prefill of 1024 (B, GPUs 0-1) and 8192 tokens (B2, GPUs 2-3) instead of 4096 (A), so the MoE and
# attention kernels see other token counts. Prefill of every sequence on each. Usage: sglang_floor_job.sh OUTDIR
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-flintrl
MODEL=$P/labs-molt/_workspace/models/flint
FSIF=$P/flint-eval/flint-sglang.sif
OUT=${1:?outdir}
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$TMPDIR" "$OUT"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMPDIR"' EXIT
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:?no GPUs}"
[ "${#G[@]}" -eq 4 ] || { echo "need 4 GPUs, got ${G[*]}"; exit 1; }
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs ${G[*]}"
BASEP=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
declare -A SRV
serve() {   # $1 name, $2 gpus, $3 port, $4 nccl port, $5 chunked prefill size
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$2 APPTAINERENV_MODEL=$MODEL APPTAINERENV_PORT=$3 APPTAINERENV_TMPDIR=$TMPDIR \
  APPTAINERENV_FLINT_TRITON_CACHE_BASE=$P/flint-eval/triton_cache \
    apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
    bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port "$4" --host 127.0.0.1 \
    --chunked-prefill-size "$5" > "$OUT/server_$1.log" 2>&1 &
  SRV[$1]=$!
}
query() {   # $1 name, $2 port
  for _ in $(seq 1 90); do
    curl -sf "127.0.0.1:$2/health_generate" >/dev/null 2>&1 && break
    kill -0 "${SRV[$1]}" 2>/dev/null || break
    sleep 20
  done
  curl -sf "127.0.0.1:$2/health_generate" >/dev/null \
    || { echo "server $1 not ready"; tail -30 "$OUT/server_$1.log"; return 1; }
  apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$FSIF" \
    python3 $W/flint_hf/parity/query_sglang.py "$SEQS" "http://127.0.0.1:$2" "$OUT/sglang_$1.pt.tmp" \
    > "$OUT/query_$1.log" 2>&1 && mv "$OUT/sglang_$1.pt.tmp" "$OUT/sglang_$1.pt"
  tail -1 "$OUT/query_$1.log"
}
serve B "${G[0]},${G[1]}" $BASEP $((BASEP + 1)) 1024
serve B2 "${G[2]},${G[3]}" $((BASEP + 2)) $((BASEP + 3)) 8192
query B2 $((BASEP + 2)) &
QB2=$!
query B $BASEP
wait $QB2
kill "${SRV[@]}" 2>/dev/null
[ -f "$OUT/sglang_B.pt" ] && [ -f "$OUT/sglang_B2.pt" ] && echo PARITY_FLOOR_DONE || { echo PARITY_FLOOR_FAIL; exit 1; }
