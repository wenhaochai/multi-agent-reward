#!/bin/bash
#SBATCH --job-name=flint-sglang023-parity
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/flint-eval/logs/%x-%j.out
# The port of flint_sglang to SGLang 0.5.23 against Recursive's kit on SGLang 0.5.16, two TP2 servers side by side
# with the eval servers' flags (the kit's serve_h200.sh): D = the port in miles-v0.1.3.sif on GPUs 0-1, A = the kit
# in flint-sglang.sif on GPUs 2-3. Then, on both: (1) logprobs and top-32 heads on S1's 8 long-game sequences
# (2k-60k tokens; D also in a batch of 8 and decode vs prefill), (2) decode throughput at 1 and 8 requests, (3) four
# short real long-game episodes (one Flint, 30k tokens) for tool-call and reasoning parsing. Limits: the whole job is
# killed at 140 min, gpu_watchdog ends it when one of its GPUs stays at 0% for 10 min, and it stops only the process
# groups it started (procs.sh). Usage: parity_job.sh
# Results: /scratch/gpfs/GROUP/USER/tmp/flint_s3/run_<job>/ (table.txt, bench_*.json, gen.txt).
set -uo pipefail
P=/scratch/gpfs/GROUP/USER/project
W=$P/mas-rl-sglang023
MODEL=$P/labs-molt/_workspace/models/flint
KSIF=$P/flint-eval/flint-sglang.sif
MSIF=$P/miles-q38-build/miles-v0.1.3.sif
PYSIF=$P/miles-q38-build/miles.sif   # the eval driver's python (jobs/smoke.sh)
S1=/scratch/gpfs/GROUP/USER/tmp/flint_s3/flint_hf_c0b0cef/flint_hf/parity   # S1's scripts at c0b0cef
SEQS=/scratch/gpfs/GROUP/USER/tmp/flint_parity_seqs.json
A_S1=/scratch/gpfs/GROUP/USER/tmp/flint_parity/run1/sglang_A.pt
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s3/run_$JID
JOBTMP=/scratch/gpfs/GROUP/USER/tmp/job/$JID
source $W/serving/check/procs.sh
source $W/serving/check/gpu_watchdog.sh
export STARTED_FILE=$OUT/started
if [ -z "${CHECK_INNER:-}" ]; then
  mkdir -p "$OUT"
  : > "$STARTED_FILE"
  CHECK_INNER=1 timeout -s KILL 140m bash "$0" "$@"
  rc=$?
  stop $(cat "$STARTED_FILE")
  rm -rf "$JOBTMP"
  echo "parity job exit $rc (137 = the 140 min limit)"
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
echo "job $JID host $(hostname) commit $(git -C $W rev-parse HEAD) GPUs ${G[*]}"
bash $W/serving/build_flint_sglang.sh "$TMPDIR/flint_pkg" || exit 1

declare -A SRV PORT
BASEP=$((20000 + ($(echo "$JID" | tr -dc 0-9 | tail -c 6) % 500) * 20))
serve() {   # $1 name, $2 sif, $3 gpus, $4 port, $5 nccl port, $6 PYTHONPATH
  start apptainer exec --nv --home "$TMPDIR" --bind /scratch/gpfs/GROUP --env "CUDA_VISIBLE_DEVICES=$3" \
    --env "MODEL=$MODEL" --env "PORT=$4" --env "TMPDIR=$TMPDIR" --env "PYTHONPATH=$6" \
    --env "FLINT_TRITON_CACHE_BASE=$TMPDIR/triton_$1" "$2" \
    bash "$MODEL/sglang/serve_h200.sh" --tp-size 2 --mem-fraction-static 0.85 --nccl-port "$5" --host 127.0.0.1 \
    > "$OUT/server_$1.log" 2>&1
  SRV[$1]=${STARTED[-1]} PORT[$1]=$4
}
ready() {   # $1 name
  for _ in $(seq 1 120); do
    curl -sf "127.0.0.1:${PORT[$1]}/health_generate" >/dev/null 2>&1 && { echo "server $1 ready $(date -Is)"; return 0; }
    kill -0 "${SRV[$1]}" 2>/dev/null || break
    sleep 15
  done
  echo "server $1 not ready"; grep -m5 -E "Traceback|Error|error" "$OUT/server_$1.log"; tail -20 "$OUT/server_$1.log"
  return 1
}
run() {   # $1 log, then the command: run it in a group of its own and wait for it
  local log=$1
  shift
  start "$@" > "$log" 2>&1
  wait "${STARTED[-1]}"
}
py() { echo apptainer exec --home "$TMPDIR" --bind /scratch/gpfs/GROUP "$KSIF" python3 "$@"; }

serve D "$MSIF" "${G[0]},${G[1]}" $BASEP $((BASEP + 1)) "$TMPDIR/flint_pkg"
serve A "$KSIF" "${G[2]},${G[3]}" $((BASEP + 2)) $((BASEP + 3)) ""
gpu_watchdog $$ &
ready D || exit 1
ready A || exit 1

# 1. parity: the port with the floor runs (batch of 8, decode vs prefill), the kit alone
run "$OUT/query_D.log" $(py $S1/query_sglang.py "$SEQS" "http://127.0.0.1:${PORT[D]}" "$OUT/sglang_D.pt" --floor) &
QD=$!
run "$OUT/query_A.log" $(py $S1/query_sglang.py "$SEQS" "http://127.0.0.1:${PORT[A]}" "$OUT/sglang_A.pt")
wait $QD
tail -n1 "$OUT/query_D.log" "$OUT/query_A.log"
run "$OUT/table.txt" $(py $W/serving/check/compare_servers.py "$SEQS" "$OUT/sglang_A.pt" "$OUT/sglang_D.pt" "$A_S1")
cat "$OUT/table.txt"

# 2. decode throughput, both servers at once (separate GPU pairs)
run "$OUT/bench_D.log" $(py $W/serving/check/bench.py "$SEQS" "http://127.0.0.1:${PORT[D]}" "$OUT/bench_D.json") &
BD=$!
run "$OUT/bench_A.log" $(py $W/serving/check/bench.py "$SEQS" "http://127.0.0.1:${PORT[A]}" "$OUT/bench_A.json")
wait $BD
echo "BENCH D $(cat $OUT/bench_D.json)"; echo "BENCH A $(cat $OUT/bench_A.json)"

# 3. four real long-game episodes on each server (main's driver, one Flint, 30k tokens, the first four smoke8
# problems); each driver has a TMPDIR of its own, since the game's work directories are named by problem
gen() {   # $1 name
  mkdir -p "$TMPDIR/gen_$1" "$TMPDIR/judge_$1"
  run "$OUT/gen_$1.log" bash -c "cd $W && timeout 60m apptainer exec --home $TMPDIR --bind /scratch/gpfs/GROUP \
    --env TMPDIR=$TMPDIR/gen_$1 --env PYTHONPATH=$W --env MA_LG_N=1 --env MA_LG_TOKENS=30000 \
    --env FCS_RM_CONCURRENCY=4 --env FCS_JUDGE_TMP=$TMPDIR/judge_$1 --env FCS_JUDGE_CACHE=$TMPDIR/judge_cache_$1 \
    $PYSIF python3 run_eval.py --base-url http://127.0.0.1:${PORT[$1]} --model flint \
    --data data/fcs11_val40_agent_smoke8.jsonl --out $OUT/gen_$1 --limit 4 --concurrency 4"
}
gen D &
GD=$!
gen A
wait $GD
run "$OUT/gen.txt" $(py $W/serving/check/gen_summary.py "$OUT/gen_D" "$OUT/gen_A")
cat "$OUT/gen.txt"
grep -c -E "Traceback" "$OUT/server_D.log" | sed 's/^/server D tracebacks: /'
echo PARITY_JOB_DONE
