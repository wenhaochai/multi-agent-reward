#!/bin/bash
#SBATCH --job-name=flint-rl
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=880G
#SBATCH --time=03:00:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/tmp/flint_s5/logs/%x-%j.out
#FETCHABLE
# RL of Flint on the long game through miles v0.1.3 (our fork), on one 8-GPU block: GRPO without std normalization,
# score centering over the sampler's top-32 candidates, routing replay (R3), one optimizer step per rollout (problem 0,
# 8 episodes of 128k tokens), the rollout engines (4 x TP2) and the FSDP actor (CPU-offloaded parameters and Adam)
# colocated. Usage: rl_job.sh smoke|test2|run20 (wq --prio 10; smoke and run20 with their own --time). smoke: the
# same path in ~50 min (2 steps of 8 episodes x 4k tokens, no actor save); run it before test2. The game runs with RL's context cap
# (rl/rl_inner.sh: 110k characters, 16k-token turns) so that a training sample stays near 64k tokens.
#   test2     2 colocated RL steps, the actor saved at the end
#   run20     20 colocated RL steps, the actor saved at the end
# train67 modes (Frontier-CS v1.1's 67 training problems, data/fcs11_train67_agent.jsonl, pre-shuffled with a fixed seed
# by rl/train67_data.py): P problems per step x G episodes each, one GRPO group per problem, one TP8 engine; usage
# rl_job.sh train67[_test|_smoke] [P G [STEPS]] [set=NAME] [seed=N] [levers=LIST] (defaults 4 8, train67 40 steps;
# set=NAME trains on the problems of data/train67_sets/NAME.txt only, default all 67, the run dirs then hold _NAME;
# seed=N, default 67: a replicate's data order and RL seeds (T67_SEED below), the run dirs then hold _s<N> after the set
# name; levers=LIST, default none: engine levers, a comma list of tp4 gate graphs arfusion pgraph moecfg (rl/rl_inner.sh), the
# run dirs then hold _k<levers, sorted, joined by ->; cmdtimeout=N, default 60: the game's per-command time limit in
# seconds (MA_AG_TIMEOUT, flint/sandbox.py; a rule of the game, owner 2026-11-01: A/B before any adoption), the run dirs
# then hold _t<N>; elastic=1: anchored elastic RL (tmp/elastic_rl/anchored.md): the run keeps its colocated engines
# and adds every engine job (rl/elastic_engine.sh RUN, any number, joining and leaving at any time) as an extra cell
# from its next weight push on; the trainer writes RUN/trainer.json every 30 s and RUN/engine_spec.json (what the
# engine jobs serve), and RUN/STOP when the run ends for good; the run dirs then hold _e; async=1: fully async RL on a
# split node (rl/rl_inner.sh: the actor on 4 GPUs, the run's own 1 x TP4 engine on the other 4, train_async.py
# --fully-async, TIS), stale=N its staleness bound (default none); on a fresh start a warm-up engine
# (rl/elastic_engine.sh EL_WARMUP=1, its files in RUN/warmup) serves on the actor's 4 GPUs until the first step's data
# is in (flint.elastic_rollout); the run dirs then hold _a, _a<N>):
#   train67_smoke  B = P x G episodes of 4k tokens, 2 steps, no save (~1 h)
#   train67_test   B episodes of 128k, 2 steps, a full checkpoint after each (host memory at its peak: the end of the
#                  optimizer step and the save; step time), resumable in runs/train67_test[_NAME]_p<P>g<G>, exit 0
#                  when done
#   train67        the run, resumable in runs/train67[_NAME][_s<N>]_p<P>g<G>: checkpoints every 2 steps, the newest 2
#                  kept, and the model part of every 20th in runs/<dir>/milestones/ (MA_RL_KEEP_EVERY, rl/rl_inner.sh)
# The first failure ends the job: nothing is retried. Results: tmp/flint_s5/runs/<job>/. One container with a PID
# namespace of its own (--pid) holds everything (see rl/rollout_job.sh); bounds: timeout -s KILL around it, the GPU
# watch (rl/gpu_watch.sh) and cleanup of only the process groups this job started (serving/check/procs.sh).
set -uo pipefail
MODE=${1:-}
# run20: wq --partial --time 23:00:00; train67: wq --partial --time 23:00:00; train67_test: wq --partial --time 6:00:00
# (2 steps of 128k at B = 32: ~20 min start, 2 x (<= 60 min rollout + ~16 min train + sync and save) ~ 3 h; its time
# stop comes 105 min before the end, at 4.25 h); train67_smoke: wq --time 1:45:00
case "$MODE" in smoke|smoke_tp4|smoke_tp8) LIMIT=4800;; train67_smoke) LIMIT=6000;; base_tp8) LIMIT=8400;; test2) LIMIT=9600;;
  run20|train67) LIMIT=81000;; train67_test) LIMIT=21600;;
  *) echo "usage: rl_job.sh smoke[_tp4|_tp8]|base_tp8|test2|run20|train67[_test|_smoke] [P G [STEPS]] [set=NAME] [seed=N] [levers=LIST] [cmdtimeout=N]"; exit 1;; esac
# train67: set=NAME, seed=N, levers=LIST and cmdtimeout=N at any place after the mode; the other arguments are P G STEPS in
# this order
T67_SET= T67_SEED= SEED_GIVEN= T67_LEVERS= LEVERS_GIVEN= T67_CMD_TIMEOUT= CMDT_GIVEN= T67_ELASTIC= ELASTIC_GIVEN=
T67_ASYNC= ASYNC_GIVEN= T67_STALE= STALE_GIVEN= T67_PLACE= PLACE_GIVEN= T67_NOSAVE= NOSAVE_GIVEN= POS=()
for x in "${@:2}"; do
  case "$x" in set=*) T67_SET=${x#set=};; seed=*) T67_SEED=${x#seed=} SEED_GIVEN=1;;
    levers=*) T67_LEVERS=${x#levers=} LEVERS_GIVEN=1;; cmdtimeout=*) T67_CMD_TIMEOUT=${x#cmdtimeout=} CMDT_GIVEN=1;;
    elastic=*) T67_ELASTIC=${x#elastic=} ELASTIC_GIVEN=1;;
    async=*) T67_ASYNC=${x#async=} ASYNC_GIVEN=1;; stale=*) T67_STALE=${x#stale=} STALE_GIVEN=1;;
    place=*) T67_PLACE=${x#place=} PLACE_GIVEN=1;; save=0) T67_NOSAVE=1 NOSAVE_GIVEN=1;;
    *) POS+=("$x");; esac
done
T67_P=${POS[0]:-4} T67_G=${POS[1]:-8} T67_STEPS=${POS[2]:-40}
case "$MODE" in train67*) [[ $T67_P =~ ^[1-9][0-9]*$ && $T67_G =~ ^[1-9][0-9]*$ && $T67_STEPS =~ ^[1-9][0-9]*$ ]] \
  || { echo "P G STEPS must be positive integers: $T67_P $T67_G $T67_STEPS"; exit 1; }
  [ ${#POS[@]} -le 3 ] || { echo "too many arguments: ${POS[*]}"; exit 1; }
  [ -z "$T67_SET" ] || [[ $T67_SET =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "set=NAME: letters, digits, _ and -: $T67_SET"; exit 1; }
  [ -z "$SEED_GIVEN" ] || { [[ $T67_SEED =~ ^[1-9][0-9]{0,3}$ ]] && [ "$T67_SEED" -ge 67 ]; } \
    || { echo "seed=N: an integer from 67 (the default run) to 9999: $T67_SEED"; exit 1; }
  if [ -n "$LEVERS_GIVEN" ]; then
    [[ $T67_LEVERS =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || { echo "levers=LIST: a comma list of tp4 gate graphs arfusion pgraph moecfg: $T67_LEVERS"; exit 1; }
    for l in ${T67_LEVERS//,/ }; do
      case "$l" in tp4|gate|graphs|arfusion|pgraph|moecfg) ;; *) echo "levers=LIST: unknown lever $l (tp4 gate graphs arfusion pgraph moecfg)"; exit 1;; esac
    done
    T67_LEVERS=$(tr , '\n' <<< "$T67_LEVERS" | sort -u | paste -sd, -)
  fi
  [ -z "$CMDT_GIVEN" ] || { [[ $T67_CMD_TIMEOUT =~ ^[1-9][0-9]{0,2}$ ]] && [ "$T67_CMD_TIMEOUT" -ge 5 ] && [ "$T67_CMD_TIMEOUT" -le 600 ]; } \
    || { echo "cmdtimeout=N: seconds per command, 5 to 600 (default 60): $T67_CMD_TIMEOUT"; exit 1; }
  [ -z "$ELASTIC_GIVEN" ] || [ "$T67_ELASTIC" = 1 ] || { echo "elastic=1 is the only value: $T67_ELASTIC"; exit 1; }
  [ -z "$ASYNC_GIVEN" ] || [ "$T67_ASYNC" = 1 ] || { echo "async=1 is the only value: $T67_ASYNC"; exit 1; }
  [ -z "$ASYNC_GIVEN" ] || [ "$MODE" != train67_smoke ] || { echo "async=1 is for train67 and train67_test"; exit 1; }
  [ -z "$STALE_GIVEN" ] || { [ -n "$ASYNC_GIVEN" ] && [[ $T67_STALE =~ ^[1-9][0-9]?$ ]]; } \
    || { echo "stale=N (1-99) goes with async=1: $T67_STALE"; exit 1; }
  [ -z "$PLACE_GIVEN" ] || { [ -n "$ASYNC_GIVEN" ] && [ "$T67_PLACE" = 2numa ]; } || { echo "place=2numa goes with async=1: $T67_PLACE"; exit 1; }
  [ -z "$NOSAVE_GIVEN" ] || [ "$MODE" = train67_test ] || { echo "save=0 is for train67_test"; exit 1; }
  ;;
  *) [ -z "$T67_SET$SEED_GIVEN$LEVERS_GIVEN$CMDT_GIVEN$ELASTIC_GIVEN$ASYNC_GIVEN$STALE_GIVEN$PLACE_GIVEN$NOSAVE_GIVEN" ] || { echo "set=NAME, seed=N, levers=LIST, cmdtimeout=N, elastic=1, async=1 and stale=N are for the train67 modes only, not $MODE"; exit 1; };; esac
T67_CMD_TIMEOUT=${T67_CMD_TIMEOUT:-60}
# seed=N (default 67: the run as before, byte for byte): the data order (rl/train67_data.py: epoch e is the permutation
# of random.Random(N + e)) and miles' seeds (rl/rl_inner.sh: --seed 1 + 100 (N - 67), the actor's torch seed and the
# SGLang engines' base seed, engine i serving with seed + i; --rollout-seed 42 + 100 (N - 67)): 100 apart, so no two
# replicates share a seed. The run dirs hold _s<N> for N != 67, so a replicate never resumes another run.
T67_SEED=${T67_SEED:-67}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
[ -n "$JID" ] || { echo "no job id"; exit 1; }
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-elastic5
MILES=/scratch/gpfs/GROUP/USER/project/miles-numa2
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/$JID
# run20 lives in one fixed directory, so that any submission resumes it (the checkpoints, steps.jsonl, the traces)
[ "$MODE" = run20 ] && OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/run20_p0
# so do the resumable train67 runs, one directory per problem set, seed and batch shape (a set run never resumes a run on
# all 67 problems, nor the reverse; a seed-N replicate never resumes the seed-67 run)
T67_TAG=${T67_SET:+_$T67_SET}
[ "$T67_SEED" = 67 ] || T67_TAG=${T67_TAG}_s$T67_SEED
# levers=LIST: another engine, so its own run dir (a run never resumes on another engine setup)
[ -z "$T67_LEVERS" ] || T67_TAG=${T67_TAG}_k${T67_LEVERS//,/-}
# cmdtimeout=N other than 60: another game, so its own run dir too
[ "$T67_CMD_TIMEOUT" = 60 ] || T67_TAG=${T67_TAG}_t$T67_CMD_TIMEOUT
# elastic=1: engines in other jobs join, so its own run dir too
[ -z "$T67_ELASTIC" ] || T67_TAG=${T67_TAG}_e
# async=1: another layout and estimator, so its own run dir too (_a, or _a<N> with a staleness bound)
[ -z "$T67_ASYNC" ] || T67_TAG=${T67_TAG}_a$T67_STALE
[ "$MODE" = train67 ] && OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/train67${T67_TAG}_p${T67_P}g${T67_G}
[ "$MODE" = train67_test ] && OUT=/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/train67_test${T67_TAG}_p${T67_P}g${T67_G}
RESUMABLE=; case "$MODE" in run20|train67|train67_test) RESUMABLE=1;; esac
# tests/test_train67_seed.py: the parsed arguments and the run dir only, nothing made or started
[ -z "${RL_JOB_PRINT_ONLY:-}" ] || { echo "MODE=$MODE P=$T67_P G=$T67_G STEPS=$T67_STEPS SET=$T67_SET SEED=$T67_SEED OUT=$OUT${T67_LEVERS:+ LEVERS=$T67_LEVERS}$([ "$T67_CMD_TIMEOUT" = 60 ] || echo " CMDTIMEOUT=$T67_CMD_TIMEOUT")${T67_PLACE:+ PLACE=$T67_PLACE}${T67_NOSAVE:+ NOSAVE=1}${T67_ELASTIC:+ ELASTIC=1}${T67_ASYNC:+ ASYNC=1}${T67_STALE:+ STALE=$T67_STALE}"; exit 0; }
# the planned time stop: one step and a save must fit after it (run20: 43 min a step; train67 at B = 32: up to the
# 60-min wall cap, ~4 x run20's 4-min train time, sync and save, ~85 min)
STOP_S=4500; case "$MODE" in train67*) STOP_S=6300;; esac
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID
mkdir -p "$OUT" "$TMPDIR/tmp" "$TMPDIR/datasets"
rm -f "$OUT/exit_code" "$OUT/stop_request" "$OUT/save_now" "$OUT/train_segment.log"   # this segment's stop state (run20 resumes in the same directory)
# elastic=1: the trainer owns the run directory; a STOP from an earlier end of the run is void once it runs again
[ -n "$T67_ELASTIC" ] && rm -f "$OUT/STOP" "$OUT/engine_spec.json"
FRESH=; [ -e "$OUT/actor/latest_checkpointed_iteration.txt" ] || FRESH=1
# an interrupted segment's episode traces belong to no rollout of this one
[ -d "$OUT/traces_live" ] && mv "$OUT/traces_live" "$OUT/traces_live.stale_$(date +%s)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source "$REPO/serving/check/procs.sh"
source "$REPO/rl/gpu_watch.sh"
WD= MEM= SMI= HB=
# (the warm-up engine's own directory too, should it be killed before its own trap ran)
trap 'stop "${STARTED[@]}"; kill $WD $MEM $SMI $HB 2>/dev/null; rm -rf -- "/scratch/gpfs/GROUP/USER/tmp/job/$JID" "/scratch/gpfs/GROUP/USER/tmp/job/${JID}w"' EXIT
trap 'exit "$(cat "$OUT/exit_code" 2>/dev/null || echo 143)"' TERM INT   # a planned stop leaves its exit code there
echo "[s5] $(date -u +%FT%TZ) $MODE node=$(hostname) job=$JID gpus=${CUDA_VISIBLE_DEVICES:-}"
# this segment's end: Slurm's (or a fetching host's) SLURM_JOB_END_TIME, else the time limit of the parent timeout (a
# fetched guest runs under `timeout <s>s`), else LIMIT; the planned time stop comes 75 min before it
END=$(( $(date +%s) + LIMIT )) SRC=LIMIT
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  [ "$SLURM_JOB_END_TIME" -lt "$END" ] && END=$SLURM_JOB_END_TIME; SRC=SLURM_JOB_END_TIME
elif d=$(tr '\0' '\n' < /proc/$PPID/cmdline 2>/dev/null | grep -xE '[0-9]+s' | tail -1 | tr -d s) && [ -n "$d" ]; then
  [ $(( $(date +%s) + d )) -lt "$END" ] && END=$(( $(date +%s) + d )); SRC="parent timeout ${d}s"
fi
echo "[s5] segment end $(date -d @"$END" +%FT%T) ($((END - $(date +%s))) s left; from $SRC); planned stop $((STOP_S / 60)) min before it"
# a segment too short for one step and a save starts nothing (a fetching host may offer a resumable job as little as 30
# min, 2026-11-02: w922 got 72 min): 98 puts the job back in the queue, not before 30 min, without counting the run
if [ "$MODE" = train67 ] && [ $((END - $(date +%s))) -lt "$STOP_S" ]; then
  echo "[s5] $((END - $(date +%s))) s left, under the $STOP_S s a step and a save need: too short for one step, back to the queue"
  exit 98
fi
echo "mas-rl $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -uno | wc -l)" \
  "miles $(git -C "$MILES" rev-parse --short HEAD) dirty=$(git -C "$MILES" status --porcelain -uno | wc -l)" | tee "$OUT/code"
python3 - "$FLINT" <<'PY' || exit 1
import json, os, sys
d = sys.argv[1]
files = sorted(set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values()))
missing = [f for f in files + ["config.json", "tokenizer.json"] if not os.path.isfile(os.path.join(d, f))]
sys.exit(f"[s5] PREFLIGHT_FAIL {d}: {missing}") if missing else print(f"[s5] preflight: {len(files)} weight files")
PY
PKG=$TMPDIR/flint_pkg
bash "$REPO/serving/build_flint_sglang.sh" "$PKG" || exit 1
# the RL problem: val40 problem 0, its directory also in the metadata (miles hands the agent only the metadata); the
# row 1000 times, so that miles' epoch (one pass over the rows, a save point) never ends inside a run
python3 - "$REPO/data/fcs11_val40_agent.jsonl" "$OUT/data_p0.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if json.loads(x)["metadata"]["pid"] == "0"]
assert len(rows) == 1
rows[0]["metadata"]["label"] = rows[0]["label"]
open(sys.argv[2], "w").write((json.dumps(rows[0]) + "\n") * 1000)
PY
# train67: the training problems (all 67, or set=NAME's) in a fixed order (seed N, 30 epochs: miles' epoch end, a save
# point, comes after n x 30 / P steps, 255 for 34 problems at P = 4, outside any run). A resumed run must read the same
# file: an existing one is checked, not rebuilt.
if [[ $MODE == train67* ]]; then
  echo "[s5] problem set: ${T67_SET:-none (all 67)}"   # train67_data.py's line below gives its size
  echo "[s5] seed $T67_SEED: data order random.Random($T67_SEED + epoch); miles --seed $((1 + 100 * (T67_SEED - 67)))" \
    "(actor torch seed; engine i: + i), --rollout-seed $((42 + 100 * (T67_SEED - 67)))"
  python3 "$REPO/rl/train67_data.py" "$REPO/data/fcs11_train67_agent.jsonl" "$OUT/data_train67.new.jsonl" "$T67_P" \
    "$T67_SEED" ${T67_SET:+--set "$T67_SET"} || exit 1
  if [ -e "$OUT/data_train67.jsonl" ] && ! cmp -s "$OUT/data_train67.jsonl" "$OUT/data_train67.new.jsonl"; then
    echo "[s5] $OUT/data_train67.jsonl differs from the one this code builds: a resumed run would see other problems"; exit 1
  fi
  mv "$OUT/data_train67.new.jsonl" "$OUT/data_train67.jsonl"
fi
# one line per GPU and minute; stops the job when a GPU idles (rl/gpu_watch.sh), counting from the first rollout
# request: the start (actor weights 8 min in w583, then the engines; GPUs idle 11 min) may take up to 35 min. An
# engine whose episodes are done idles until the group's slowest episode ends (128k episodes take 17-34 min; w611's
# 8k episodes 18-28 min and one over 31), so a loaded GPU may idle 45 min (the host is cancelled at 90)
# train67: 32 episodes keep the one engine busy until the last one ends (up to the 60-min cap): 30 min at 0% is a hang
IDLE=45; [[ $MODE == train67* ]] && IDLE=30
# async=1: the actor's 4 GPUs wait for the first P groups (up to the 60-min wall cap) before its first step, so idle
# minutes count from the first step on (GPU_ACTIVE_CMD), with 85 min at 0% in any phase as the outer bound; between
# steps they wait for the next P groups by design (with one engine about a rollout minus a train step, 30-50 min:
# rollouts of 30-61 min, train steps of 5-14 min in w786), so a hang is 60 min at 0%, under the host's 90-min cancel
ACTIVE= MAXIDLE=
[ -n "$T67_ASYNC" ] && ACTIVE="grep -q 'perf/train_time' '$OUT/train_segment.log' 2>/dev/null" MAXIDLE=85 IDLE=60
GPU_ACTIVE_CMD=$ACTIVE GPU_IDLE_MAX_MIN=$MAXIDLE GPU_IDLE_MIN=$IDLE gpu_watch $$ "$OUT/gpu.log" "grep -q 'POST /v1/chat/completions' '$OUT/train_segment.log' 2>/dev/null" 35 &
WD=$!
# host memory every 10 s (the node's used MiB, this job's cgroup MiB: what --mem bounds, and the cgroup's
# non-reclaimable MiB: memory.current minus its page cache, memory.stat file - shmem) and the planned stops. The guard
# reads the non-reclaimable part: the kernel drops clean page cache before the cgroup reaches memory.max (900 GiB on a
# pilot host), and that cache is not this job's to count: w765 (2026-11-01) was stopped at 875 GiB of memory.current
# while ~80 GiB of it was clean cache of the mmap-loaded model files (fadvise DONTNEED on w766's node freed 74 GiB);
# a pilot host's cgroup also keeps the cache of earlier guests, and checkpoint writes add more. A
# stop request ($OUT/stop_request: "<exit code> <reason>", from here or from flint.miles_agent.log_rollout's rules)
# asks miles for a checkpoint at its next save point (--save-trigger-sentinel $OUT/save_now); once miles has written it
# (the sentinel is gone) the job stops with that exit code. Requests: non-reclaimable memory over 860 GiB (3: a
# failure); less than 75 min left of LIMIT (99: wq requeues it, the next segment resumes from the checkpoint); run20's
# first segment stops after its first checkpoint (99), so the second one proves the resume. Over 875 GiB the job stops
# at once (3).
CG=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup | sed -E 's#(/job_[0-9]+)/.*#\1#')/memory.current   # the job's
MS=${CG%memory.current}memory.stat
TRACKER=$OUT/actor/latest_checkpointed_iteration.txt
( t0=$SECONDS asked=
  request() { [ -e "$OUT/stop_request" ] || echo "$1" > "$OUT/stop_request"; }
  finish() { echo "$1" | cut -d' ' -f1 > "$OUT/exit_code"; echo "PLANNED_STOP: $1" >&2; kill -TERM $$; }
  while sleep 10; do
    cur=$(cat "$CG" 2>/dev/null || echo 0)
    pc=$(awk '$1 == "file" {f = $2} $1 == "shmem" {s = $2} END {printf "%.0f\n", f - s}' "$MS" 2>/dev/null)   # page cache bytes
    [[ $pc =~ ^[0-9]+$ ]] || pc=0   # no memory.stat: count everything (as before)
    cg=$((cur / 1048576)) nr=$(((cur - pc) / 1048576))
    echo "$(date +%s) $(free -m | awk '/Mem:/{print $3}') $cg $nr"
    if [ "$nr" -gt $((875 * 1024)) ]; then finish "3 host memory: $nr MiB non-reclaimable (cgroup $cg MiB) over 875 GiB"; break; fi
    [ "$nr" -gt $((860 * 1024)) ] && request "3 host memory: $nr MiB non-reclaimable (cgroup $cg MiB) over 860 GiB"
    [ -n "$RESUMABLE" ] && [ "$(date +%s)" -gt $((END - STOP_S)) ] && request "99 time: less than $((STOP_S / 60)) min left"
    if [ "$MODE" = run20 ] && [ -n "$FRESH" ] && [ -e "$TRACKER" ]; then finish "99 first segment: checkpoint $(cat "$TRACKER") written, resume next"; break; fi
    if [ -e "$OUT/stop_request" ] && [ -z "$asked" ]; then touch "$OUT/save_now"; asked=1; fi
    if [ -n "$asked" ] && [ ! -e "$OUT/save_now" ]; then finish "$(cat "$OUT/stop_request")"; break; fi
  done ) >> "$OUT/host_mem.txt" 2>> "$OUT/stops.log" &
MEM=$!
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$OUT/gpu_mem.csv" 2>/dev/null &
SMI=$!
# elastic=1: the node's routable address (miles' weight push to engines in other jobs: Ray, the NCCL master), and the
# trainer's heartbeat for those jobs (rl/registry.py trainer-beat, every 30 s while this job runs)
IP=127.0.0.1
if [ -n "$T67_ELASTIC" ]; then
  IP=$(ip -4 -o addr show eno8303 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
  [ -n "$IP" ] || IP=$(hostname -i | tr ' ' '\n' | grep -E '^172\.' | head -1)
  [ -n "$IP" ] || { echo "[s5] elastic: no routable address (eno8303)"; exit 1; }
  ( T0=$(date +%s); while true; do timeout 60 python3 "$REPO/rl/registry.py" trainer-beat "$OUT" "$JID" "$(hostname)" "$IP" "$T0"; sleep 30; done ) &
  HB=$!
  echo "[s5] elastic: trainer at $IP; engine jobs: wq submit --partial $REPO/rl/elastic_engine.sh $OUT"
fi
# place=2numa: which GPUs the actor takes (miles MILES_GPU_ORDER, rl/gpu_order.py): 2 per NUMA node, so its host memory fits; the run's own engine gets the rest
GPU_ORDER=
if [ -n "$T67_PLACE" ]; then
  if GPU_ORDER=$(timeout 90 python3 "$REPO/rl/gpu_order.py" "${CUDA_VISIBLE_DEVICES:-}"); then
    echo "[s5] place=2numa: GPU order $GPU_ORDER (the actor takes the first 4)"
  else
    GPU_ORDER=
    echo "[s5] place=2numa: this node's GPUs are not 4 + 4 on two NUMA nodes: the actor takes the first 4 of ${CUDA_VISIBLE_DEVICES:-}"
  fi
fi
# async=1: the warm-up engine's files are this segment's (flint.elastic_rollout reads RUN/warmup/launched)
[ -z "$T67_ASYNC" ] || rm -rf -- "$OUT/warmup"
SEG_T0=$(date +%s)
start timeout -k 300 "$LIMIT" apptainer exec --pid --nv --contain --cleanenv --writable-tmpfs \
  --bind /scratch/gpfs/GROUP --bind "$TMPDIR/tmp":/tmp --bind "$TMPDIR/datasets":/root/datasets --bind /proc:/host/proc \
  --bind /dev/shm --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env JID="$JID" --env OUT="$OUT" --env PKG="$PKG" --env TMPDIR="$TMPDIR" --env T67_P="$T67_P" --env T67_G="$T67_G" \
  --env T67_STEPS="$T67_STEPS" --env T67_SEED="$T67_SEED" --env T67_LEVERS="$T67_LEVERS" --env T67_CMD_TIMEOUT="$T67_CMD_TIMEOUT" \
  --env T67_ELASTIC="$T67_ELASTIC" --env T67_ASYNC="$T67_ASYNC" --env T67_STALE="$T67_STALE" --env IP="$IP" --env MILES_GPU_ORDER="$GPU_ORDER" --env T67_NOSAVE="$T67_NOSAVE" \
  "$SIF" bash "$REPO/rl/rl_inner.sh" "$MODE"
# async=1: the warm-up engine on the actor's 4 GPUs (the first 4 of this job's, or of GPU_ORDER: miles puts the actor there and its own engine on the other 4), loading the
# base checkpoint on a fresh start and the checkpoint's weights at a resume while the trainer starts; it ends on its own (RUN/warmup/leave at the first step's data, or a
# failure) and with this job at the latest (the EXIT trap). So the actor's GPUs serve during the first wait, which is up to one rollout (the 60-min wall cap) and passed CSES's 30 min.
WARM_CKPT=
if [ -n "$T67_ASYNC" ] && [ -z "$FRESH" ] && [ -z "$T67_NOSAVE" ]; then
  WARM_CKPT="$OUT/actor/iter_$(printf %07d "$(cat "$OUT/actor/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)")/model"
  [ -d "$WARM_CKPT" ] || { echo "[s5] async: no checkpoint at $WARM_CKPT: no warm-up engine at this resume"; WARM_CKPT=; }
fi
if [ -n "$T67_ASYNC" ] && { [ -n "$FRESH" ] || [ -n "$WARM_CKPT" ]; }; then
  IFS=, read -r -a GPUS <<< "${GPU_ORDER:-${CUDA_VISIBLE_DEVICES:-}}"
  mkdir -p "$OUT/warmup" && echo "${JID}w" > "$OUT/warmup/launched"
  start env EL_WARMUP=1 EL_SPEC_AFTER="$((SEG_T0 - 120))" EL_WARM_CKPT="$WARM_CKPT" FETCHED_JOB_ID="${JID}w" CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]:0:4}")" \
    bash "$REPO/rl/elastic_engine.sh" "$OUT" > "$OUT/warmup/engine.log" 2>&1
  echo "[s5] async: warm-up engine ${JID}w on GPUs ${GPUS[*]:0:4}${WARM_CKPT:+, from $WARM_CKPT} (log $OUT/warmup/engine.log)"
fi
wait "${STARTED[0]}"
rc=$?
echo "[s5] $(date -u +%FT%TZ) container exit $rc"
# elastic=1: the engine jobs end (exit 0) when the run is over: done, or failed; a stop that wq requeues is not
if [ -n "$T67_ELASTIC" ]; then
  case $rc in 98|99|124|143) ;; *) echo "trainer $JID exit $rc ($(date -u +%FT%TZ))" > "$OUT/STOP";; esac
fi
python3 - "$OUT" <<'PY'
import collections, sys
out = sys.argv[1]
peak = collections.defaultdict(int)
for line in open(f"{out}/gpu_mem.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3 and p[2].isdigit():
        peak[int(p[1])] = max(peak[int(p[1])], int(p[2]))
rows = [l.split() for l in open(f"{out}/host_mem.txt")]
host = max((int(r[2]) for r in rows if len(r) >= 3), default=0)
nonrec = max((int(r[3]) for r in rows if len(r) >= 4), default=0)
print("[s5] peak GPU memory (2 s samples, MiB):", dict(sorted(peak.items())), "| peak job memory (10 s samples, MiB):",
      f"cgroup {host}, non-reclaimable {nonrec}")
PY
exit $rc
