#!/bin/bash
# Dry run of the external-engine pair (rl/engine_job.sh, rl/rl_job_ext.sh) with fake binaries (tests/fake_ext:
# apptainer, nvidia-smi, ray, a fake SGLang server, a fake train.py), on della-vis1 (needs eno8303). No GPU, no
# container, no Slurm. Usage: bash tests/dryrun_ext.sh; exit 0 when every case passes. Each job runs under a hard
# timeout (CASE_S, 300 s). The body is one function, parsed whole before it runs, so editing this file during a run
# cannot make bash resume at a shifted offset (the 2026-10-30 group-D hang: the file was edited mid-run and bash
# started a second engine without its env prefixes).
set -u
main() {
R=$(cd "$(dirname "$0")/.." && pwd)
F=$R/tests/fake_ext
W=${TMPDIR:-/scratch/gpfs/GROUP/USER/tmp}/dryrun_ext_$$
mkdir -p "$W"
cd "$W" || exit 1   # not the repo: `python3 -c` puts the cwd first on sys.path, and the repo's flint/ would shadow the stub
export PATH=$F:$PATH EXT_REPO=$R EXT_MILES=$F EXT_SIF=/nonexistent.sif CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EXT_BEAT_S=2
unset SLURM_JOB_ID SLURM_JOB_END_TIME
IP=$(ip -4 -o addr show eno8303 | awk '{print $4}' | cut -d/ -f1)
fails=()
JOBS0=$(ls /scratch/gpfs/GROUP/USER/tmp/job/ 2>/dev/null)   # per-job TMPDIRs before this run
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails+=("$1"); fi; }
# engine/trainer ID RUN: a job in the foreground; with BG=1 in the background, its PID in BGPID: `timeout` itself
# (a simple command after `&` is exec'd in the fork), so `kill -TERM $BGPID` reaches the job (a function with a
# ( ... ) body adds a bash in between, which a TERM kills alone)
engine() {
  if [ -n "${BG:-}" ]; then FETCHED_JOB_ID=$1 timeout -k 10 "${CASE_S:-300}" bash "$R/rl/engine_job.sh" "$2" > "$W/$1.out" 2>&1 & BGPID=$!
  else FETCHED_JOB_ID=$1 timeout -k 10 "${CASE_S:-300}" bash "$R/rl/engine_job.sh" "$2" > "$W/$1.out" 2>&1; fi; }
trainer() {
  if [ -n "${BG:-}" ]; then FETCHED_JOB_ID=$1 timeout -k 10 "${CASE_S:-300}" bash "$R/rl/rl_job_ext.sh" smoke_ext "$2" > "$W/$1.out" 2>&1 & BGPID=$!
  else FETCHED_JOB_ID=$1 timeout -k 10 "${CASE_S:-300}" bash "$R/rl/rl_job_ext.sh" smoke_ext "$2" > "$W/$1.out" 2>&1; fi; }
upto() { local i; for i in $(seq "$1"); do eval "$2" && return 0; sleep 1; done; echo "timeout after $1 s: $2"; return 1; }

echo "== A: the pair: the engine registers, the trainer finds it, trains, checkpoints, writes STOP; the engine leaves"
RUN=$W/a
BG=1 engine wa_e "$RUN"; EP=$BGPID
FAKE_TRAIN_S=25 trainer wa_t "$RUN"; trc=$?   # > the probe trigger's 15-s poll
wait $EP; erc=$?
ok "A trainer exit 0 ($trc)" '[ $trc = 0 ]'
ok "A engine exit 0 on STOP ($erc)" '[ $erc = 0 ]'
ok "A STOP names the trainer" 'grep -q "trainer wa_t exit 0" "$RUN/STOP"'
ok "A engine record removed" '[ -z "$(ls "$RUN/engines")" ]'
ok "A engine bound the routable address" 'grep -q "url=http://$IP:" "$W/wa_e.out"'
ok "A engine went loading -> serving" 'grep -q "serving after" "$W/wa_e.out"'
ok "A trainer used the registered engine" 'grep -q "engines after .*: $IP:[0-9]* (TP 8, 8 GPUs; jobs wa_e)" "$W/wa_t.out"'
A=$RUN/fake_train_args.json
ok "A train.py got --rollout-external-engine-addrs IP:PORT" "python3 -c \"import json,sys; a=json.load(open('$A')); i=a.index('--rollout-external-engine-addrs'); sys.exit(not a[i+1].startswith('$IP:'))\""
ok "A train.py got broadcast, 8 x TP8, no --colocate" "python3 -c \"import json,sys; a=' '.join(json.load(open('$A'))); sys.exit(not ('--update-weight-transfer-mode broadcast' in a and '--rollout-num-gpus 8 --rollout-num-gpus-per-engine 8' in a and '--colocate' not in a and '--load $RUN/actor' in a and '--num-rollout 2' in a))\""
ok "A ray bound the routable address" 'grep -q -- "--node-ip-address $IP" "$RUN/fake_ray.log"'
ok "A startup probe ran on the agents' settings" 'grep -q "flint probe\] start: stub" "$RUN/probe_start.log"'
ok "A trainer heartbeat written" 'python3 -c "import json; d=json.load(open(\"$RUN/trainer.json\")); assert d[\"job\"]==\"wa_t\" and d[\"ip\"]==\"$IP\""'
ok "A per-job TMPDIRs removed" '[ ! -e /scratch/gpfs/GROUP/USER/tmp/job/wa_e ] && [ ! -e /scratch/gpfs/GROUP/USER/tmp/job/wa_t ]'

echo "== B: a trainer without an engine fails fast and writes STOP"
RUN=$W/b
EXT_WAIT_S=4 trainer wb_t "$RUN"; trc=$?
ok "B trainer exit 2 ($trc)" '[ $trc = 2 ]'
ok "B STOP written" 'grep -q "no engine serving" "$RUN/STOP"'
engine wb_e "$RUN"; erc=$?
ok "B a later engine exits 3 at once ($erc)" '[ $erc = 3 ]'

echo "== C: an engine without a trainer leaves after the grace and writes STOP; a later trainer exits 3"
RUN=$W/c
EXT_TRAINER_GRACE_S=6 engine wc_e "$RUN"; erc=$?
ok "C engine exit 5 ($erc)" '[ $erc = 5 ]'
ok "C STOP written" 'grep -q "no trainer within 6 s" "$RUN/STOP"'
ok "C engine record removed" '[ -z "$(ls "$RUN/engines")" ]'
trainer wc_t "$RUN"; trc=$?
ok "C a later trainer exits 3 ($trc)" '[ $trc = 3 ]'

echo "== D: an engine whose trainer stopped beating leaves (exit 4), without writing STOP"
RUN=$W/d; mkdir -p "$RUN"
( for i in $(seq 15); do python3 "$R/rl/registry.py" trainer-beat "$RUN" wd_t n "$IP" 0; sleep 2; done ) & BP=$!
t0=$SECONDS
EXT_TRAINER_STALE_S=8 EXT_TRAINER_GRACE_S=60 engine wd_e "$RUN"; erc=$?   # the trainer beats for 30 s, then stops
kill $BP 2>/dev/null; wait $BP 2>/dev/null
ok "D engine exit 4 ($erc) within $((SECONDS - t0)) s" '[ $erc = 4 ] && [ $((SECONDS - t0)) -lt 60 ]'
ok "D no STOP" '[ ! -e "$RUN/STOP" ]'

echo "== D2: one trainer beat, already older than --stale at the engine's first look, is a dead trainer (exit 4)"
RUN=$W/d2; mkdir -p "$RUN"
python3 "$R/rl/registry.py" trainer-beat "$RUN" wd2_t n "$IP" 0   # the engine serves ~11 s later: the beat is ~11 s old
t0=$SECONDS
EXT_TRAINER_STALE_S=8 EXT_TRAINER_GRACE_S=60 engine wd2_e "$RUN"; erc=$?
ok "D2 engine exit 4 ($erc) within $((SECONDS - t0)) s" '[ $erc = 4 ] && [ $((SECONDS - t0)) -lt 45 ]'
ok "D2 no STOP" '[ ! -e "$RUN/STOP" ]'

echo "== E: a SIGTERM stops the engine at once (143) and removes its record"
RUN=$W/e
BG=1 engine we_e "$RUN"; EP=$BGPID
upto 120 "grep -qs '\"serving\"' '$RUN/engines/we_e.json'"
kill -TERM $EP; wait $EP; erc=$?
ok "E engine exit 143 ($erc)" '[ $erc = 143 ]'
ok "E record removed" '[ ! -e "$RUN/engines/we_e.json" ]'
ok "E no fake server left" '! pgrep -u "$USER" -f "fake_sglang.py $IP" >/dev/null'

echo "== F: the trainer stops when its engine is gone for EXT_ENGINE_LOST_S (exit 1, STOP)"
RUN=$W/f
BG=1 engine wf_e "$RUN"; EP=$BGPID
FAKE_TRAIN_S=120 EXT_ENGINE_LOST_S=20 BG=1 trainer wf_t "$RUN"; TP=$BGPID
upto 120 "grep -qs 'train_wait start' '$RUN/train.log'"
kill -TERM $EP; wait $EP
t0=$SECONDS; wait $TP; trc=$?
ok "F trainer exit 1 ($trc) within $((SECONDS - t0)) s" '[ $trc = 1 ] && [ $((SECONDS - t0)) -lt 60 ]'
ok "F stops.log names the lost engine" 'grep -q "engine lost: no fresh serving record of wf_e" "$RUN/stops.log"'
ok "F STOP written" 'grep -q "trainer wf_t exit 1" "$RUN/STOP"'
ok "F no fake train.py left" '! pgrep -u "$USER" -f "$F/train.py" >/dev/null'

echo "== G: a trainer whose train.py fails exits non-zero and writes STOP; the engine leaves with 0"
RUN=$W/g
BG=1 engine wg_e "$RUN"; EP=$BGPID
FAKE_TRAIN_S=1 FAKE_TRAIN_RC=7 trainer wg_t "$RUN"; trc=$?
wait $EP; erc=$?
ok "G trainer exit 7 ($trc)" '[ $trc = 7 ]'
ok "G engine exit 0 ($erc)" '[ $erc = 0 ]'

ok "no per-job TMPDIR left by this run" '[ -z "$(comm -13 <(echo "$JOBS0" | sort) <(ls /scratch/gpfs/GROUP/USER/tmp/job/ | sort) | grep -E "^w[a-g]2?_")" ]'
[ ${#fails[@]} -eq 0 ] && rm -rf "$W" || echo "logs kept in $W"
echo "${#fails[@]} failed${fails:+: ${fails[*]}}"
[ ${#fails[@]} -eq 0 ]
}
main "$@"; exit
