#!/bin/bash
# Dry run of the elastic engine job (rl/elastic_engine.sh) with fake binaries (tests/fake_ext: apptainer, nvidia-smi,
# a fake SGLang server), on della-vis1 or della-vis2 (needs eno8303). No GPU, no container, no Slurm, no trainer: a
# background loop plays the trainer's heartbeat (rl/registry.py trainer-beat). Usage: bash tests/dryrun_elastic.sh;
# exit 0 when every case passes. Each job runs under a hard timeout (CASE_S, 120 s); the body is one function, so an
# edit during a run cannot shift bash's read offset.
set -u
main() {
R=$(cd "$(dirname "$0")/.." && pwd)
F=$R/tests/fake_ext
W=${TMPDIR:-/scratch/gpfs/GROUP/USER/tmp}/dryrun_elastic_$$
mkdir -p "$W"
cd "$W" || exit 1
export PATH=$F:$PATH EL_HERE=$R EL_SIF=/nonexistent.sif CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EL_BEAT_S=2 EL_STALE_S=6 EL_DRAIN_MIN_S=2 EL_DRAIN_MAX_S=8 EL_LEAVE_BEFORE_S=1200 EL_MIN_LEFT_S=60
unset SLURM_JOB_ID SLURM_JOB_END_TIME
IP=$(ip -4 -o addr show eno8303 | awk '{print $4}' | cut -d/ -f1)
fails=()
ok() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails+=("$1"); fi; }
upto() { local i; for i in $(seq "$1"); do eval "$2" && return 0; sleep 1; done; echo "timeout after $1 s: $2"; return 1; }
engine() {   # engine ID RUN: in the background, its PID in BGPID
  FETCHED_JOB_ID=$1 timeout -k 10 "${CASE_S:-120}" bash "$R/rl/elastic_engine.sh" "$2" > "$W/$1.out" 2>&1 & BGPID=$!; }
beat() {     # beat RUN: the trainer's heartbeat every second until killed, its PID in HBPID (returns once it is written)
  ( while true; do python3 "$R/rl/registry.py" trainer-beat "$1" t1 node "$IP" 0; sleep 1; done ) & HBPID=$!
  upto 20 "[ -s '$1/trainer.json' ]"; }
spec() {     # spec RUN TP: a run's engine spec (the fields rl/elastic_engine.sh reads)
  mkdir -p "$1"; python3 -c "import json, sys; json.dump({'tp': int(sys.argv[2]), 'repo': sys.argv[3], 'miles': sys.argv[4], 'flint': '/nonexistent', 'argv': [], 'env': {}}, open(sys.argv[1] + '/engine_spec.json', 'w'))" "$1" "$2" "$R" "$F"; }
serving() { python3 -c "import json, sys; r = json.load(open(sys.argv[1])); sys.exit(not (r['state'] == 'serving' and not r.get('leaving')))" "$1/engines/$2.json" 2>/dev/null; }

echo "== A: no live trainer: back to the queue at once (98), nothing registered"
RUN=$W/a; spec "$RUN" 4
engine dry910 "$RUN"; wait $BGPID; rc=$?
ok "A exit 98 ($rc)" '[ $rc = 98 ]'
ok "A no record" '[ -z "$(ls "$RUN/engines" 2>/dev/null)" ]'

echo "== B: STOP at the start: the run is over (0)"
RUN=$W/b; spec "$RUN" 4; echo "done" > "$RUN/STOP"
engine dry920 "$RUN"; wait $BGPID; rc=$?
ok "B exit 0 ($rc)" '[ $rc = 0 ]'

echo "== C: TP4: two engines register, serve, and end on STOP (0)"
RUN=$W/c; spec "$RUN" 4; beat "$RUN"; HB=$HBPID
engine dry930 "$RUN"; EP=$BGPID
upto 60 'serving "$RUN" dry930'
ok "C two TP4 engines in one record, routable, GPUs split" "python3 -c \"import json; r = json.load(open('$RUN/engines/dry930.json')); e = r['engines']; assert len(e) == 2 and all(x['tp'] == 4 and x['url'].startswith('http://$IP:') for x in e) and e[0]['gpus'] == [0, 1, 2, 3] and e[1]['gpus'] == [4, 5, 6, 7], r\""
ok "C the registry lists both as serving" "[ \$(python3 '$R/rl/registry.py' live '$RUN' | wc -l) = 2 ]"
echo "run over" > "$RUN/STOP"
wait $EP; rc=$?; kill $HB 2>/dev/null
ok "C exit 0 on STOP ($rc)" '[ $rc = 0 ]'
ok "C record removed" '[ -z "$(ls "$RUN/engines")" ]'
ok "C per-job TMPDIR removed" '[ ! -e /scratch/gpfs/GROUP/USER/tmp/job/dry930 ]'

echo "== D: the host's end: leaving, drain until no request runs, then 99"
RUN=$W/d; spec "$RUN" 8; beat "$RUN"; HB=$HBPID
echo 2 > "$W/d_running"
FAKE_RUNNING=$W/d_running SLURM_JOB_END_TIME=$(( $(date +%s) + 1200 + 75 )) engine dry940 "$RUN"; EP=$BGPID
upto 60 'serving "$RUN" dry940'
ok "D one TP8 engine" "python3 -c \"import json; r = json.load(open('$RUN/engines/dry940.json')); assert [x['tp'] for x in r['engines']] == [8], r\""
upto 90 "grep -q leaving '$RUN/engines/dry940.json' 2>/dev/null"
ok "D record says leaving" 'grep -q leaving "$RUN/engines/dry940.json"'
ok "D the registry no longer lists it" "[ \$(python3 '$R/rl/registry.py' live '$RUN' | wc -l) = 0 ]"
sleep 4
ok "D still draining while requests run" 'kill -0 $EP 2>/dev/null'
echo 0 > "$W/d_running"
wait $EP; rc=$?; kill $HB 2>/dev/null
ok "D exit 99 after the drain ($rc)" '[ $rc = 99 ]'
ok "D drained" 'grep -q "drained after" "$W/dry940.out"'
ok "D record removed" '[ -z "$(ls "$RUN/engines")" ]'

echo "== E: the trainer's heartbeat goes stale while serving: leave (98)"
RUN=$W/e; spec "$RUN" 4; beat "$RUN"; HB=$HBPID
engine dry950 "$RUN"; EP=$BGPID
upto 60 'serving "$RUN" dry950'
kill $HB
wait $EP; rc=$?
ok "E exit 98 ($rc)" '[ $rc = 98 ]'
ok "E said why" 'grep -q "heartbeat is .* s old" "$W/dry950.out"'

echo "== F: a server process ends while serving: exit 1"
RUN=$W/f; spec "$RUN" 4; beat "$RUN"; HB=$HBPID
engine dry960 "$RUN"; EP=$BGPID
upto 60 'serving "$RUN" dry960'
port=$(python3 -c "import json; print(json.load(open('$RUN/engines/dry960.json'))['engines'][1]['url'].rsplit(':', 1)[1])")
pkill -f "fake_sglang.py $IP $port"
wait $EP; rc=$?; kill $HB 2>/dev/null
ok "F exit 1 ($rc)" '[ $rc = 1 ]'
ok "F record removed" '[ -z "$(ls "$RUN/engines")" ]'
ok "F no fake server left" '! pgrep -f "fake_sglang.py $IP" >/dev/null'

echo "== G: too little time left on the host: back to the queue (98) before loading"
RUN=$W/g; spec "$RUN" 4; beat "$RUN"; HB=$HBPID
SLURM_JOB_END_TIME=$(( $(date +%s) + 1200 + 30 )) engine dry970 "$RUN"; wait $BGPID; rc=$?; kill $HB 2>/dev/null
ok "G exit 98 ($rc)" '[ $rc = 98 ]'
ok "G nothing registered" '[ -z "$(ls "$RUN/engines" 2>/dev/null)" ]'

echo "== H: an operator's leave file: the same clean leave, then 0"
RUN=$W/h; spec "$RUN" 4; beat "$RUN"; HB=$HBPID
engine dry980 "$RUN"; EP=$BGPID
upto 60 'serving "$RUN" dry980'
touch "$RUN/engines/dry980.leave"
upto 30 "grep -q leaving '$RUN/engines/dry980.json' 2>/dev/null"
ok "H record says leaving" 'grep -q leaving "$RUN/engines/dry980.json"'
wait $EP; rc=$?; kill $HB 2>/dev/null
ok "H exit 0 ($rc)" '[ $rc = 0 ]'
ok "H drained" 'grep -q "drained after" "$W/dry980.out"'

echo "== W: the warm-up engine (EL_WARMUP=1): 4 GPUs, no trainer, this segment's spec, RUN/warmup only; leave, gone"
RUN=$W/w; spec "$RUN" 4; touch -d '1 hour ago' "$RUN/engine_spec.json"; T0W=$(date +%s)
EL_WARMUP=1 EL_SPEC_AFTER=$T0W CUDA_VISIBLE_DEVICES=0,1,2,3 engine dryw10 "$RUN"; EP=$BGPID
sleep 4
ok "W waits for this segment's spec" '[ ! -e "$RUN/warmup/engines/dryw10.json" ] && kill -0 $EP 2>/dev/null'
spec "$RUN" 4   # rewritten: newer than EL_SPEC_AFTER
upto 60 'serving "$RUN/warmup" dryw10'
ok "W one TP4 engine on its 4 GPUs" "python3 -c \"import json; r = json.load(open('$RUN/warmup/engines/dryw10.json')); e = r['engines']; assert len(e) == 1 and e[0]['tp'] == 4 and e[0]['gpus'] == [0, 1, 2, 3], r\""
ok "W nothing in RUN/engines (miles never sees it)" '[ -z "$(ls "$RUN/engines" 2>/dev/null)" ]'
ok "W mem_fraction_static 0.72" 'grep -q "mem=0.72" "$RUN/engine_logs/dryw10.0.log"'
echo "1 the first step's data is in" > "$RUN/warmup/leave"
wait $EP; rc=$?
ok "W exit 0 after the leave ($rc)" '[ $rc = 0 ]'
ok "W drained" 'grep -q "drained after" "$W/dryw10.out"'
ok "W gone, its GPUs under 12 GiB" "python3 -c \"import json; g = json.load(open('$RUN/warmup/gone')); assert g['job'] == 'dryw10' and int(g['max_gpu_mib']) < 12288, g\""
ok "W record removed" '[ -z "$(ls "$RUN/warmup/engines")" ]'

echo "== W2: the warm-up engine asked to leave before its spec comes: gone, exit 0"
RUN=$W/w2; mkdir -p "$RUN/warmup"; echo "1 never needed" > "$RUN/warmup/leave"
EL_WARMUP=1 EL_SPEC_AFTER=$(date +%s) CUDA_VISIBLE_DEVICES=0,1,2,3 engine dryw20 "$RUN"; wait $BGPID; rc=$?
ok "W2 exit 0 ($rc)" '[ $rc = 0 ]'
ok "W2 gone" '[ -s "$RUN/warmup/gone" ]'

echo "== W3: the warm-up engine of a resume: exports the checkpoint's weights and serves from them; leave, gone"
RUN=$W/w3; spec "$RUN" 4; T0W=$(date +%s)
EL_WARMUP=1 EL_SPEC_AFTER=$((T0W - 60)) EL_WARM_CKPT=/nonexistent/iter/model FAKE_EXPORT=ok CUDA_VISIBLE_DEVICES=0,1,4,5 engine dryw30 "$RUN"; EP=$BGPID
upto 60 'serving "$RUN/warmup" dryw30'
ok "W3 serves from the export" 'grep -q "model=.*dryw30/warm_weights" "$RUN/engine_logs/dryw30.0.log"'
ok "W3 said so" 'grep -q "export_ckpt" "$W/dryw30.out"'
ok "W3 the engine alive after the export ended" 'kill -0 $EP 2>/dev/null && serving "$RUN/warmup" dryw30'
echo "1 the first step's data is in" > "$RUN/warmup/leave"
wait $EP; rc=$?
ok "W3 exit 0 after the leave ($rc)" '[ $rc = 0 ]'
ok "W3 gone" '[ -s "$RUN/warmup/gone" ]'

echo "== W4: the export fails: exit 1, gone, so the run goes on without this engine"
RUN=$W/w4; spec "$RUN" 4
EL_WARMUP=1 EL_SPEC_AFTER=$(($(date +%s) - 60)) EL_WARM_CKPT=/nonexistent/iter/model FAKE_EXPORT=fail CUDA_VISIBLE_DEVICES=0,1,4,5 engine dryw40 "$RUN"; wait $BGPID; rc=$?
ok "W4 exit 1 ($rc)" '[ $rc = 1 ]'
ok "W4 gone" '[ -s "$RUN/warmup/gone" ]'
ok "W4 the reason in its log" 'grep -q "export boom" "$W/dryw40.out"'
ok "W4 no record left" '[ -z "$(ls "$RUN/warmup/engines" 2>/dev/null)" ]'

echo "== W5: asked to leave during a slow export: out within seconds, gone, nothing left running"
RUN=$W/w5; spec "$RUN" 4
EL_WARMUP=1 EL_SPEC_AFTER=$(($(date +%s) - 60)) EL_WARM_CKPT=/nonexistent/iter/model FAKE_EXPORT=slow CUDA_VISIBLE_DEVICES=0,1,4,5 engine dryw50 "$RUN"; EP=$BGPID
upto 30 "grep -q exporting '$W/dryw50.out'"
sleep 2
T1=$(date +%s); echo "1 the first step's data is in" > "$RUN/warmup/leave"
wait $EP; rc=$?; T2=$(date +%s)
ok "W5 exit 0 ($rc) in $((T2 - T1)) s" '[ $rc = 0 ] && [ $((T2 - T1)) -lt 20 ]'
ok "W5 gone" '[ -s "$RUN/warmup/gone" ]'
ok "W5 no export left" '! pgrep -f "sleep 100" -u "$USER" > /dev/null || ! pgrep -f "dryw50" -u "$USER" > /dev/null'

echo "${#fails[@]} failed${fails:+: ${fails[*]}}"
[ ${#fails[@]} = 0 ] && rm -rf -- "$W"
[ ${#fails[@]} = 0 ]
}
main "$@"
