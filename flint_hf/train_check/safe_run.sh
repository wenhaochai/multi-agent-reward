#!/bin/bash
# Safety frame for the S2 GPU jobs (controller 2026-10-28): run a command in its own process group with a hard time
# limit and an idle-GPU watchdog, and kill the whole group (Ray, servers, trainers) when it ends, times out or idles.
# Usage: safe_run.sh LIMIT_S IDLE_MIN cmd [args...]   (exit: the command's; 137 at the limit; 3 when idled)
# The watchdog polls this job's GPUs (CUDA_VISIBLE_DEVICES) every 60 s; IDLE_MIN consecutive minutes with every GPU at
# <= 1% utilization stop the job, well under the 90 min after which CSES cancels a host with an idle GPU.
LIMIT=${1:?limit seconds} IDLE_MIN=${2:?idle minutes}
shift 2
setsid "$@" < /dev/null &
pid=$!
stop() { kill -TERM -- -$pid 2>/dev/null; sleep 20; kill -KILL -- -$pid 2>/dev/null; }
trap 'stop; exit 143' TERM INT
(
  idle=0 start=$SECONDS
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    if (( SECONDS - start > LIMIT )); then echo "[safe_run] time limit ${LIMIT}s: killing the job"; kill -KILL -- -$pid; exit 0; fi
    busy=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
           | awk '$1 > 1 {n++} END {print n + 0}')
    if [ "$busy" -eq 0 ]; then idle=$((idle + 1)); else idle=0; fi
    if [ "$idle" -ge "$IDLE_MIN" ]; then echo "[safe_run] GPUs idle ${idle} min: killing the job"; touch "${TMPDIR:-/tmp}/safe_run_idle_$pid"; kill -KILL -- -$pid; exit 0; fi
  done
) &
watch=$!
wait $pid; rc=$?
kill $watch 2>/dev/null
kill -KILL -- -$pid 2>/dev/null   # leftovers of the group (Ray workers, servers)
[ -e "${TMPDIR:-/tmp}/safe_run_idle_$pid" ] && { rm -f "${TMPDIR:-/tmp}/safe_run_idle_$pid"; rc=3; }
exit $rc
