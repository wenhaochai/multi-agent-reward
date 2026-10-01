#!/bin/bash
# official Frontier-CS judge (go-judge + node orchestrator) inside an Ubuntu sif; ports from $GJP (go-judge) and $PORT (api)
G=/scratch/gpfs/GROUP/USER/project/gojudge
NODE=/scratch/gpfs/GROUP/USER/nvm/versions/node/v22.17.1/bin/node
cd $G
./go-judge -mount-conf $G/mount.yaml -http-addr 127.0.0.1:$GJP -parallelism ${GJ_PARALLELISM:-8} -dir /dev/shm/gj_$GJP > $G/gj_$GJP.log 2>&1 &
sleep 2
cd $G/app && GJ_ADDR=http://127.0.0.1:$GJP JUDGE_WORKERS=${JUDGE_WORKERS:-8} TESTLIB_INSIDE=/testlib PORT=$PORT \
  SUBMISSIONS_DIR=/tmp/gj_subs_$PORT exec $NODE server.js
