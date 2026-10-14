#!/bin/bash
# Agentic baseline probe (owner 2026-10-13): mini-swe-agent scaffold (miles_team.fcs_agent), the untrained SFT
# initialization, the 98 classic val problems x 2 episodes, every episode traced. Config base: the par_indiv run's
# (same model, parallelism and context), env read off submit_fcs_oracle.sh 388dd8b; MA_FO_* stay for fcs_oracle's
# import (post_process); MA_AG_* set the agent.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/miles-q38-build-agentic
O=/scratch/gpfs/GROUP/USER/tmp/agentic/probe_sft; D=/scratch/gpfs/GROUP/USER/project/miles-q38-build/data
R=/scratch/gpfs/GROUP/USER/project/miles-q38-build/runs
python3 tools/make_evalonly.py fcs_oracle_par_indiv_q9b-fcs-sft_s42 agent 2 agentprobe $O --fresh --module fcs_agent \
  --data $D/fcs_val_classic_agent.jsonl --smoke-data $D/fcs_val_classic_agent_smoke8.jsonl \
  --env MA_FO_GAME=par MA_FO_REWARD=indiv MA_FO_SUBS=4 MA_FO_ROUNDS=5 MA_FO_BUDGET=32768 MA_FO_PLAN_BUDGET=8192 MA_FO_MAX_LEN=65536 \
  --set MA_AG_STEPS=40 MA_AG_TURN=16384 MA_AG_TOKENS=131072 MA_AG_MAX_LEN=65536 MA_AG_TIMEOUT=60 \
        MA_AG_TRACE_DIR=$R/fcs_evalonly_agentprobe_sft/traces \
  --smoke-set MA_AG_STEPS=8 MA_AG_TURN=4096 MA_AG_TOKENS=16384 MA_AG_TRACE_DIR=$R/fcs_evalonly_smoke_agentprobe_sft/traces
