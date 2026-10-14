#!/bin/bash
# eval-only configs (2026-10-13). Env read off submit_fcs_oracle.sh: team arms be49a4a (submitted 10-04 14:30),
# seq/par 388dd8b (10-03 20:46); MA_FO_MAX_LEN of seq/par is their extra_env's 65536 (it overrode 61440).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/miles-q38-build-evalonly
O=/scratch/gpfs/GROUP/USER/tmp/evalonly
T="MA_FO_SUBS=4 MA_FO_ROUNDS=5 MA_FO_BUDGET=32768 MA_FO_PLAN_BUDGET=16384 MA_FO_SUB_TESTS=3 MA_FO_CF=0 MA_FO_MAX_LEN=65536"
for r in shared bonus diff; do
  python3 tools/make_evalonly.py fcs_oracle_team_${r}_q9b-fcs-sft_s42 team 4 ev4 $O/team_$r --env MA_FO_GAME=team MA_FO_REWARD=$r $T
done
S="MA_FO_SUBS=4 MA_FO_ROUNDS=5 MA_FO_BUDGET=32768 MA_FO_PLAN_BUDGET=8192 MA_FO_MAX_LEN=65536"
python3 tools/make_evalonly.py fcs_oracle_seq_shared_q9b-fcs-sft_s42 seq 4 ev4 $O/seq_shared --env MA_FO_GAME=seq MA_FO_REWARD=shared $S
python3 tools/make_evalonly.py fcs_oracle_par_indiv_q9b-fcs-sft_s42 par 4 ev4r12 $O/par_indiv --rounds 12 --env MA_FO_GAME=par MA_FO_REWARD=indiv $S
