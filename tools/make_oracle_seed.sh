#!/bin/bash
# EMIT dirs for a second seed of the five oracle arms on the page's main figure (user 2026-10-19 by popup: "团队奖励第 2
# 个种子"): the team value-pretraining producer, team shared, team bonus (both replay it), team difference with 4 samples
# per side (cfk4, live rollouts 0-9 like its s42 run), seq shared (5 revisions) and par indiv (best of 5).
# Recipe = submit_fcs_oracle.sh (the s42 arms' launcher, judge as of 99bf9c7/fdad018) with SEED, plus the run-dir
# settings the s42 arms carried at their end (logs/<arm>-<job>.out train args and runs/<arm>/extra_{env,args}):
#   no-pad training (--variable-rollout-samples GBS --bshd-pad-per-sample), NCCL_DEBUG=WARN, NCCL_NVLS_ENABLE=0,
#   MA_FO_CF_K=4 for the difference arm, training data fcs_train200_team (owner: the same 200 problems for a fair
#   comparison, 2026-10-11)
# Changed from s42, none of them in the training math:
#   in-training eval on the arm's own game, the 98 classic val problems x 1 (owner 2026-10-08: interactive skipped; the
#     cfk4 run switched to classic sets on 2026-10-14); step 0 is not re-evaluated (--skip-eval-before-train): the SFT
#     model's step 0 is the s42 runs' step 0
#   --save-interval 2 (s42: 5): the segments are fetchable-partial, so a host may end one early
#   bash tools/make_oracle_seed.sh SEED OUT_DIR   (writes OUT_DIR/<arm>/{smoke,main}/{run,env,args} and the run dirs'
#   extra_env; refuses an OUT_DIR or a run dir that exists)
set -euo pipefail
seed=$1 out=$2
code=$(cd "$(dirname "$0")/.." && pwd)
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build D=$B/data
[[ -e $out ]] && { echo "$out exists" >&2; exit 1; }
nccl=$'# CUDA 999 / hang after wake_up (fcs_oracle.yaml cuda999_nvls_2026_10_08), as the s42 arms ended\nNCCL_DEBUG=WARN\nNCCL_NVLS_ENABLE=0'
emit() {  # emit ARM PART GBS EXTRA_ENV KEY=VALUE... (the submit_fcs_oracle.sh settings)
  local arm=$1 part=$2 gbs=$3 xenv=$4; shift 4
  local dir=$out/$arm/$part xa="--variable-rollout-samples $gbs --bshd-pad-per-sample"
  if [[ $part == main ]]; then
    local game; game=$(printf '%s\n' "$@" | sed -n 's/^GAME=//p')
    case " $* " in *" PRETRAIN=1 "*) xa+=" --skip-eval-before-train" ;;
      *) xa+=" --eval-prompt-data fo_${game}_classic $D/fcs_val98_fo_$game.jsonl --n-samples-per-eval-prompt 1 --save-interval 2 --skip-eval-before-train" ;; esac
  fi
  env -i HOME="$HOME" PATH="$PATH" "$@" EMIT="$dir" EXTRA_ARGS="$xa" bash "$code/submit_fcs_oracle.sh" > /dev/null
  local run; run=$(cat "$dir/run")
  [[ ! -e $B/runs/$run/extra_env && ! -e $B/runs/$run/ckpt ]] || { echo "run dir $run is not new" >&2; exit 1; }
  printf '%s\n' "$xenv" > "$B/runs/$run/extra_env"
  echo "$arm $part $run"
}
s=SEED=$seed
emit vp smoke 36 "$nccl" GAME=team REWARD=shared SMOKE=1 PRETRAIN=1 RUN_TAG=s$seed
emit vp main 1440 "$nccl" GAME=team REWARD=shared PRETRAIN=1 $s
for r in shared bonus; do
  emit team_$r smoke 36 "$nccl" GAME=team REWARD=$r SMOKE=1 RUN_TAG=s$seed
  emit team_$r main 1440 "$nccl" GAME=team REWARD=$r VP=1 VP_DEP=pending $s
done
cf=$'# 2026-10-11 (owner): difference reward with 4 samples per side, as fcs_oracle_team_diff_q9b-fcs-sft_s42_cfk4\nMA_FO_CF_K=4\n'"$nccl"
emit team_diff_cfk4 smoke 36 "$cf" GAME=team REWARD=diff SMOKE=1 RUN_TAG=s${seed}_cfk4
emit team_diff_cfk4 main 1440 "$cf" GAME=team REWARD=diff RUN_TAG=cfk4 $s
emit seq_shared smoke 20 "$nccl" GAME=seq REWARD=shared SMOKE=1 RUN_TAG=s$seed
emit seq_shared main 480 "$nccl" GAME=seq REWARD=shared $s
emit par_indiv smoke 20 "$nccl" GAME=par REWARD=indiv SMOKE=1 RUN_TAG=s$seed
emit par_indiv main 480 "$nccl" GAME=par REWARD=indiv $s
