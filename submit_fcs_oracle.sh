#!/bin/bash
# Frontier-CS games with an oracle (labs-molt-docs docs/fcs_multiagent_reward.md; miles_team/fcs_oracle.py) under
# EasyPPO on miles, Qwen3.5-9B EasyPPO SFT init, text-only. The EasyPPO flags are submit_fcs.sh's verbatim; what differs
# is the game, the batch shape and the evals. Every submission returns its total and per-test-case scores; an episode
# keeps its best submission (V); every game has 5 submissions of up to BUDGET tokens.
#
# GAME=team REWARD=shared|indiv|gain|diff  1 lead (plans 4 tasks, then writes the final) + 4 subagents
# GAME=seq  REWARD=shared|indiv|diff       one agent, 5 rounds of revision with feedback (baseline: shared)
# GAME=par  REWARD=shared|indiv|diff       5 independent attempts (baseline: indiv = single-agent RL)
# Batch: 480 samples per step: team 16 prompts x 5 episodes x 6 turns (plan, final, 4 subagents); seq/par 16 x 6 x 5.
#   Critic batch 120 (EasyPPO: 4 critic mini-batches).
# Screen: NUM_ROLLOUT 60 (10 critic-only, 6-rollout lr warmup), val every 20 on two oracle-game sets of 172 problems x
#   N_EVAL (2) episodes: team arms play team + par (the same weights as 5 independent attempts), seq plays seq + par,
#   par plays par + team. Step 0 is evaluated once per game (team: the producer; seq and par: their runs).
# Shared value pretraining: PRETRAIN=1 runs the 10 critic-only rollouts of GAME once (runs/fcs_vp_oracle_<game>_...),
#   dumped; VP=1 VP_DEP=<its jid> makes an arm replay them, relabeled with its reward (fcs_oracle.post_process).
# Usage: GAME=... REWARD=... [SEED=42] [NUM_ROLLOUT] [EVAL_EVERY] [N_EVAL] [BUDGET] [MODEL] [SMOKE=1] [PRETRAIN=1]
#        [VP=1 VP_DEP=jid] [QOS] [TIME] [DEP] [NICE] [DRY=1] bash submit_fcs_oracle.sh
#   SMOKE=1: 2 rollouts of 2 prompts x 1 episode, 4096-token turns, 16-problem val sets of the game at the end.
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
D=$B/data
MILES_SRC=/scratch/gpfs/GROUP/USER/project/miles-easyppo
MODEL=${MODEL:-$W/models/Qwen3.5-9B-FCS-SFT}
MG=$MODEL-text; SG=$MODEL-lm
for d in $MG $SG; do [ -f $d/config.json ] || { echo "missing $d: run tools/make_qwen35_text_ckpts.py $MODEL" >&2; exit 1; }; done
: "${GAME:?GAME=team|seq|par}" "${REWARD:?REWARD=shared|indiv|gain|diff}"
case "$GAME:$REWARD" in team:shared|team:indiv|team:gain|team:diff|seq:shared|seq:indiv|seq:diff|par:shared|par:indiv|par:diff) ;;
  *) echo "bad GAME:REWARD $GAME:$REWARD" >&2; exit 1 ;; esac
SEED=${SEED:-42}
TAG=$(basename $MODEL | tr 'A-Z.' 'a-z_' | sed 's/^qwen3_5-9b/q9b/')
case $GAME in team) NS_FULL=5; PER=6; EVS=(team par) ;; seq) NS_FULL=6; PER=5; EVS=(seq par) ;; par) NS_FULL=6; PER=5; EVS=(par team) ;; esac
if [ -n "${SMOKE:-}" ]; then
  # batch and critic batch must divide by DP 2: team 2 x 1 x 6 = 12 (critic 6), seq/par 2 x 2 x 5 = 20 (critic 10)
  R=fcs_oracle_smoke_${GAME}_${REWARD}_$TAG; NR=${NUM_ROLLOUT:-2}; RB=2; NS=$([ "$GAME" = team ] && echo 1 || echo 2); NVAL=16
  BUDGET=${BUDGET:-4096}; PLAN=1024; CO=1; WU=1; EV=${EVAL_EVERY:-$NR}; EN=${N_EVAL:-1}; CGBS=$((RB * NS * PER / 2))
  QOS=${QOS:-pli-short}; TIME=${TIME:-01:30:00}
  extra=(--skip-eval-before-train)
else
  R=fcs_oracle_${GAME}_${REWARD}_${TAG}_s$SEED; NR=${NUM_ROLLOUT:-60}; RB=16; NS=$NS_FULL; NVAL=172
  BUDGET=${BUDGET:-32768}; PLAN=8192; CO=10; WU=6; EV=${EVAL_EVERY:-20}; EN=${N_EVAL:-2}; CGBS=120
  QOS=${QOS:-pli-short}; TIME=${TIME:-24:00:00}
  extra=(--use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo
         --wandb-group fcs_oracle --disable-wandb-random-suffix)
  [ "$GAME" = team ] && extra+=(--skip-eval-before-train)   # team step 0: the producer's eval
fi
VPR=fcs_vp_oracle_${GAME}_${TAG}_s$SEED; [ -n "${SMOKE:-}" ] && VPR=fcs_vp_oracle_smoke_${GAME}_$TAG
if [ -n "${PRETRAIN:-}" ]; then  # the producer: CO critic-only rollouts, dumped; step-0 eval
  R=$VPR; NR=$CO; EV=1000000
  extra=(--save-debug-rollout-data $B/runs/$R/rollout_data/{rollout_id}.pt)
  [ -n "${SMOKE:-}" ] || extra+=(--use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo
                                 --wandb-group fcs_oracle --disable-wandb-random-suffix)
elif [ -n "${VP:-}" ]; then
  extra+=(--replay-rollout-data $B/runs/$VPR/rollout_data/{rollout_id}.pt --replay-rollout-until $CO)
  case " ${extra[*]} " in *" --skip-eval-before-train "*) ;; *) extra+=(--skip-eval-before-train) ;; esac
fi
for g in "${EVS[@]}"; do [ -f $D/fcs_val${NVAL}_fo_$g.jsonl ] || { echo "missing data/fcs_val${NVAL}_fo_$g.jsonl: python3 tools/make_fcs_team_data.py" >&2; exit 1; }; done
evd=(); for g in "${EVS[@]}"; do evd+=(fo_$g $D/fcs_val${NVAL}_fo_$g.jsonl); done
GBS=$((RB * NS * PER))
# longest sample: the lead's final turn or round 5 (problem <= 8192 + 4 programs of <= 8000 chars with feedback) + BUDGET
MAXTOK=$((BUDGET + 28672))
fo=(MA_FO_GAME=$GAME MA_FO_REWARD=$REWARD MA_FO_SUBS=4 MA_FO_ROUNDS=5 MA_FO_BUDGET=$BUDGET MA_FO_PLAN_BUDGET=$PLAN
    MA_FO_MAX_LEN=$MAXTOK MA_FO_TRACE_DIR=$B/runs/$R/traces MA_FO_TRACE_EVERY=32)
envs=(FCS_RM_CONCURRENCY=8 FCS_CASE_WORKERS=8 MILES_SRC=$MILES_SRC RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.5-9B
      "${fo[@]}")
CK=$B/runs/$R/ckpt
args=(--hf-checkpoint $SG --megatron-hf-checkpoint $MG --megatron-to-hf-mode bridge
  --ref-load $MG --load $CK --save $CK --critic-load ${CK}_critic --critic-save ${CK}_critic --save-interval $EV
  --custom-megatron-post-save-hook-path miles_team.ckpt_rotate.keep_latest
  --prompt-data $D/fcs_train200_team.jsonl --input-key prompt --label-key label --apply-chat-template --rollout-shuffle
  --custom-generate-function-path miles_team.fcs_oracle.generate
  --custom-rollout-log-function-path miles_team.fcs_oracle.log_rollout
  --custom-eval-rollout-log-function-path miles_team.fcs_oracle.log_eval
  --custom-reward-post-process-path miles_team.fcs_oracle.post_process
  --num-rollout $NR --rollout-batch-size $RB --n-samples-per-prompt $NS --global-batch-size $GBS
  --rollout-max-prompt-len 8192 --rollout-max-response-len $BUDGET --rollout-max-context-len $MAXTOK
  --rollout-temperature 1.0 --rollout-top-p 1.0
  --eval-prompt-data "${evd[@]}" --eval-interval $EV --n-samples-per-eval-prompt $EN
  --eval-temperature 1.0 --eval-top-p 1.0 --eval-max-prompt-len 8192 --eval-max-response-len $BUDGET
  --eval-max-context-len $MAXTOK "${extra[@]}"
  # EasyPPO algorithm (submit_fcs.sh)
  --advantage-estimator ppo --gamma 1.0 --lambd 1.0 --normalize-advantages
  --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 3.0 --calculate-per-token-loss
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0
  --num-critic-only-steps $CO --critic-global-batch-size $CGBS
  --critic-variance-weighted-loss --critic-variance-weight-beta 0.5 --critic-variance-weight-min 0.25
  --actor-only-overlong-filter --value-clip 0.2 --value-loss-scale 0.5
  --optimizer adam --lr 1e-6 --critic-lr 2e-6 --lr-decay-style constant --lr-warmup-iters $WU
  --lr-decay-iters $NR
  --critic-lr-warmup-iters $((WU * GBS / CGBS)) --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
  # 8x H100, colocated Megatron TP4 x DP2 (TP2 OOMed at the first actor step, s42 14869367) + 8 SGLang engines
  --tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-distributed-optimizer --balance-data
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format bshd
  --micro-batch-size 1 --max-tokens-per-gpu $MAXTOK
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-dtype bfloat16
  --attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate --seed $SEED --rollout-seed $SEED)
sb=(); [ -n "${NICE:-}" ] && sb=(--nice="$NICE")
dep=(); [ -n "${DEP:-}" ] && dep+=(afterany:"$DEP"); [ -n "${VP_DEP:-}" ] && dep+=(afterok:"$VP_DEP")
[ ${#dep[@]} -gt 0 ] && sb+=(--dependency=$(IFS=,; echo "${dep[*]}"))
[ -n "${VP:-}" ] && [ -z "${VP_DEP:-}" ] && [ ! -f $B/runs/$VPR/rollout_data/$((CO - 1)).pt ] && {
  echo "VP=1: $VPR has no rollout $((CO - 1)) dump yet; pass VP_DEP=<its jid>" >&2; exit 1; }
echo "$R: ${fo[*]} rb=$RB ns=$NS gbs=$GBS cgbs=$CGBS maxtok=$MAXTOK evals=${EVS[*]} miles=$(git -C $MILES_SRC rev-parse --short HEAD) repo=$(git -C $B rev-parse --short HEAD)"
[ -n "${DRY:-}" ] && { echo "  ${args[*]}"; exit 0; }
mkdir -p $B/runs/$R
unset $(compgen -e | grep '^MA_' || true)   # only this run's MA_* reach the job
jid=$(env "${envs[@]}" sbatch --parsable --partition=pli-c --account=group --qos=$QOS --time=$TIME "${sb[@]}" \
  --gres=gpu:8 --cpus-per-task=64 --mem=640G --job-name="miles-$R" $B/sbatch_miles.sh "${args[@]}")
echo "  submitted: job $jid"
