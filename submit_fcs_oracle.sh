#!/bin/bash
# Frontier-CS games with an oracle (labs-molt-docs docs/fcs_multiagent_reward.md; miles_team/fcs_oracle.py) under
# EasyPPO on miles, Qwen3.5-9B EasyPPO SFT init, text-only. The EasyPPO flags are submit_fcs.sh's verbatim; what differs
# is the game, the batch shape and the evals. Every judged program returns its total score, per-test-case scores and
# compiler errors; a reply's judged program is its LAST ```cpp block.
#
# GAME=team REWARD=shared|bonus|diff  1 lead (plans 4 tasks, then submits the final) + 4 helper subagents (each may test
#                                     up to SUB_TESTS=3 programs, sees each result, then reports); outcome = the lead's
#                                     final submission S. shared: all S; bonus: subagent S + 0.5 x its best test score;
#                                     diff: subagent S - S_-j (counterfactual final without its work)
# GAME=seq  REWARD=shared|indiv|diff  one agent, 5 rounds of revision with feedback; outcome = the last round
# GAME=par  REWARD=shared|indiv|diff  5 independent attempts (indiv = single-agent RL); outcome = the best attempt
# Batch: team 16 prompts x 5 episodes x 18 sample slots (plan, final, 4 subagents x 4 turns; unused slots are masked
#   pads) = 1440; seq/par 16 x 6 x 5 = 480. Critic batch = batch / 4 (EasyPPO: 4 critic mini-batches).
# Screen: NUM_ROLLOUT 60 (10 critic-only, no lr warmup), val every 20 on two oracle-game sets of 172 problems x
#   N_EVAL (2) episodes: team arms play team + par (the same weights as 5 independent attempts), seq plays seq + par,
#   par plays par + team. Step 0 is evaluated once per game (team and par sets: the team producer; seq: its run).
# Shared value pretraining: PRETRAIN=1 runs the 10 critic-only rollouts of GAME once (runs/fcs_vp_oracle_<game>_...),
#   dumped; VP=1 VP_DEP=<its jid> makes an arm replay them, relabeled with its reward (fcs_oracle.post_process).
# Usage: GAME=... REWARD=... [SUB_TESTS=3] [SEED=42] [RUN_TAG=<suffix for a separate run dir>] [NUM_ROLLOUT] [EVAL_EVERY] [N_EVAL] [BUDGET] [MODEL] [SMOKE=1] [PRETRAIN=1]
#        [VP=1 VP_DEP=jid] [QOS] [TIME] [DEP] [NICE] [MILES_SRC=<worktree>] [EXTRA_ARGS="..."] [DRY=1] bash submit_fcs_oracle.sh
#   SMOKE=1: 2 rollouts of 2 prompts x 1 episode, 4096-token turns, 16-problem val sets of the game at the end.
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
D=$B/data
MILES_SRC=${MILES_SRC:-/scratch/gpfs/GROUP/USER/project/miles-easyppo}  # the miles worktree the job imports
MODEL=${MODEL:-$W/models/Qwen3.5-9B-FCS-SFT}
MG=$MODEL-text; SG=$MODEL-lm
for d in $MG $SG; do [ -f $d/config.json ] || { echo "missing $d: run tools/make_qwen35_text_ckpts.py $MODEL" >&2; exit 1; }; done
: "${GAME:?GAME=team|seq|par}" "${REWARD:?REWARD=shared|bonus|diff (team), shared|indiv|diff (seq, par)}"
case "$GAME:$REWARD" in team:shared|team:bonus|team:diff|seq:shared|seq:indiv|seq:diff|par:shared|par:indiv|par:diff) ;;
  *) echo "bad GAME:REWARD $GAME:$REWARD" >&2; exit 1 ;; esac
SEED=${SEED:-42}
TAG=$(basename $MODEL | tr 'A-Z.' 'a-z_' | sed 's/^qwen3_5-9b/q9b/')
# team: plan + final + 4 subagents x (3 tests + a report turn) = 18 sample slots per episode (unused ones are masked pads)
SUB_TESTS=${SUB_TESTS:-3}
case $GAME in team) NS_FULL=5; PER=$((2 + 4 * (SUB_TESTS + 1))); EVS=(team par) ;; seq) NS_FULL=6; PER=5; EVS=(seq par) ;; par) NS_FULL=6; PER=5; EVS=(par team) ;; esac
if [ -n "${SMOKE:-}" ]; then
  # batch and critic batch must divide by DP 2: team 2 x 1 x 18 = 36 (critic 18), seq/par 2 x 2 x 5 = 20 (critic 10)
  R=fcs_oracle_smoke_${GAME}_${REWARD}_$TAG${RUN_TAG:+_$RUN_TAG}; NR=${NUM_ROLLOUT:-2}; RB=2; NS=$([ "$GAME" = team ] && echo 1 || echo 2); NVAL=16
  BUDGET=${BUDGET:-4096}; PLAN=2048; CO=1; WU=0; EV=${EVAL_EVERY:-$NR}; SAVE=$NR; EN=${N_EVAL:-1}; CGBS=$((RB * NS * PER / 2))
  QOS=${QOS:-pli-short}; TIME=${TIME:-01:30:00}
  extra=(--skip-eval-before-train)
else
  R=fcs_oracle_${GAME}_${REWARD}_${TAG}_s$SEED${RUN_TAG:+_$RUN_TAG}; NR=${NUM_ROLLOUT:-60}; RB=16; NS=$NS_FULL; NVAL=172
  # PLAN 16384: the plan turn thinks first (the SFT model averages ~16.5k tokens per solve), and a cut plan gives
  # every subagent NO_TASK and is masked from the actor (audit 2026-10-04; 8192 at first)
  # SAVE 5: a 24 h wall throws away everything since the last save (a team rollout ~36 min, an eval ~2 h)
  BUDGET=${BUDGET:-32768}; PLAN=16384; CO=10; WU=0; EV=${EVAL_EVERY:-20}; SAVE=${SAVE_EVERY:-5}; EN=${N_EVAL:-2}
  CGBS=$((RB * NS * PER / 4))  # four critic steps per rollout
  QOS=${QOS:-pli-short}; TIME=${TIME:-24:00:00}
  extra=(--use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo
         --wandb-group $R --wandb-run-id $R --disable-wandb-random-suffix)
  # step 0 once per game: team and par sets in the team producer, the seq set in the seq run
  [ "$GAME" != seq ] && extra+=(--skip-eval-before-train)
fi
# the producer's dumps fix the samples per episode: a SUB_TESTS other than 3 gets its own producer
VT=$([ "$SUB_TESTS" = 3 ] || echo "_t$SUB_TESTS")
VPR=fcs_vp_oracle_${GAME}${VT}_${TAG}_s$SEED; [ -n "${SMOKE:-}" ] && VPR=fcs_vp_oracle_smoke_${GAME}${VT}_$TAG
if [ -n "${PRETRAIN:-}" ]; then  # the producer: CO critic-only rollouts, dumped; step-0 eval
  # NR = CO + 1 with an exit after CO rollouts: miles forces an eval and a (104 GB critic) save on the last rollout
  # (should_run_periodic_action), which would repeat the step-0 eval on the same frozen weights
  R=$VPR; NR=$((CO + 1)); EV=1000000; SAVE=1000000
  extra=(--save-debug-rollout-data $B/runs/$R/rollout_data/{rollout_id}.pt --debug-exit-after-rollout $CO)
  [ -n "${SMOKE:-}" ] || extra+=(--use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo
                                 --wandb-group $R --wandb-run-id $R --disable-wandb-random-suffix)
elif [ -n "${VP:-}" ]; then
  extra+=(--replay-rollout-data $B/runs/$VPR/rollout_data/{rollout_id}.pt --replay-rollout-until $CO)
  case " ${extra[*]} " in *" --skip-eval-before-train "*) ;; *) extra+=(--skip-eval-before-train) ;; esac
fi
for g in "${EVS[@]}"; do [ -f $D/fcs_val${NVAL}_fo_$g.jsonl ] || { echo "missing data/fcs_val${NVAL}_fo_$g.jsonl: python3 tools/make_fcs_team_data.py" >&2; exit 1; }; done
evd=(); for g in "${EVS[@]}"; do evd+=(fo_$g $D/fcs_val${NVAL}_fo_$g.jsonl); done
GBS=$((RB * NS * PER))
# longest sample: the lead's final turn (problem <= 8192 + plan + 4 x (a tested program <= 8000 chars with its result
# and a report <= 6000 chars)) or a subagent's 4th turn, + BUDGET
MAXTOK=$((BUDGET + 32768))
fo=(MA_FO_GAME=$GAME MA_FO_REWARD=$REWARD MA_FO_SUBS=4 MA_FO_ROUNDS=5 MA_FO_BUDGET=$BUDGET MA_FO_PLAN_BUDGET=$PLAN
    MA_FO_SUB_TESTS=$SUB_TESTS MA_FO_CF=$([ -n "${PRETRAIN:-}" ] && echo 1 || echo 0)
    MA_FO_REPORT_BUDGET=$((BUDGET < 16384 ? BUDGET : 16384))
    MA_FO_MAX_LEN=$MAXTOK MA_FO_TRACE_DIR=$B/runs/$R/traces MA_FO_TRACE_EVERY=32)
envs=(FCS_RM_CONCURRENCY=56 FCS_CASE_WORKERS=1 MILES_SRC=$MILES_SRC RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.5-9B
      "${fo[@]}")
CK=$B/runs/$R/ckpt
args=(--hf-checkpoint $SG --megatron-hf-checkpoint $MG --megatron-to-hf-mode bridge
  --ref-load $MG --load $CK --save $CK --critic-load ${CK}_critic --critic-save ${CK}_critic --save-interval $SAVE
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
  # EasyPPO algorithm, Frontier-CS settings of the paper (submit_fcs.sh): upper clip 0.2, floor 0.075, no LR warmup
  --advantage-estimator ppo --gamma 1.0 --lambd 1.0 --normalize-advantages
  --eps-clip 0.2 --eps-clip-high 0.2 --eps-clip-c 3.0 --calculate-per-token-loss
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0
  --num-critic-only-steps $CO --critic-global-batch-size $CGBS
  --critic-variance-weighted-loss --critic-variance-weight-beta 0.5 --critic-variance-weight-min 0.075
  --actor-only-overlong-filter --value-clip 0.2 --value-loss-scale 0.5
  --optimizer adam --lr 1e-6 --critic-lr 2e-6 --lr-decay-style constant --lr-warmup-iters $WU
  # constant LR: decay iters only feed Megatron's assert warmup < decay, in each trainer's own steps (the critic takes
  # GBS/CGBS steps per rollout and warms up for WU*GBS/CGBS of them; --lr-decay-iters NR failed it for short runs)
  --lr-decay-iters $(((NR + WU) * GBS / CGBS + 1))
  --critic-lr-warmup-iters $((WU * GBS / CGBS)) --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
  # 8x H100, colocated Megatron TP4 x DP2 + 8 SGLang engines. TP2 OOMed at the first actor step (s42 14869367): the
  # fused cross-entropy's fp32 buffer for one 20-33k-token sample took 9.4-15.2 GiB per rank (micro-batch 1, so
  # --max-tokens-per-gpu does not bound it); TP4 halves the vocab shard and the per-rank weights/grads/optimizer
  --tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-distributed-optimizer --balance-data
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format bshd
  --micro-batch-size 1 --max-tokens-per-gpu $MAXTOK
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-dtype bfloat16
  --attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate --seed $SEED --rollout-seed $SEED)
sb=(); [ -n "${NICE:-}" ] && sb=(--nice="$NICE")
# EXTRA_ARGS: whitespace-separated flags appended last (argparse: the last occurrence wins), e.g. replay or test flags
[ -n "${EXTRA_ARGS:-}" ] && read -r -a _xa <<< "$EXTRA_ARGS" && args+=("${_xa[@]}")
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
