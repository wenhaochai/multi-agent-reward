#!/bin/bash
# Frontier-CS multi-agent reward screen (labs-molt-docs docs/fcs_multiagent_reward.md): game G1 (miles_team/fcs_team.py)
# under EasyPPO on miles, Qwen3.5-9B EasyPPO SFT init, text-only (MODEL-text for Megatron, MODEL-lm for sglang).
# The EasyPPO flags are submit_fcs.sh's verbatim; what differs is the game, the batch shape, the evals and the screen.
#
# ARM: shared | indiv | mix | unique | adopt | synth  (MA_FT_REWARD; ALPHA for mix/synth, default 0.5; BETA for unique,
#      default 0) | solocm (single agent through the same code path, MA_FT_MATES=0)
# Batch: team = 16 prompts x 8 episodes x 4 agents = 512 samples per step; solocm = 16 x 32 x 1 = 512 samples. Every
#   agent turn may generate up to BUDGET (32768) tokens, so both take at most 512 x 32768 generated tokens per step
#   and as many samples: solocm is the solo EasyPPO run's shape (16 x 32, 32k) - the compute-matched baseline.
#   (The lead also reads up to 3 programs: its prompt is longer; that prefill is the team's only extra cost.)
# Screen: NUM_ROLLOUT 60 (EasyPPO's 200 scaled: 10 critic-only rollouts, 6-rollout lr warmup), val every 20 on
#   fcs_val (the game, lead's final, reward S) and fcs_val_solo (the same weights as one agent), N_EVAL samples each
#   (default 4: 172 x 4). solocm also plays the game at eval (MA_FT_EVAL_MATES=3). The step-0 policy is the same for
#   every arm, so only ARM=shared and ARM=solocm evaluate before training.
# Usage: ARM=... [SEED=42] [NUM_ROLLOUT=60] [EVAL_EVERY=20] [N_EVAL=4] [BUDGET=32768] [MODEL=<hf dir>] [SMOKE=1]
#        [QOS] [TIME] [DEP] [NICE] [DRY=1] bash submit_fcs_team.sh
#   SMOKE=1: 3 rollouts of 2 prompts x 2 episodes, 4096-token turns, 16-problem val sets at the end, pli-cp.
#   Plain sbatch only (user 2026-10-02: no queue workers); runs longer than TIME chain with DEP (afterany), each
#   segment resuming from --load = --save.
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
D=$B/data
MILES_SRC=/scratch/gpfs/GROUP/USER/project/miles-easyppo
MODEL=${MODEL:-$W/models/Qwen3.5-9B-FCS-SFT}
MG=$MODEL-text; SG=$MODEL-lm
for d in $MG $SG; do [ -f $d/config.json ] || { echo "missing $d: run tools/make_qwen35_text_ckpts.py $MODEL" >&2; exit 1; }; done
for f in fcs_train200_team fcs_val172_team fcs_val172_solo fcs_val16_team fcs_val16_solo; do
  [ -f $D/$f.jsonl ] || { echo "missing data/$f.jsonl: python3 tools/make_fcs_team_data.py" >&2; exit 1; }; done
: "${ARM:?ARM=shared|indiv|mix|unique|adopt|synth|solocm}"
SEED=${SEED:-42}; ALPHA=${ALPHA:-0.5}; BETA=${BETA:-0}
TAG=$(basename $MODEL | tr 'A-Z.' 'a-z_' | sed 's/^qwen3_5-9b/q9b/')
case $ARM in
  shared|indiv|adopt) ft=(MA_FT_MATES=3 MA_FT_REWARD=$ARM); AN=$ARM ;;
  mix|synth) ft=(MA_FT_MATES=3 MA_FT_REWARD=$ARM MA_FT_ALPHA=$ALPHA); AN=$ARM$(python3 -c "print(round($ALPHA*100))") ;;
  unique) ft=(MA_FT_MATES=3 MA_FT_REWARD=unique MA_FT_BETA=$BETA); AN=unique$(python3 -c "print(round($BETA*100))") ;;
  solocm) ft=(MA_FT_MATES=0 MA_FT_EVAL_MATES=3 MA_FT_REWARD=shared); AN=solocm ;;
  *) echo "unknown ARM $ARM" >&2; exit 1 ;;
esac
if [ "$ARM" = solocm ]; then NS_TEAM=32; PER=1; else NS_TEAM=8; PER=4; fi
if [ -n "${SMOKE:-}" ]; then
  R=fcs_team_smoke_${AN}_$TAG; NR=${NUM_ROLLOUT:-3}; RB=2; NS=$([ "$ARM" = solocm ] && echo 8 || echo 2)
  BUDGET=${BUDGET:-4096}; CO=1; WU=1; EV=${EVAL_EVERY:-$NR}; EN=${N_EVAL:-1}; CGBS=8
  VT=$D/fcs_val16_team.jsonl; VS=$D/fcs_val16_solo.jsonl; QOS=${QOS:-pli-cp}; TIME=${TIME:-01:30:00}
  extra=(--skip-eval-before-train)
else
  R=fcs_team_${AN}_${TAG}_s$SEED; NR=${NUM_ROLLOUT:-60}; RB=16; NS=$NS_TEAM
  BUDGET=${BUDGET:-32768}; CO=10; WU=6; EV=${EVAL_EVERY:-20}; EN=${N_EVAL:-4}; CGBS=128
  VT=$D/fcs_val172_team.jsonl; VS=$D/fcs_val172_solo.jsonl; QOS=${QOS:-pli-short}; TIME=${TIME:-24:00:00}
  extra=(--use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo
         --wandb-group fcs_team --disable-wandb-random-suffix)
  case $ARM in shared|solocm) ;; *) extra+=(--skip-eval-before-train) ;; esac
fi
GBS=$((RB * NS * PER))
ft+=(MA_FT_MATE_BUDGET=$BUDGET MA_FT_LEAD_BUDGET=$BUDGET MA_FT_MAX_LEN=65536 MA_FT_TRACE_DIR=$B/runs/$R/traces
     MA_FT_TRACE_EVERY=32)
envs=(FCS_RM_CONCURRENCY=8 FCS_CASE_WORKERS=8 MILES_SRC=$MILES_SRC RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.5-9B
      "${ft[@]}")
CK=$B/runs/$R/ckpt
# longest training sample: the lead's prompt (problem <= 8192 + 3 programs of <= 12000 chars) + BUDGET
MAXTOK=$((BUDGET + 24576))
args=(--hf-checkpoint $SG --megatron-hf-checkpoint $MG --megatron-to-hf-mode bridge
  --ref-load $MG --load $CK --save $CK --critic-load ${CK}_critic --critic-save ${CK}_critic --save-interval $EV
  --custom-megatron-post-save-hook-path miles_team.ckpt_rotate.keep_latest
  --prompt-data $D/fcs_train200_team.jsonl --input-key prompt --label-key label --apply-chat-template --rollout-shuffle
  --custom-generate-function-path miles_team.fcs_team.generate
  --custom-rollout-log-function-path miles_team.fcs_team.log_rollout
  --custom-eval-rollout-log-function-path miles_team.fcs_team.log_eval
  --num-rollout $NR --rollout-batch-size $RB --n-samples-per-prompt $NS --global-batch-size $GBS
  --rollout-max-prompt-len 8192 --rollout-max-response-len $BUDGET --rollout-max-context-len $MAXTOK
  --rollout-temperature 1.0 --rollout-top-p 1.0
  --eval-prompt-data fcs_val $VT fcs_val_solo $VS --eval-interval $EV --n-samples-per-eval-prompt $EN
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
  # one actor step per rollout; miles' default train_iters counts RB*NS samples, not the 1+K per episode (0 in the smoke)
  --lr-decay-iters $NR
  --critic-lr-warmup-iters $((WU * GBS / CGBS)) --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
  # parallelism: 8x H100, colocated Megatron TP2 x DP4 (actor and critic in turn) + 8 SGLang engines
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-distributed-optimizer --balance-data
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format bshd
  --micro-batch-size 1 --max-tokens-per-gpu $MAXTOK
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-dtype bfloat16
  --attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate --seed $SEED
  # --seed sets Megatron and the SGLang engines (engine i gets SEED + i, so replicate seeds sit >= 8 apart); the prompt
  # order comes from --rollout-seed (miles default 42), so a replicate changes it too
  --rollout-seed $SEED)
sb=(); [ -n "${NICE:-}" ] && sb=(--nice="$NICE")
[ -n "${DEP:-}" ] && sb+=(--dependency=afterany:"$DEP")
echo "$R: arm=$ARM ${ft[*]} rb=$RB ns=$NS gbs=$GBS miles=$(git -C $MILES_SRC rev-parse --short HEAD) repo=$(git -C $B rev-parse --short HEAD)"
[ -n "${DRY:-}" ] && { echo "  ${args[*]}"; exit 0; }
mkdir -p $B/runs/$R
unset $(compgen -e | grep '^MA_' || true)   # only this arm's MA_* reach the job
jid=$(env "${envs[@]}" sbatch --parsable --partition=pli-c --account=group --qos=$QOS --time=$TIME "${sb[@]}" \
  --gres=gpu:8 --cpus-per-task=64 --mem=640G --job-name="miles-$R" $B/sbatch_miles.sh "${args[@]}")
echo "  submitted: job $jid"
