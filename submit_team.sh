#!/bin/bash
# The labs-molt Qwen3.8-27B team game (logs/peer_team_lead.yaml in labs-molt) on miles: one arm per job, 8x H100,
# colocated Megatron (TP4 x DP2, LoRA r32/a32 through the bridge) + 2 SGLang engines x TP4 serving the adapter.
# The game, prompts, knobs and rewards are molt's (miles_team/team_core.py, a verbatim copy checked by
# tests/test_core_parity.py); the advantage is molt's FlashREINFORCE (reward minus the batch mean, one sample per
# prompt); eval is per-episode task accuracy on the same 153-problem set (T 1.0, 4 samples).
# Usage: ARM=lead|nob|solo [SMOKE=1] [SEED=42] [NUM_ROLLOUT] [EVAL_EVERY] [QOS] [TIME] [DRY=1] bash submit_team.sh
#   SMOKE=1: 2 rollouts x 8 episodes, then an aime_2024 eval (1 sample each), on pli-cp.
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
D=$B/data
: "${ARM:?ARM=lead|nob|solo}"
SEED=${SEED:-42}
case $ARM in
  lead) team=(MA_TM_MATES=3 MA_TM_COLLAB_BONUS=0.1) ;;
  nob)  team=(MA_TM_MATES=3 MA_TM_COLLAB_BONUS=0) ;;
  solo) team=(MA_TM_MATES=0 MA_TM_COLLAB_BONUS=0) ;;
  *) echo "ARM must be lead, nob or solo" >&2; exit 1 ;;
esac
if [ -n "${SMOKE:-}" ]; then
  R=miles_q38_smoke_team_$ARM; NR=${NUM_ROLLOUT:-2}; RB=8; EV=${EVAL_EVERY:-$NR}; EN=1
  EVAL=(aime_2024 $D/eval_aime_2024.jsonl); QOS=${QOS:-pli-cp}; TIME=${TIME:-02:30:00}
else
  R=miles_q38_team_${ARM}_s$SEED; NR=${NUM_ROLLOUT:-200}; RB=32; EV=${EVAL_EVERY:-50}; EN=4
  EVAL=(); for c in aime_2024 aime_2025 aime_2026 hmmt_feb_2025 hmmt_feb_2026; do EVAL+=($c $D/eval_$c.jsonl); done
  QOS=${QOS:-pli-short}; TIME=${TIME:-24:00:00}
fi
ma=("${team[@]}" MA_TM_LEAD_TURNS=4 MA_TM_MATE_TURNS=2 MA_TM_LEAD_THINK=0 MA_TM_MATE_THINK=0 MA_TM_LEAD_BUDGET=12288
    MA_TM_MATE_BUDGET=12288 MA_TM_MAX_LEN=32768 MA_TM_TRACE_DIR=$B/runs/$R/traces MA_TM_TRACE_EVERY=${TRACE_EVERY:-16})
args=(--hf-checkpoint $W/models/Qwen3.8-27B --megatron-to-hf-mode bridge
  --lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 --target-modules all-linear --no-gradient-accumulation-fusion
  --lora-base-cpu-backup
  --prompt-data $D/q38mid_train.jsonl --input-key prompt --label-key label --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}' --rollout-shuffle
  --custom-generate-function-path miles_team.team_rollout.generate
  --custom-reward-post-process-path miles_team.reward_post.flash_reinforce
  --num-rollout $NR --rollout-batch-size $RB --n-samples-per-prompt 1 --global-batch-size $RB
  --rollout-max-response-len 12288 --rollout-max-context-len 32768 --rollout-temperature 1.0 --rollout-top-p 1.0
  --eval-prompt-data "${EVAL[@]}" --eval-interval $EV --n-samples-per-eval-prompt $EN --eval-temperature 1.0
  --eval-top-p 1.0 --eval-max-response-len 12288 --eval-max-context-len 32768 --skip-eval-before-train
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format bshd
  --micro-batch-size 1 --max-tokens-per-gpu 32768
  --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.6 --sglang-dtype bfloat16
  --sglang-max-lora-rank 32 --sglang-lora-backend triton
  --save $B/runs/$R/ckpt --save-interval $EV
  --attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate --seed $SEED)
echo "$R: miles=$(git -C /scratch/gpfs/GROUP/USER/project/miles-q38 rev-parse --short HEAD) repo=$(git -C $B rev-parse --short HEAD) qos=$QOS time=$TIME"
echo "  ${ma[*]}"
[ -n "${DRY:-}" ] && { echo "  ${args[*]}"; exit 0; }
unset $(compgen -e | grep '^MA_' || true)
jid=$(env "${ma[@]}" RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.8-27B sbatch --parsable --partition=pli-c --account=group \
  --qos=$QOS --time=$TIME --gres=gpu:8 --cpus-per-task=64 --mem=640G --job-name="miles-$R" $B/sbatch_miles.sh "${args[@]}")
echo "  submitted: job $jid"
