#!/bin/bash
# miles smoke 1: single-agent GRPO LoRA on Qwen3.8-27B (8x H100, pli-cp, colocated Megatron TP4 x DP2 + 2 SGLang
# engines x TP4), the molt q38 data (prep_q38mid, 12288-token answers, T 1.0, non-thinking) and molt's grader.
# 3 rollouts x 8 prompts x 4 samples, then an aime_2024 eval (1 sample each). Checks: bridge loads the HF weights,
# LoRA trains, adapters reach SGLang, GDN runs in both engines, the reward is non-trivial.
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
D=$B/data
R=${RUN_NAME:-miles_q38_smoke_single}
args=(--hf-checkpoint /scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/Qwen3.8-27B --megatron-to-hf-mode bridge
  --lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 --target-modules all-linear --no-gradient-accumulation-fusion
  --lora-base-cpu-backup
  --prompt-data $D/q38mid_train.jsonl --input-key prompt --label-key label --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}' --rollout-shuffle --custom-rm-path miles_team.rm.molt_math_rm
  --num-rollout ${NUM_ROLLOUT:-3} --rollout-batch-size 8 --n-samples-per-prompt 4 --rollout-max-response-len 12288
  --rollout-temperature 1.0 --global-batch-size 32
  --eval-prompt-data aime_2024 $D/eval_aime_2024.jsonl --eval-interval ${NUM_ROLLOUT:-3} --n-samples-per-eval-prompt 1
  --eval-temperature 1.0 --eval-top-p 1.0 --eval-max-response-len 12288 --skip-eval-before-train
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format bshd
  --micro-batch-size 1 --max-tokens-per-gpu 16384
  --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.6 --sglang-dtype bfloat16
  --sglang-max-lora-rank 32 --sglang-lora-backend triton
  --save $B/runs/$R/ckpt --save-interval ${NUM_ROLLOUT:-3}
  --attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate --seed 42)
[ -n "${DRY:-}" ] && { echo "${args[*]}"; exit 0; }
jid=$(RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.8-27B sbatch --parsable --partition=pli-c --account=group --qos=pli-cp \
  --time=02:00:00 --gres=gpu:8 --cpus-per-task=64 --mem=640G --job-name="miles-$R" $B/sbatch_miles.sh "${args[@]}")
echo "$R: job $jid"
