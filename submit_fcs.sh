#!/bin/bash
# EasyPPO (arXiv 2609.36802) on Frontier-CS with Qwen3.5-9B, full-parameter, on miles (logs/fcs_easyppo.yaml in
# labs-molt). Code: miles branch `easyppo` (worktree miles-easyppo; patches/easyppo here), which adds EasyPPO's three
# changes as flags; parity with EasyPPO's own functions: tests/test_easyppo_parity.py.
#   data: 200 FrontierSmith synthetic problems (train), the 172-problem Frontier-CS public set (val, 5 samples, T 1.0,
#     top-p 1.0); prompts FrontierSmith's; default chat template (thinking on); reward = local judge score / 100
#     (miles_team/fcs_rm.py).
#   EasyPPO paper, Frontier-CS: 16 prompts x 32 samples per step, 32768 response tokens, T 1.0, actor lr 1e-6,
#     critic lr 2e-6, 200 updates (NUM_ROLLOUT; here 200 rollouts including the 30 critic-only ones).
#   EasyPPO configs/easyppo_aime.yaml algorithm: GAE gamma = lambda = 1 with batch advantage whitening; clip 0.2/0.28,
#     dual clip 3.0; token-mean losses; low_var_kl loss 0.001, no KL in reward, no entropy bonus; Adam (0.9, 0.999),
#     weight decay 0.01, grad clip 1.0, constant lr with 20 warmup steps; 30 critic-only steps; value clip 0.2, 0.5 x
#     value loss; critic weights 1/max(std of the prompt's rewards, 0.25) (beta 0.5); four critic mini-batches per
#     step; truncated responses masked from the actor only.
#   miles units: --lr-warmup-iters counts optimizer steps of the trainer's own global batch, so the actor (one step per
#     rollout) takes 20 and the critic (four steps per rollout) 80; both warm up over 20 rollouts as in EasyPPO.
# Usage: [SMOKE=1] [SEED=42] [NUM_ROLLOUT] [EVAL_EVERY] [QOS] [TIME] [NICE] [DEP=<jid>] [MODEL=<hf dir>] [DRY=1] bash submit_fcs.sh
#   SMOKE=1: 3 rollouts of 4 prompts x 8 samples (rollout 0 critic-only), 1-rollout lr warmup (Megatron asserts
#     warmup < total steps; smoke 14850360 died on 20 > 3), 4096-token answers, then a val eval
#     (1 sample each), on pli-cp, ~1 h: checks the critic + actor + judge pipeline, not 32k memory.
#   Full: --load = --save (actor) and the critic's own dir, so the same command resumes after a 24 h wall; chain
#     segments with DEP. wandb offline (project fcs_easyppo), synced from a vis node.
set -euo pipefail
B=/scratch/gpfs/GROUP/USER/project/miles-q38-build
W=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace
D=$B/data
MILES_SRC=/scratch/gpfs/GROUP/USER/project/miles-easyppo
MODEL=${MODEL:-$W/models/Qwen3.5-9B}
# text-only: no vision tower anywhere (tools/make_qwen35_text_ckpts.py MODEL builds both). Megatron trains
# MODEL-text (Qwen3_5ForCausalLM -> GPTModel); sglang serves MODEL-lm (language_model_only, no vision weights).
MG=$MODEL-text; SG=$MODEL-lm
for d in $MG $SG; do [ -f $d/config.json ] || { echo "missing $d: run tools/make_qwen35_text_ckpts.py $MODEL" >&2; exit 1; }; done
SEED=${SEED:-42}
# the run name carries the init (--load = --save resumes): a base-model run's checkpoint must never seed an SFT run
TAG=$(basename $MODEL | tr 'A-Z.' 'a-z_' | sed 's/^qwen3_5-9b/q9b/')
if [ -n "${SMOKE:-}" ]; then
  R=fcs_easyppo_smoke_$TAG; NR=${NUM_ROLLOUT:-3}; RB=4; NS=8; GBS=32; CGBS=8; LEN=4096; CO=1; WU=1; EV=${EVAL_EVERY:-$NR}; EN=1
  QOS=${QOS:-pli-cp}; TIME=${TIME:-01:30:00}
  extra=(--skip-eval-before-train)
else
  R=fcs_easyppo_${TAG}_s$SEED; NR=${NUM_ROLLOUT:-200}; RB=16; NS=32; GBS=512; CGBS=128; LEN=32768; CO=30; WU=20
  EV=${EVAL_EVERY:-10}; EN=5; QOS=${QOS:-pli-short}; TIME=${TIME:-24:00:00}
  extra=(--use-wandb --wandb-mode offline --wandb-dir $B/runs/$R --wandb-project fcs_easyppo
         --wandb-group easyppo --disable-wandb-random-suffix)
fi
CK=$B/runs/$R/ckpt
args=(--hf-checkpoint $SG --megatron-hf-checkpoint $MG --megatron-to-hf-mode bridge
  --ref-load $MG --load $CK --save $CK --critic-load ${CK}_critic --critic-save ${CK}_critic --save-interval $EV
  --custom-megatron-post-save-hook-path miles_team.ckpt_rotate.keep_latest
  --prompt-data $D/fcs_train200.jsonl --input-key prompt --label-key label --apply-chat-template --rollout-shuffle
  --custom-rm-path miles_team.fcs_rm.fcs_rm
  --num-rollout $NR --rollout-batch-size $RB --n-samples-per-prompt $NS --global-batch-size $GBS
  --rollout-max-prompt-len 8192 --rollout-max-response-len $LEN --rollout-max-context-len $((LEN + 8192))
  --rollout-temperature 1.0 --rollout-top-p 1.0
  --eval-prompt-data fcs_val $D/fcs_val172.jsonl --eval-interval $EV --n-samples-per-eval-prompt $EN
  --eval-temperature 1.0 --eval-top-p 1.0 --eval-max-prompt-len 8192 --eval-max-response-len $LEN
  --eval-max-context-len $((LEN + 8192)) "${extra[@]}"
  # EasyPPO algorithm
  --advantage-estimator ppo --gamma 1.0 --lambd 1.0 --normalize-advantages
  --eps-clip 0.2 --eps-clip-high 0.28 --eps-clip-c 3.0 --calculate-per-token-loss
  --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl --kl-coef 0 --entropy-coef 0
  --num-critic-only-steps $CO --critic-global-batch-size $CGBS
  --critic-variance-weighted-loss --critic-variance-weight-beta 0.5 --critic-variance-weight-min 0.25
  --actor-only-overlong-filter --value-clip 0.2 --value-loss-scale 0.5
  --optimizer adam --lr 1e-6 --critic-lr 2e-6 --lr-decay-style constant --lr-warmup-iters $WU
  --critic-lr-warmup-iters $((WU * GBS / CGBS)) --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.999 --clip-grad 1.0
  # parallelism: 8x H100, colocated Megatron TP2 x DP4 (actor and critic in turn) + 8 SGLang engines
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1
  --use-distributed-optimizer --balance-data
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --qkv-format bshd
  --micro-batch-size 1 --max-tokens-per-gpu $((LEN + 8192))
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-dtype bfloat16
  --attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate --seed $SEED
  # --seed sets Megatron and the SGLang engines (engine i gets SEED + i, so replicate seeds sit >= 8 apart); the prompt
  # order comes from --rollout-seed (miles default 42), so a replicate changes it too
  --rollout-seed $SEED)
sb=(); [ -n "${NICE:-}" ] && sb=(--nice="$NICE")
[ -n "${DEP:-}" ] && sb+=(--dependency=afterany:"$DEP")
echo "$R: miles=$(git -C $MILES_SRC rev-parse --short HEAD) repo=$(git -C $B rev-parse --short HEAD) qos=$QOS time=$TIME"
[ -n "${DRY:-}" ] && { echo "  ${args[*]}"; exit 0; }
mkdir -p $B/runs/$R
jid=$(env FCS_RM_CONCURRENCY=8 FCS_CASE_WORKERS=8 MILES_SRC=$MILES_SRC RUN_NAME=$R GPUS=8 MILES_MODEL_TYPE=qwen3.5-9B \
  sbatch --parsable --partition=pli-c --account=group --qos=$QOS --time=$TIME "${sb[@]}" --gres=gpu:8 \
  --cpus-per-task=64 --mem=640G --job-name="miles-$R" $B/sbatch_miles.sh "${args[@]}")
echo "  submitted: job $jid"
