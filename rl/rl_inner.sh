#!/bin/bash
# Inside miles-v0.1.3.sif with a PID namespace of its own (rl/rl_job.sh): a job-private Ray head on job-derived ports,
# started with the run's environment (Ray's workers, the agent processes among them, take theirs from it), then miles'
# train.py. miles: our fork (branch flint-grpo of wenhaochai/miles-private, worktree $MILES); the training model is
# flint_hf from this repository. Usage: rl_inner.sh smoke|test2|run20|train67|train67_test|train67_smoke|args; env JID,
# OUT, PKG (the built Flint plugin), TMPDIR; train67*: also T67_P, T67_G, T67_SEED, T67_LEVERS (rl/rl_job.sh). Mode args: the argument
# checks alone, off GPU (rl/args_all.sh runs them on della-vis1).
set -uo pipefail
: "${JID:?}" "${OUT:?}" "${PKG:?}" "${TMPDIR:?}"
MODE=$1
REPO=/scratch/gpfs/GROUP/USER/project/mas-rl-elastic13
FLINT=/scratch/gpfs/GROUP/USER/project/labs-molt/_workspace/models/flint
MILES=/scratch/gpfs/GROUP/USER/project/miles-numa5
MEMPROBE_DATA=/scratch/gpfs/GROUP/USER/tmp/flint_s5/memprobe_data   # a recorded batch for the schedule check
export HOME=$TMPDIR/home TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export SGLANG_CACHE_DIR=$TMPDIR/sglang FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer
mkdir -p "$HOME"
export PYTHONPATH=$PKG:$REPO:$MILES PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1   # the container's writable layer is 64 MB
export RAY_USAGE_STATS_ENABLED=0 no_proxy=127.0.0.1,localhost MASTER_ADDR=127.0.0.1
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1 SGLANG_RETURN_ORIGINAL_LOGPROB=0
# the game: one Flint, 128k generated tokens per episode (budget end with forced restart, its defaults); the bash tool
# without strict (see flint/chat.py); the judge counts the user's tasks on the node (rl/rl_job.sh binds /proc there)
export MA_LG_N=1 MA_LG_TOKENS=128000 MA_AG_TOOL_STRICT=0 FCS_PROC=/host/proc
# RL's context cap (NOT the eval's 400k characters and 32k-token turns): a training sample (one TITO leaf: the prompt
# plus one completion) must stay near the 64k tokens S2's actor step was measured at (55-60 GB per GPU), since the FSDP
# actor has no context parallelism. The prompt is <= 8.9k tokens (system, instance) + 0.31 tokens per history
# character (w562: 132,040 tokens at most with 400k characters) and the completion <= MA_AG_TURN, so 110k characters
# and 16k-token turns bound a sample at ~59k tokens (w562: 2 of 640 turns were over 16,384 tokens).
# flint.miles_agent.log_rollout ends the run before training if a sample is over MA_RL_MAX_SAMPLE_TOKENS.
export MA_LG_CTX_CHARS=110000 MA_AG_TURN=16384 MA_RL_MAX_SAMPLE_TOKENS=70000
# a wall cap per episode: 128k episodes take 17-44 min (w634, w647), and one at 67 min (w647 rollout 0) kept 3 engines
# at 0% for 45-50 min; at 45 min an episode ends as at its token budget (scored the same way), so an engine idles at
# most ~28 min. The game RL optimizes is then the time-limited one; it changes only episodes slower than 45 min.
export MA_LG_WALL_MIN=45
# flint.miles_agent.log_rollout: the run directory, and where the agents write traces (moved per rollout)
export MA_RL_OUT=$OUT MA_AG_TRACE_DIR=$OUT/traces_live
# smoke: the same path, short: 8 episodes x 4k tokens per step, the context cut to 20k characters (trims happen). The
# game's settings must be in the environment before `ray start` below: Ray's workers, the agents among them, take the
# environment Ray started with (w624's smoke set them after it, and its agents played 128k). Each episode checks its
# game's settings against $OUT/game_settings.json (flint.miles_agent.run) and stops the run if they differ.
case "$MODE" in smoke*|train67_smoke) export MA_LG_TOKENS=4000 MA_LG_CTX_CHARS=20000;; esac
# train67 (Frontier-CS v1.1's 67 training problems): P problems per step x G episodes each, B = P x G episodes on the
# one TP8 engine at once (defaults 4 x 8 = 32), a GRPO group per problem (score centering within each group). What
# changes with B, and why:
# - the wall cap: 32 sessions share one engine, so each decodes slower than run20's 8 (1 x TP8 by concurrency c: 138
#   tokens/s per request at c = 8, ~5.5 + 0.22c ms per step, so ~1.5x the decode time at c = 32; the LLM is ~70% of an
#   episode's wall), and an episode takes ~1.35x as long; 60 min keeps the share of episodes ended by the cap near
#   run20's (4.4% at 45 min) instead of ~9-12%. The game RL optimizes is the 60-min one (game_settings.json);
# - the stop rules as fractions of the rollout's episodes (flint.miles_agent._stop_rule): more than 25% at the wall
#   cap in 2 rollouts in a row (run20: more than 1 of 8); more than 25% out of the loss (a failed episode takes its
#   group of G out: 1 group of 4 continues, 2 stop, after a checkpoint);
# - the sample check runs in the rollout process on the samples miles holds (no 4 x 5.7 GB rollout.pt per step is
#   written and read back: host memory); every 5th rollout's samples are written per episode (r<k>/samples/);
# - the judge: at most 10 programs judged at once over all 32 episodes (each episode is a process of its own, whose
#   own limit would allow 11 each), and at most 12 case-worker processes per episode (each ~50 MB; the default, one
#   per core, would allow 96 per episode).
T67_P=${T67_P:-4} T67_G=${T67_G:-8}
T67_B=$((T67_P * T67_G))
case "$MODE" in
  train67*)
    export MA_LG_WALL_MIN=60 MA_RL_P=$T67_P MA_RL_WALL_STOP_FRAC=0.25 MA_RL_MAX_DROPPED_FRAC=0.25 MA_RL_DUMP_EVERY=5 \
      MA_RL_DATA=$OUT/data_train67.jsonl FCS_RM_NODE_SLOTS=10 FCS_CASE_PROCS=12 MA_LG_DEDUP_SUBMIT=1 ;;
esac
# dupend=N (rl/rl_job.sh -> T67_DUP_END; owner popups 2026-11-06 「RL里面应该也惩罚这个行为」, 「首先prompt里写，然后
# 触发的那个turn就记0分，前面都正常」): the N-th unjudged duplicate submit in a row ends the agent's game at that turn
# (flint/game.py MA_LG_DUP_END, stated in the instance message), and that turn's reward is 0 (flint.miles_agent:
# split_dup_end_turns in the sample filter, reward_post_process); unset: the game and the rewards as before
DUPEND_TRAIN=()
if [ -n "${T67_DUP_END:-}" ]; then
  case "$MODE" in train67*) ;; *) echo "[s5] dupend=N is a train67 setting"; exit 1;; esac
  [[ $T67_DUP_END =~ ^[1-9][0-9]*$ ]] || { echo "[s5] T67_DUP_END: $T67_DUP_END"; exit 1; }
  export MA_LG_DUP_END=$T67_DUP_END
  DUPEND_TRAIN=(--custom-reward-post-process-path flint.miles_agent.reward_post_process)
  echo "[s5] game: $MA_LG_DUP_END unjudged duplicate submits in a row end an agent's game; that turn's reward is 0"
fi
# cutzero=1 (rl/rl_job.sh -> T67_CUT_ZERO; owner 2026-11-06 「嗯用方案B」): a reply cut off at the turn limit
# (MA_AG_TURN) is a sample of its own with reward 0 (flint.miles_agent: split_cut_turns in the sample filter,
# reward_post_process), and the instance message states it (flint/game.py MA_LG_CUT_ZERO); unset: as before
if [ -n "${T67_CUT_ZERO:-}" ]; then
  case "$MODE" in train67*) ;; *) echo "[s5] cutzero=1 is a train67 setting"; exit 1;; esac
  [ "$T67_CUT_ZERO" = 1 ] || { echo "[s5] T67_CUT_ZERO: $T67_CUT_ZERO"; exit 1; }
  export MA_LG_CUT_ZERO=1
  DUPEND_TRAIN=(--custom-reward-post-process-path flint.miles_agent.reward_post_process)
  echo "[s5] game: a reply cut off at the ${MA_AG_TURN:-?}-token turn limit is a sample of its own with reward 0"
fi
# kl=B (rl/rl_job.sh -> T67_KL; owner popups 2026-11-06 「方案A肯定也要做啊，这不是GRPO里的必备项吗」 and 「和 rollout
# 引擎共用 4 张卡（推荐）」): a KL loss to the original Flint at weight B (low_var_kl, k3 clamped; on the sampled token's
# full-vocabulary log-probs while the policy score stays over the replayed support: miles flint-grpo-kl), against the
# drift that runs against their reward (ledger note_2026_11_06_w794_trace_alarm_kl). The trainer has no room for a
# reference model (host memory peaks at 790-819 of the 860 GiB guard), so a reference server with the original weights
# shares the run's own engine GPUs (start_ref_engine below) and the producer scores every sample of a finished group
# with it (flint/ref_scorer.py; miles flint-grpo-kl-rollref --ref-log-probs-from-rollout). Async only: the scoring
# hook is the async producer's. The rollout engine gives up GPU memory for it: 0.80 of each GPU instead of 0.85, and a
# sliding-window KV pool of 0.1 of the full one instead of 0.4 (the window is 513 tokens: w1400's engine used 3% of its
# 1.41M sliding-window tokens at its peak, 36% of its 3.52M full ones)
KL_TRAIN=()
if [ -n "${T67_KL:-}" ]; then
  case "$MODE" in train67|train67_test) ;; *) echo "[s5] kl=B is a train67 or train67_test setting"; exit 1;; esac
  [ -n "${T67_ASYNC:-}" ] || { echo "[s5] kl=B needs async=1 (the reference scoring is the async producer's)"; exit 1; }
  [[ $T67_KL =~ ^0\.[0-9]*[1-9]$ ]] || { echo "[s5] T67_KL: $T67_KL"; exit 1; }
  KL_TRAIN=(--use-kl-loss --kl-loss-coef "$T67_KL" --kl-loss-type low_var_kl --ref-log-probs-from-rollout)
  # the producer runs in a Ray worker, which takes the environment of `ray start` below: it reads the server's URL
  # from MA_REF_FILE, which start_ref_engine writes once the server passed its check (this segment's server: a file
  # from an earlier segment names a port that is gone)
  export MA_REF_FILE=$OUT/ref_engine.url MA_REF_LOG=$OUT/ref_scoring.jsonl
  rm -f -- "$MA_REF_FILE"
  echo "[s5] loss: a KL to the original Flint at weight $T67_KL (low_var_kl), the reference scored during rollout"
fi
# - milestones (train67 only): besides the newest 2 full checkpoints, the model part (DCP model/ and meta.json, ~149 GB;
#   no optimizer) of every checkpoint iter_k with k % 20 == 0 stays, in $OUT/milestones/iter_k: hard links made by
#   flint.miles_agent.log_rollout before its pruning deletes the checkpoint, and once more after train.py ends (the
#   last checkpoint comes after the last rollout's hook). The full checkpoints and the resume are as before.
[ "$MODE" = train67 ] && export MA_RL_KEEP_EVERY=20
# - a submit of code the agent already submitted is not judged again (flint/game.py MA_LG_DEDUP_SUBMIT; trace analysis
#   2026-10-31: run20 rollout 17, one episode submitted one solution.cpp 124 times in a row, with a positive advantage);
#   run20, base_tp8 and the evals keep judging every submit
python3 -c "import json, os; json.dump({k: os.environ[k] for k in ('MA_LG_N', 'MA_LG_TOKENS', 'MA_LG_CTX_CHARS', 'MA_AG_TURN', 'MA_LG_WALL_MIN', 'MA_LG_DEDUP_SUBMIT', 'MA_LG_DUP_END', 'MA_LG_CUT_ZERO') if k in os.environ}, open('$OUT/game_settings.json', 'w'))"
ulimit -n "$(ulimit -Hn)"
NG=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
JNUM=${JID//[!0-9]/}
P=$((30000 + (JNUM % 1000) * 16))

# the FSDP actor (S2's step, flint_hf through the fork's Flint adapter): full fine-tuning, fp32 master weights and Adam
# on the CPU (FSDP2 CPU offload), gradient checkpointing; GRPO advantages without std normalization; score centering
# over the sampler's top-32 candidates (support mode: top_k 20 <= 32), no importance weight; one optimizer step per
# rollout (8 episodes = 8 rollouts of miles' compact samples per step); routing replay of the rollout's experts, with
# the replayed-vs-own top-k mismatch logged per step. Micro-batches of at most 64k tokens (one sample alone if longer);
# a step with too few samples to give every rank its micro-batches gets zero-weight padding copies (the fork's schedule).
# train67 replaces the batch flags (TRAIN67 below); the 64k-token micro-batch is the actor's memory bound per GPU, not
# a batch size, so it stays: a step of B = 32 has ~4x run20's micro-batches (~32 per rank) and ~4x its train time.
# malloc=keep (rl/rl_job.sh -> T67_MALLOC): from the end of the process's first optimizer step on, large buffers come from glibc's heap and stay there when freed (miles: _keep_freed_memory);
# one arena, no trim below 100 GiB. Set from the start, MALLOC_MMAP_MAX_=0 left 110 GiB of a checkpoint load on rank 0 (w885) and a peak 75 GiB higher at the first step (w958)
TRAIN_ENV='{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "11", "MKL_NUM_THREADS": "11"'
[ -z "${T67_MALLOC:-}" ] || TRAIN_ENV+=', "MALLOC_ARENA_MAX": "1", "MALLOC_TRIM_THRESHOLD_": "107374182400", "MILES_KEEP_FREED_AFTER_FIRST_STEP": "1"'
TRAIN_ENV+='}'
TRAIN=(--train-backend fsdp --hf-checkpoint "$FLINT" --attn-implementation eager
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NG" --seed 1 --fsdp-cpu-offload
  --train-env-vars "$TRAIN_ENV"
  --advantage-estimator grpo --disable-grpo-std-normalization --kl-coef 0 --entropy-coef 0
  --loss-type score_centering --score-centering-is none --rollout-top-logprobs-num 32
  --optimizer adam --lr 2e-7 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --gradient-checkpointing --use-dynamic-batch-size --max-tokens-per-gpu 65536
  --rollout-batch-size 1 --n-samples-per-prompt 8 --global-batch-size 8
  --use-rollout-routing-replay --log-routing-replay-mismatch)
# expandable segments only in the actor: torch_memory_saver, which frees the engines' memory while the actor trains,
# turns itself off in a process that has them

# Flint's serving flags (the eval servers: TP2, fa3, Triton MoE, fp32 LM head); 0.72 of each GPU for the engine (its
# weights ~37 GB, KV for 2 episodes of up to ~150k tokens), the rest for what the offloaded actor keeps on the GPU
# engine width: 4 x TP2 by default; smoke_tp4 / smoke_tp8 serve the same episodes from 2 x TP4 / 1 x TP8 engines, so
# no engine waits for the group's slowest episode (a serving change: same weights, same sampling)
ENGINE_TP=2
case "$MODE" in *_tp4) ENGINE_TP=4;; *_tp8|run20|train67*) ENGINE_TP=8;; esac   # run20: 1 x TP8 from rollout 4 (w662)
SERVE=(--rollout-num-gpus "$NG" --rollout-num-gpus-per-engine "$ENGINE_TP"
  --sglang-attention-backend fa3 --sglang-moe-runner-backend triton --sglang-enable-fp32-lm-head
  --sglang-context-length 262144 --sglang-chunked-prefill-size 4096 --sglang-swa-full-tokens-ratio 0.4
  --sglang-mem-fraction-static 0.72 --sglang-grammar-backend flint)
# engine levers of the train67 modes (rl/rl_job.sh levers=LIST -> T67_LEVERS; empty or unset: the engine as before,
# argv byte for byte), each measured in tmp/rollout_eff (engine bench w768, kernels w769, consistency gates T1-T3):
#   tp4       2 x TP4 engines instead of 1 x TP8 (KV 2.77M tokens per engine vs 3.46M at TP8)
#   gate      the fused sigmoid router (FLINT_FUSED_SIGMOID_GATE=1, serving/flint_sglang/flint_sglang/model.py)
#   graphs    a decode CUDA graph at every batch size 1..32 (SGLang's default list pads between 1 2 4 8 12 16 24 32)
#   arfusion  FlashInfer allreduce fusion (trtllm backend)
#   pgraph    prefill CUDA graphs (breakable backend; breaks colocated RL: SGLang refuses it with miles'
#             SGLANG_MEMORY_SAVER_CUDA_GRAPH, w787)
#   moecfg    fused-MoE Triton configs tuned for TP8 (E=256, N=64; kernels job w779): SGLANG_MOE_CONFIG_DIR =
#             serving/moe_configs (other shapes, e.g. N=128 at TP4, keep SGLang's defaults)
# Set before `ray start` below: the engines take Ray's environment.
T67_LEVERS=${T67_LEVERS:-} T67_TP=8
for lever in ${T67_LEVERS//,/ }; do
  case "$lever" in
    tp4) T67_TP=4;;
    gate) export FLINT_FUSED_SIGMOID_GATE=1;;
    graphs) SERVE+=(--sglang-cuda-graph-bs-decode $(seq 1 32) $(seq 40 8 256) $(seq 272 16 512));;
    arfusion) SERVE+=(--sglang-flashinfer-allreduce-fusion-backend trtllm);;
    pgraph) SERVE+=(--sglang-cuda-graph-backend-prefill breakable);;
    moecfg) [ -f "$REPO/serving/moe_configs/configs/triton_3_8_0/E=256,N=64,device_name=NVIDIA_H100_80GB_HBM3.json" ] \
              || { echo "[s5] moecfg: no tuned config in $REPO/serving/moe_configs"; exit 1; }
            export SGLANG_MOE_CONFIG_DIR=$REPO/serving/moe_configs;;
    *) echo "[s5] unknown engine lever: $lever"; exit 1;;
  esac
done
case "$MODE" in train67*) ENGINE_TP=$T67_TP; SERVE[3]=$ENGINE_TP;; esac
[ -z "$T67_LEVERS" ] || echo "[s5] engine levers: $T67_LEVERS (train67 engines: $((8 / T67_TP)) x TP$T67_TP, FLINT_FUSED_SIGMOID_GATE=${FLINT_FUSED_SIGMOID_GATE:-unset}, SGLANG_MOE_CONFIG_DIR=${SGLANG_MOE_CONFIG_DIR:-unset})"
# cmdtimeout=N (rl/rl_job.sh -> T67_CMD_TIMEOUT; unset or 60: the game as before): the per-command limit in seconds
# (flint/sandbox.py MA_AG_TIMEOUT, also stated in the instance message), set before `ray start` so every agent has it
if [ -n "${T67_CMD_TIMEOUT:-}" ] && [ "$T67_CMD_TIMEOUT" != 60 ]; then
  [[ $T67_CMD_TIMEOUT =~ ^[1-9][0-9]*$ ]] || { echo "[s5] T67_CMD_TIMEOUT: $T67_CMD_TIMEOUT"; exit 1; }
  export MA_AG_TIMEOUT=$T67_CMD_TIMEOUT
  echo "[s5] game: command time limit $MA_AG_TIMEOUT s (default 60)"
fi
# elastic=1 (rl/rl_job.sh -> T67_ELASTIC, IP: the node's eno8303 address): anchored elastic RL (tmp/elastic_rl/anchored.md).
# The colocated engines stay; engine jobs that register in $OUT/engines (rl/elastic_engine.sh) join as extra cells
# (flint.engine_registry, miles' --extra-inference-engine-provider-path, branch flint-grpo-elastic), each routed from the
# next weight push on: IPC to the colocated engines, NCCL broadcast to the others in the same session. Ray and the
# NCCL master bind the routable address; the push to another node gets 900 s. $OUT/engine_spec.json (below, at the
# start) is what the engine jobs serve: this argv and the serving environment the colocated engines get.
ELASTIC=()
if [ -n "${T67_ELASTIC:-}" ]; then
  : "${IP:?}"
  [[ $MODE == train67* ]] || { echo "[s5] elastic=1 is for the train67 modes"; exit 1; }
  # an agent's request whose engine left is retried for 10 min (flint/chat.py), past the run's withdrawal of the engine
  # the weight push to engines on other nodes: NCCL over TCP on bond0 (100 Gb/s IPoIB; 7.9 GB/s node to node in 4
  # streams, 2026-11-02), not eno8303 (1 Gb/s: w867 pushed 80 GB in 678 s per step) and not IB verbs (the containers
  # run --contain, without /dev/infiniband); NET logs show the transport each communicator picked
  export NCCL_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 NCCL_NSOCKS_PERTHREAD=4 NCCL_SOCKET_NTHREADS=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
  export MA_ENGINE_REGISTRY=$OUT MASTER_ADDR=$IP no_proxy=127.0.0.1,localhost,$IP MA_AG_RETRY_S=600
  ELASTIC=(--extra-inference-engine-provider-path flint.engine_registry.provider --update-weights-timeout 900)
fi
# async=1 (rl/rl_job.sh -> T67_ASYNC, T67_STALE): fully async RL on a split node. miles' async with an FSDP actor needs
# the engines on GPUs of their own (train_async.py refuses --colocate unless --fully-async, and --fully-async --colocate
# refuses FSDP), so the actor trains on 4 GPUs (FSDP with CPU offload, as before) and the run's own engine serves 1 x TP4
# on the other 4 (mem 0.85: the GPUs are its own); with elastic=1 every engine job adds 2 x TP4. The producer
# (flint.elastic_rollout) keeps MA_ASYNC_SAMPLES_PER_ENGINE (16) episodes per routed engine in flight; each step trains on
# the first P finished groups; a weight push pauses running requests in place and resumes them on the new weights, so an
# episode's tokens may come from several weight versions: score centering weights each token by TIS (pi_train /
# mu_rollout, clipped at 2) instead of none, and --max-weight-staleness T67_STALE (when set) drops a group whose oldest
# token is more than that many steps old. The engines keep the colocated runs' all-reduce (custom v2 off; miles turns
# it on for engines that are not colocated), so the async engines serve with production's numerics.
ASYNC=()
if [ -n "${T67_ASYNC:-}" ]; then
  [[ $MODE == train67 || $MODE == train67_test ]] || { echo "[s5] async=1 is for train67 and train67_test"; exit 1; }
  export MA_RL_ASYNC=1 MA_ASYNC_SAMPLES_PER_ENGINE=${MA_ASYNC_SAMPLES_PER_ENGINE:-16} SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
  # the warm-up engine of a fresh start (rl/rl_job.sh, flint.elastic_rollout): its files; a turn its leave cuts short
  # is retried on the run's own engine (flint/chat.py), as an engine job's (elastic=1)
  export MA_WARMUP_DIR=$OUT/warmup MA_AG_RETRY_S=600
  ENGINE_TP=4 T67_TP=4
  SERVE[1]=4 SERVE[3]=4
  ASYNC=(--actor-num-gpus-per-node 4 --sglang-mem-fraction-static 0.85 --fully-async
    --rollout-function-path flint.elastic_rollout.ElasticFullyAsyncRolloutFn --score-centering-is tis)
  [ -z "${T67_STALE:-}" ] || ASYNC+=(--max-weight-staleness "$T67_STALE")
  # stale=N >= 2: the finished-group buffer holds the whole window (miles' default holds 2 P), so that a put never blocks the producer
  [ "${T67_STALE:-1}" -le 1 ] || ASYNC+=(--async-data-buffer-capacity-factor "$((T67_STALE + 1))")
  # order=oldest (rl/rl_job.sh -> T67_ORDER): a drain takes the oldest weight version first and waits for an older group when it must (flint.elastic_rollout.OldestFirstDataBuffer)
  [ -z "${T67_ORDER:-}" ] || ASYNC+=(--custom-async-data-buffer-path flint.elastic_rollout.OldestFirstDataBuffer)
  echo "[s5] async: actor 4 GPUs, own engine 1 x TP4, $MA_ASYNC_SAMPLES_PER_ENGINE episodes per engine in flight, TIS, staleness bound ${T67_STALE:-none}${T67_ORDER:+, drains $T67_ORDER first}"
  # kl=B: the rollout engine shares its GPUs with the reference server (KL_TRAIN above; later flags win in miles' parser).
  # SGLang keeps (1 - mem fraction) of the memory free at the engine's own start (pool_configurator: slack =
  # pre_model_load_memory x (1 - mem_fraction_static)), and the reference server holds ~23 GB by then: 0.70 keeps
  # ~16 GB for the engine's own buffers (~4.2 GB), CUDA graphs (~3.1), the weight push (~2.9) and the reference
  # server's scoring (~0.7 at 512-row log-prob chunks), and leaves a KV pool of ~19 GB, ~2.2M full-attention tokens
  # (audit 2026-11-06: at 0.80 the slack is ~10.6 GB and the first scored group runs a GPU out of memory)
  [ -z "${T67_KL:-}" ] || ASYNC+=(--sglang-mem-fraction-static 0.70 --sglang-swa-full-tokens-ratio 0.1)
fi
# kl=B: the reference server (rl/ref_engine_argv.py: the run's own engine's arguments, the original Flint, a KV pool
# for one sample at a time) on the GPUs miles gives its own engine (the last 4 of MILES_GPU_ORDER or of this job's
# GPUs), started and checked before the trainer, so that the rollout engine sizes its KV pool around it. The check
# (flint.ref_scorer check): the server's own sampled tokens, scored again, come back with their sampling log-probs.
# Prints the URL the producer reads (MA_REF_URL); its log is $OUT/ref_engine.log, its per-group scoring $OUT/ref_scoring.jsonl.
start_ref_engine() {
  local gpus base port out env_line argv_line t0=$SECONDS
  # miles orders the GPUs by MILES_GPU_ORDER, else by id (placement_group.py): without an order, sort this job's list
  IFS=, read -r -a gpus <<< "${MILES_GPU_ORDER:-$(tr , '\n' <<< "$CUDA_VISIBLE_DEVICES" | sort -n | paste -sd,)}"
  [ ${#gpus[@]} -eq 8 ] || { echo "[s5] ref: 8 GPUs expected, got ${gpus[*]}"; return 1; }
  base=$(python3 - <<'PY'
import random, socket
for _ in range(500):
    b, socks = random.randrange(20000, 29000), []
    try:
        for p in range(b, b + 10):
            s = socket.socket(); socks.append(s); s.bind(("0.0.0.0", p))
        print(b); break
    except OSError:
        pass
    finally:
        for s in socks: s.close()
PY
)
  [ -n "$base" ] || { echo "[s5] ref: no free ports"; return 1; }
  port=$((base + 9))
  # a KV pool for one sample of up to MA_RL_MAX_SAMPLE_TOKENS (70k; a group with a longer one is not scored) and its
  # prefix in the radix cache
  out=$(REF_MEM_FRACTION=0.35 REF_MAX_TOKENS=100000 python3 "$REPO/rl/ref_engine_argv.py" "$OUT/engine_spec.json" "$port" "$base") \
    || { echo "[s5] ref: argv failed: $out"; return 1; }
  env_line=$(grep '^ENV ' <<< "$out" | cut -d' ' -f2-) argv_line=$(grep '^ARGV ' <<< "$out" | cut -d' ' -f2-)
  echo "[s5] ref: differs from the run's own engine in: $(grep '^DIFF ' <<< "$out" | cut -d' ' -f2-)"
  (
    for kv in $env_line; do export "$kv"; done
    # 512-row log-prob chunks: a 2,048-row chunk of fp32 logits over 150,144 tokens takes ~2.6 GB at its peak, ~0.7 here
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpus[*]:4:4}")" SGLANG_CACHE_DIR=$TMPDIR/sglang_ref FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer_ref \
      SGLANG_LOGPROB_CHUNK_SIZE=512
    eval "exec $argv_line"
  ) > "$OUT/ref_engine.log" 2>&1 &
  REF_PID=$!
  echo "[s5] ref: reference server pid $REF_PID on GPUs ${gpus[*]:4:4}, port $port (log $OUT/ref_engine.log)"
  until [ "$(curl -s --noproxy '*' -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health_generate")" = 200 ]; do
    kill -0 $REF_PID 2>/dev/null || { echo "[s5] ref: the server ended while loading: $(tail -n 5 "$OUT/ref_engine.log")"; return 1; }
    [ $((SECONDS - t0)) -lt 1800 ] || { echo "[s5] ref: not healthy after 30 min"; return 1; }
    sleep 10
  done
  grep -hE "Load weight end|KV Cache is allocated|Memory pool end" "$OUT/ref_engine.log" | grep "TP0" | sed 's/^/[s5] ref: /' | cut -c1-220
  python3 -m flint.ref_scorer check "http://127.0.0.1:$port" "$FLINT" || { echo "[s5] ref: the alignment check failed"; return 1; }
  echo "http://127.0.0.1:$port" > "$MA_REF_FILE.tmp" && mv -f -- "$MA_REF_FILE.tmp" "$MA_REF_FILE"
  echo "[s5] ref: serving after $((SECONDS - t0)) s"
}
write_engine_spec() {   # write_engine_spec FILE ARGV...: the spec rl/elastic_engine.sh serves (rl/elastic_engine_argv.py)
  python3 - "$1" "$ENGINE_TP" "$REPO" "$MILES" "$FLINT" "${@:2}" <<'PY'
import json, os, subprocess, sys
out, tp, repo, miles, flint, argv = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
# the serving variables (not this job's cache paths under TMPDIR: an engine job keeps its own)
env = {k: v for k, v in os.environ.items() if k.startswith(("SGLANG_", "FLASHINFER_", "FLINT_")) and not v.startswith(os.environ["TMPDIR"])}
head = lambda d: subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
spec = {"tp": tp, "argv": argv, "env": env, "repo": repo, "miles": miles, "flint": flint,
        "repo_commit": head(repo), "miles_commit": head(miles), "levers": os.environ.get("T67_LEVERS", "")}
json.dump(spec, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
print(f"[s5] elastic: {out}: TP{tp}, env {sorted(env)}")
PY
}
# the long game as a miles agent on problem 0 (flint/miles_agent.py), sampling as in the eval; the engines return each
# token's routed experts, incremental rows per turn (pause mode in_place); an episode the judge could not score takes
# its group out of the loss (drop_failed_groups) and fails the rollout's check (log_rollout), which ends the run
DATA=$OUT/data_p0.jsonl
case "$MODE" in train67*) DATA=$OUT/data_train67.jsonl;; esac
GAME=(--prompt-data "$DATA" --input-key prompt --label-key label --metadata-key metadata
  --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-top-k 20
  --rollout-max-response-len 16384 --max-seq-len 262144
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path flint.miles_agent.run --use-session-server v2 --tito-model flint
  --session-message-matcher flint.miles_agent.message_matches
  --sglang-tool-call-parser flint --sglang-reasoning-parser flint --pause-generation-mode in_place
  --rollout-sample-filter-path flint.miles_agent.drop_failed_groups
  --custom-rollout-log-function-path flint.miles_agent.log_rollout
  --save-debug-rollout-data "$OUT/r{rollout_id}/rollout.pt")
# colocated: the engines release weights and KV while the actor trains, the actor goes to the CPU while they generate;
# the weights go to the engines after every step. The actor (model only, no optimizer) is saved once, after the last
# step: 1000 rows of one problem keep miles' epoch boundary, also a save point, out of the run.
RL=(--colocate --save "$OUT/actor" --save-interval 100000 --no-save-optim --save-trigger-sentinel "$OUT/save_now")
# run20 resumes: actor, optimizer, scheduler, RNG and the data position every 2 steps (and at the end), loaded from the
# same directory on every start (miles skips the load when there is none yet); log_rollout keeps the newest 2
RUN20=(--colocate --save "$OUT/actor" --load "$OUT/actor" --save-interval 2 --save-trigger-sentinel "$OUT/save_now")
# train67: P prompts (problems) per rollout from the pre-shuffled file (rl/train67_data.py; miles reads it in order, no
# --rollout-shuffle), G episodes each, one optimizer step over all B episodes (global batch B = B compact rollouts); no
# per-step debug dump (the hook writes every 5th rollout's samples). train67: run20's resumable saves (actor,
# optimizer, data position every 2 steps, the newest 2 kept); train67_test: 2 steps, the full checkpoint after each
# (the save is the host-memory peak, so the test meets it after step 1), resumable the same way (wq runs over 3 h are
# resumable); train67_smoke: 2 steps of 4k, no save.
TRAIN67=(--rollout-batch-size "$T67_P" --n-samples-per-prompt "$T67_G" --global-batch-size "$T67_B")
# seed=N (rl/rl_job.sh; the default 67 adds nothing, so the argv stays as it was): --seed 1 + 100 (N - 67) after TRAIN's
# --seed 1 (miles' parser keeps the last) is the actor's torch seed and the engines' SGLang seed (miles: engine i serves
# with --seed + i); --rollout-seed 42 + 100 (N - 67) (miles' default 42; it seeds the prompt shuffle and
# deterministic-inference sampling, both off here). 100 apart: two replicates' engine seeds never meet. The data order
# follows N in rl/train67_data.py. The game's own randomness (the command order, flint/game.py) is seeded per episode
# from a random group index (flint/miles_agent.py), not from a run seed.
T67_SEED=${T67_SEED:-67}
[[ $T67_SEED =~ ^[1-9][0-9]*$ ]] || { echo "[s5] T67_SEED: $T67_SEED"; exit 1; }
[ "$T67_SEED" = 67 ] || TRAIN67+=(--seed $((1 + 100 * (T67_SEED - 67))) --rollout-seed $((42 + 100 * (T67_SEED - 67))))
GAME67=() skip=
for x in "${GAME[@]}"; do
  if [ -n "$skip" ]; then skip=; elif [ "$x" = --save-debug-rollout-data ]; then skip=1; else GAME67+=("$x"); fi
done
T67_RUN=(--colocate --save "$OUT/actor" --load "$OUT/actor" --save-interval 2 --save-trigger-sentinel "$OUT/save_now")
T67_TEST=(--colocate --save "$OUT/actor" --load "$OUT/actor" --save-interval 1 --save-trigger-sentinel "$OUT/save_now")
[ -z "${T67_NOSAVE:-}" ] || T67_TEST=(--colocate)   # rl_job.sh train67_test save=0: a placement test needs no checkpoints (2 x 480 GiB)
[ ${#ASYNC[@]} -eq 0 ] || { T67_RUN=("${T67_RUN[@]:1}"); T67_TEST=("${T67_TEST[@]:1}"); }   # async: not colocated

MEMPROBE=(--prompt-data "$OUT/data_p0.jsonl" --input-key prompt --label-key label --metadata-key metadata --num-rollout 2
  --load-debug-rollout-data "$MEMPROBE_DATA/rollout_{rollout_id}.pt" --rollout-temperature 1.0 --rollout-top-p 0.95
  --rollout-top-k 20)
if [ "$MODE" = args ]; then   # off-GPU: miles parses and checks each mode's arguments (rl/args_check.py)
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${MEMPROBE[@]}" || exit 1
  python3 "$REPO/rl/schedule_check.py" "$MEMPROBE_DATA/rollout_0.pt" "${TRAIN[@]}" "${MEMPROBE[@]}" || exit 1
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 2 || exit 1
  python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RUN20[@]}" --num-rollout 20 || exit 1
  # train67, one argv per mode as the job builds it (TP8 engine), and the schedule of a B-episode step: the
  # recorded 8-episode batch, its episodes repeated as B (rl/schedule_check.py --episodes)
  SERVE8=("${SERVE[@]}"); SERVE8[3]=$T67_TP
  DATA=$OUT/data_train67.jsonl; GAME[1]=$DATA; GAME67[1]=$DATA
  export T67=1 T67_P T67_G T67_SEED
  for m in train67 train67_test train67_smoke; do
    case $m in
      train67) A=("${T67_RUN[@]}" --num-rollout 40);;
      train67_test) A=("${T67_TEST[@]}" --num-rollout 2);;
      train67_smoke) A=(--colocate --num-rollout 2);;
    esac
    echo "== $m"
    T67_MODE=$m python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${TRAIN67[@]}" "${SERVE8[@]}" "${GAME67[@]}" "${A[@]}" || exit 1
  done
  echo "== train67 elastic"
  T67_MODE=train67 T67_ELASTIC=1 python3 "$REPO/rl/args_check.py" "${TRAIN[@]}" "${TRAIN67[@]}" "${SERVE8[@]}" "${GAME67[@]}" \
    "${T67_RUN[@]}" --num-rollout 40 --extra-inference-engine-provider-path flint.engine_registry.provider --update-weights-timeout 900 || exit 1
  echo "== train67 elastic: an engine job's argv from the spec (rl/elastic_engine_argv.py, at a made-up address)"
  ENGINE_TP=$T67_TP write_engine_spec "$OUT/engine_spec.json" "${TRAIN[@]}" "${TRAIN67[@]}" "${SERVE8[@]}" "${GAME67[@]}" \
    "${T67_RUN[@]}" --num-rollout 40 --extra-inference-engine-provider-path flint.engine_registry.provider --update-weights-timeout 900 || exit 1
  python3 "$REPO/rl/elastic_engine_argv.py" "$OUT/engine_spec.json" 172.17.0.1 20040 20000 10 || exit 1
  echo "== train67 async (split node, fully async) with elastic engines; an engine job's argv from its spec"
  SERVE4=("${SERVE[@]}"); SERVE4[1]=4; SERVE4[3]=4
  ASYNC_ARGS=(--actor-num-gpus-per-node 4 --sglang-mem-fraction-static 0.85 --fully-async
    --rollout-function-path flint.elastic_rollout.ElasticFullyAsyncRolloutFn --score-centering-is tis --max-weight-staleness 4
    --async-data-buffer-capacity-factor 5 --custom-async-data-buffer-path flint.elastic_rollout.OldestFirstDataBuffer)
  A=("${TRAIN[@]}" "${TRAIN67[@]}" "${SERVE4[@]}" "${GAME67[@]}" "${T67_RUN[@]:1}" --num-rollout 40 "${ASYNC_ARGS[@]}"
    --extra-inference-engine-provider-path flint.engine_registry.provider --update-weights-timeout 900)
  T67_MODE=train67 T67_ASYNC=1 T67_ELASTIC=1 python3 "$REPO/rl/args_check.py" "${A[@]}" || exit 1
  SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 ENGINE_TP=4 write_engine_spec "$OUT/engine_spec.json" "${A[@]}" || exit 1
  python3 "$REPO/rl/elastic_engine_argv.py" "$OUT/engine_spec.json" 172.17.0.1 20040 20000 10 || exit 1
  echo "== train67 async with kl=0.01: the KL loss read from the rollout's reference scores (miles flint-grpo-kl-rollref), the rollout engine beside the reference server, and the reference server's argv"
  AK=("${A[@]}" --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl --ref-log-probs-from-rollout
    --sglang-mem-fraction-static 0.70 --sglang-swa-full-tokens-ratio 0.1)
  T67_MODE=train67 T67_ASYNC=1 T67_ELASTIC=1 python3 "$REPO/rl/args_check.py" "${AK[@]}" || exit 1
  SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 ENGINE_TP=4 write_engine_spec "$OUT/engine_spec_kl.json" "${AK[@]}" || exit 1
  REF_MEM_FRACTION=0.35 REF_MAX_TOKENS=100000 python3 "$REPO/rl/ref_engine_argv.py" "$OUT/engine_spec_kl.json" 20049 20040 || exit 1
  echo "== train67 schedule (B=$T67_B)"
  python3 "$REPO/rl/schedule_check.py" "$MEMPROBE_DATA/rollout_0.pt" --episodes "$T67_B" --group "$T67_G" \
    "${TRAIN[@]}" "${TRAIN67[@]}" "${MEMPROBE[@]}"
  exit
fi

RAY_IP=127.0.0.1; [ ${#ELASTIC[@]} -eq 0 ] || RAY_IP=$IP
echo "[s5] $(date -u +%FT%TZ) inner $MODE: $NG GPUs, ray $RAY_IP:$P"
ray start --head --node-ip-address "$RAY_IP" --num-gpus "$NG" --port=$P --dashboard-port=$((P + 1)) \
  --dashboard-agent-listen-port=$((P + 2)) --dashboard-agent-grpc-port=$((P + 3)) --metrics-export-port=$((P + 4)) \
  --ray-client-server-port=$((P + 5)) --runtime-env-agent-port=$((P + 6)) --node-manager-port=$((P + 7)) \
  --object-manager-port=$((P + 8)) --include-dashboard=false --disable-usage-stats --temp-dir=/tmp/ray >/dev/null \
  || { echo "[s5] ray start failed"; exit 1; }
export RAY_ADDRESS=$RAY_IP:$P
trap 'ray stop --force >/dev/null 2>&1' EXIT   # this PID namespace's Ray only

# train.py in the background; once the engines serve (the first rollout request), the served-weights probe runs once
# on the initial weights (flint.miles_agent.startup_probe): if it fails, train.py is stopped, minutes into the run
train() {
  local driver=train.py
  [ ${#ASYNC[@]} -eq 0 ] || { set -- "$@" "${ASYNC[@]}"; driver=train_async.py; }
  [ ${#ELASTIC[@]} -eq 0 ] || set -- "$@" "${ELASTIC[@]}"
  # what the engines miles does not launch serve (engine jobs, async's warm-up engine): this argv, and the run's own
  # engines' serving environment
  if [ ${#ELASTIC[@]} -gt 0 ] || [ ${#ASYNC[@]} -gt 0 ]; then
    write_engine_spec "$OUT/engine_spec.json" "$@" || return 1
  fi
  # kl=B: the reference server takes its GPU memory before the rollout engine sizes its KV pool
  [ -z "${T67_KL:-}" ] || start_ref_engine || return 1
  if [ -n "${APPEND:-}" ]; then echo "=== segment $(date -u +%FT%TZ)" >> "$OUT/train.log"; else : > "$OUT/train.log"; fi
  local from=$(( $(wc -l < "$OUT/train.log") + 1 ))   # this segment's lines (run20 appends one segment after another)
  tail -n +"$from" -F "$OUT/train.log" > "$OUT/train_segment.log" 2>/dev/null &
  local tl=$!
  python3 "$MILES/$driver" "$@" >> "$OUT/train.log" 2>&1 &
  local tp=$!
  # a start check that stops train.py leaves this marker, and train() then returns 3: a SIGTERM's 143 would make wq
  # requeue a --partial job, and the next segment would fail the same way (audit 2026-11-06)
  local failed=$OUT/start_check_failed
  rm -f -- "$failed"
  (
    until grep -q 'POST /v1/chat/completions' "$OUT/train_segment.log" 2>/dev/null; do kill -0 $tp 2>/dev/null || exit; sleep 15; done
    # kl=B: the rollout engine's own free memory after its CUDA graphs, beside the reference server, must leave the
    # weight push (~2.9 GB), the reference's scoring (~0.7) and a margin: under 7 GB the run stops here, minutes in,
    # not at the first scored group (standalone at 0.85 it is ~4.3 GB, all the engine's own)
    if [ -n "${T67_KL:-}" ]; then
      free_gb=$(grep -h "TP0\] Capture target decode CUDA graph end" "$OUT/train_segment.log" | tail -1 | sed -n 's/.*avail mem=\([0-9.]*\) GB.*/\1/p')
      echo "[s5] ref: the rollout engine has ${free_gb:-?} GB free after its CUDA graphs"
      if [ -z "$free_gb" ] || awk -v f="$free_gb" 'BEGIN { exit !(f < 7.0) }'; then
        echo "[s5] ref: under 7 GB free on the engine GPUs (or not found): stopping"
        echo "engine free memory ${free_gb:-?} GB" > "$failed"; kill $tp; exit
      fi
    fi
    python3 -c "import sys; from flint.miles_agent import startup_probe as p; sys.exit(p('$OUT', '$OUT/train_segment.log', '$FLINT'))" \
      > "$OUT/probe_start.log" 2>&1 || {
        echo "[s5] startup probe failed: $(tail -1 "$OUT/probe_start.log")"
        # if train.py is already going down, the probe failed because of it: let it print its own error (w971, 2026-11-03: this kill cut it off)
        for _ in $(seq 24); do kill -0 $tp 2>/dev/null || exit; sleep 5; done
        echo "startup probe" > "$failed"; kill $tp
      }
  ) &
  wait $tp
  local rc=$?
  kill $tl 2>/dev/null
  [ -e "$failed" ] && { echo "[s5] the start check stopped train.py ($(cat "$failed")): exit 3, not requeued"; rc=3; }
  return $rc
}

t0=$SECONDS
case "$MODE" in
  base_tp8)   # control: the original weights at 1 x TP8 in the RL game, 2 rollouts of 8 episodes, no training
    train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" --debug-rollout-only --num-rollout 2 ;;
  smoke|smoke_tp4|smoke_tp8)   # 2 steps, no actor save: rollout, log_rollout (check, probe), training step, weight sync, rollout 2, probe
    train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" --colocate --num-rollout 2 ;;
  test2) train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RL[@]}" --num-rollout 2 ;;
  run20)
    echo "[s5] run20 starts from checkpoint: $(cat "$OUT/actor/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
    APPEND=1 train "${TRAIN[@]}" "${SERVE[@]}" "${GAME[@]}" "${RUN20[@]}" --num-rollout 20 ;;
  train67)
    echo "[s5] train67 (P=$T67_P G=$T67_G, ${T67_STEPS:-40} steps, seed $T67_SEED) starts from checkpoint: $(cat "$OUT/actor/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
    APPEND=1 train "${TRAIN[@]}" "${TRAIN67[@]}" "${DUPEND_TRAIN[@]}" "${KL_TRAIN[@]}" "${SERVE[@]}" "${GAME67[@]}" "${T67_RUN[@]}" --num-rollout "${T67_STEPS:-40}" ;;
  train67_test)
    echo "[s5] train67_test (P=$T67_P G=$T67_G, seed $T67_SEED) starts from checkpoint: $(cat "$OUT/actor/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
    APPEND=1 train "${TRAIN[@]}" "${TRAIN67[@]}" "${DUPEND_TRAIN[@]}" "${KL_TRAIN[@]}" "${SERVE[@]}" "${GAME67[@]}" "${T67_TEST[@]}" --num-rollout 2 ;;
  train67_smoke) train "${TRAIN[@]}" "${TRAIN67[@]}" "${DUPEND_TRAIN[@]}" "${KL_TRAIN[@]}" "${SERVE[@]}" "${GAME67[@]}" --colocate --num-rollout 2 ;;
esac
rc=$?
echo "[s5] $(date -u +%FT%TZ) train.py exit $rc after $((SECONDS - t0)) s; $(tail -1 "$OUT/probe_start.log" 2>/dev/null)"
if [ "${MA_RL_KEEP_EVERY:-0}" -gt 0 ]; then   # the milestone of a checkpoint written after the last rollout's hook
  python3 -c "from flint.miles_agent import _keep_newest_checkpoints as k; k('$OUT/actor', 2, $MA_RL_KEEP_EVERY)" \
    || echo "[s5] milestone pass failed (the checkpoints are untouched)"
  echo "[s5] milestones: $(ls "$OUT/milestones" 2>/dev/null | tr '\n' ' ')"
fi
python3 "$REPO/rl/steps.py" "$OUT"
exit $rc
