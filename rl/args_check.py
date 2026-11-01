"""Off-GPU check of an RL run's arguments (rl/rl_inner.sh args): miles' own parser and validation on the exact argv
train.py gets, then the settings the run depends on. Usage: args_check.py <train.py args...>"""
import os
import sys

from miles.utils.arguments import parse_args

sys.argv = ["train.py"] + sys.argv[1:]
a = parse_args()
want = {"train_backend": "fsdp", "fsdp_cpu_offload": True, "advantage_estimator": "grpo",
        "grpo_std_normalization": False, "loss_type": "score_centering", "score_centering_is": "none",
        "rollout_top_logprobs_num": 32, "rollout_top_k": 20, "lr": 2e-7, "kl_coef": 0, "entropy_coef": 0,
        "global_batch_size": 8, "n_samples_per_prompt": 8, "rollout_batch_size": 1, "use_routing_replay": True,
        "use_rollout_routing_replay": True, "log_routing_replay_mismatch": True, "gradient_checkpointing": True}
if a.load_debug_rollout_data is None:   # the colocated RL run
    want |= {"colocate": True, "offload_train": True, "offload_rollout": True, "rollout_num_gpus_per_engine": 2,
             "use_session_server": "v2", "tito_model": "flint", "pause_generation_mode": "in_place",
             "sglang_mem_fraction_static": 0.72, "max_tokens_per_gpu": 65536, "rollout_max_response_len": 16384,
             "custom_rollout_log_function_path": "flint.miles_agent.log_rollout",
             "rollout_sample_filter_path": "flint.miles_agent.drop_failed_groups"}
if a.load_debug_rollout_data is None and a.load is None and os.environ.get("T67"):
    pass   # train67_smoke: no save (its settings below)
elif a.load_debug_rollout_data is None and a.load is None:   # test2: one save at the end, no optimizer
    want |= {"no_save_optim": True, "save_interval": 100000}
elif a.load_debug_rollout_data is None:   # run20: resumable, optimizer saved every 2 steps, loaded from the same place
    want |= {"no_save_optim": False, "save_interval": 2, "load": a.save}
if os.environ.get("T67"):   # train67 (rl/rl_inner.sh args): P problems x G episodes, one step over all B, one TP8 engine
    P, G, mode = int(os.environ["T67_P"]), int(os.environ["T67_G"]), os.environ["T67_MODE"]
    want |= {"rollout_batch_size": P, "n_samples_per_prompt": G, "global_batch_size": P * G,
             "rollout_num_gpus_per_engine": 8, "rollout_shuffle": False, "save_debug_rollout_data": None,
             "num_rollout": {"train67": 40}.get(mode, 2)}
    want["prompt_data"] = a.prompt_data if a.prompt_data.endswith("/data_train67.jsonl") else "<.../data_train67.jsonl>"
    if mode == "train67":
        want |= {"no_save_optim": False, "save_interval": 2, "load": a.save}
    elif mode == "train67_test":
        want |= {"no_save_optim": False, "save_interval": 1, "load": a.save}
    else:
        want |= {"save": None}
    # seed=N (rl/rl_job.sh, default 67): miles' --seed 1 + 100 (N - 67) (the actor; engine i serves with seed + i) and
    # --rollout-seed 42 + 100 (N - 67)
    n = int(os.environ.get("T67_SEED", "67"))
    want |= {"seed": 1 + 100 * (n - 67), "rollout_seed": 42 + 100 * (n - 67)}
    # one optimizer step per rollout: miles' step count is rollouts x P x G / global batch
    want["steps_per_rollout"] = 1
    a.steps_per_rollout = a.rollout_batch_size * a.n_samples_per_prompt // a.global_batch_size
    # the engine holds B sessions: one decode request each, far below SGLang's request cap; KV: B x ~60k tokens
    # of the TP8 pool's 3,163,753 at --sglang-mem-fraction-static 0.72 (run20's log), checked here as a budget
    want["kv_fits"] = True
    a.kv_fits = P * G * 60000 <= 0.9 * 3163753
    print(f"train67 {mode}: B = {P * G} episodes, KV {P * G * 60000:,} of 3,163,753 tokens "
          f"({P * G * 60000 / 3163753:.0%}), session-server workers {a.session_server_workers}")
bad = {k: (getattr(a, k, "<missing>"), v) for k, v in want.items() if getattr(a, k, "<missing>") != v}
print("parsed:", {k: getattr(a, k, None) for k in ("num_rollout", "rollout_num_gpus", "actor_num_gpus_per_node",
                                                   "max_tokens_per_gpu", "train_env_vars", "save", "colocate",
                                                   "rollout_batch_size", "n_samples_per_prompt", "global_batch_size",
                                                   "seed", "rollout_seed")})
print("ARGS_FAIL", bad) if bad else print("ARGS_OK", len(want), "settings")
sys.exit(1 if bad else 0)
