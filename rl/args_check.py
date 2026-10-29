"""Off-GPU check of an RL run's arguments (rl/rl_inner.sh args): miles' own parser and validation on the exact argv
train.py gets, then the settings the run depends on. Usage: args_check.py <train.py args...>"""
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
if a.load_debug_rollout_data is None and a.load is None:   # test2: one save at the end, no optimizer
    want |= {"no_save_optim": True, "save_interval": 100000}
elif a.load_debug_rollout_data is None:   # run20: resumable, optimizer saved every 2 steps, loaded from the same place
    want |= {"no_save_optim": False, "save_interval": 2, "load": a.save}
bad = {k: (getattr(a, k, "<missing>"), v) for k, v in want.items() if getattr(a, k, "<missing>") != v}
print("parsed:", {k: getattr(a, k, None) for k in ("num_rollout", "rollout_num_gpus", "actor_num_gpus_per_node",
                                                   "max_tokens_per_gpu", "train_env_vars", "save", "colocate")})
print("ARGS_FAIL", bad) if bad else print("ARGS_OK", len(want), "settings")
sys.exit(1 if bad else 0)
