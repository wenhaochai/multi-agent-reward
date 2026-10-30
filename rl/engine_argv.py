"""The SGLang launch line of an external Flint engine (rl/engine_inner.sh), from miles' own _compute_server_args on
the trainer's arguments (rl/ext_args.sh), so the serving flags have one source. Also prints the environment miles
gives the engines it launches (compute_inference_engine_env_vars). Runs in miles-v0.1.3.sif.

Usage: engine_argv.py HOST PORT BASE_PORT -- <train.py args...>
  BASE_PORT: the first of 40 consecutive free ports: nccl BASE, engine-info bootstrap BASE+1, dist-init BASE+8..+39.
Prints two lines: "ENV k=v k=v ..." and "ARGV <shell-quoted python -m sglang.launch_server argv>"."""
import shlex
import sys

from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.backends.sglang_utils.server_args_utils import server_args_to_argv
from miles.ray.specs.inference import compute_inference_engine_env_vars
from miles.utils.arguments import parse_args

host, port, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
assert sys.argv[4] == "--", "usage: engine_argv.py HOST PORT BASE_PORT -- <train.py args...>"
assert not host.startswith("127.") and host != "localhost", f"an external engine must bind a routable address, got {host}"
sys.argv = ["train.py"] + sys.argv[5:]
args = parse_args()
assert not args.colocate and not args.offload_rollout, "the external engine's argv must come from a non-colocated run"
kw = _compute_server_args(
    args, node_rank=0, dist_init_addr=f"{host}:{base + 8}", nccl_port=base, host=host, port=port,
    disaggregation_bootstrap_port=None, base_gpu_id=0, engine_info_bootstrap_port=base + 1, sglang_overrides=None,
    num_gpus_per_engine=args.rollout_num_gpus_per_engine, gated_launch_port=None, random_seed=args.seed)
kw = {k: v for k, v in kw.items() if v is not None}
must = {"enable_return_routed_experts": True, "enable_fp32_lm_head": True, "tp_size": args.rollout_num_gpus_per_engine,
        "reasoning_parser": "flint", "tool_call_parser": "flint", "grammar_backend": "flint",
        "attention_backend": "fa3", "moe_runner_backend": "triton", "enable_memory_saver": False}
bad = {k: (kw.get(k), v) for k, v in must.items() if kw.get(k) != v}
assert not bad, f"engine argv lacks run20's serving settings: {bad}"
env = compute_inference_engine_env_vars(args)
print("ENV " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
print("ARGV " + shlex.join(["python3", "-m", "sglang.launch_server", *server_args_to_argv({**kw, "host": host})]))
