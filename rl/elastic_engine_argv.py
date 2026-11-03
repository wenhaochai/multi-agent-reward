"""The SGLang launch line of one engine of an elastic engine job (rl/elastic_engine.sh), for the run whose spec is
SPEC (RUN/engine_spec.json, written by rl/rl_inner.sh: the trainer's train.py argv and the serving environment it gives
its engines). Runs in miles-v0.1.3.sif with the run's miles on the path.

The arguments are the run's own engine's (colocated, or on GPUs of its own with async=1), as miles computes them
(_compute_server_args on the trainer's argv, the engine group's overrides), with only these changes:
  host, port, nccl_port, dist_init_addr, engine_info_bootstrap_port   this job's addresses
  base_gpu_id 0                                                       the GPUs are this engine's (CUDA_VISIBLE_DEVICES)
  gated_launch_port None                                              nothing gates the launch (miles' actor does, colocated)
  enable_memory_saver False                                           nothing to release for an actor
  mem_fraction_static 0.85                                            the whole GPU is the engine's (KV pool size only);
                                                                      MEM_FRACTION in the environment instead (the
                                                                      warm-up engine: 0.72, beside the idle actor)
  random_seed seed + SEED_OFFSET                                      apart from the colocated engines' seed + i
  model_path                                                          MODEL_PATH in the environment, if set (the warm-up engine of a resume: the
                                                                      checkpoint's weights, rl/export_ckpt.py); the tokenizer stays the base's
The environment is miles' engine environment for the run (compute_inference_engine_env_vars: e.g. custom all-reduce
v2 off at TP > 1 when colocated), then the spec's serving variables (the levers; async=1 turns v2 off), which win.

Usage: elastic_engine_argv.py SPEC HOST PORT BASE SEED_OFFSET
  BASE: the first of 40 consecutive free ports: nccl BASE, engine-info bootstrap BASE+1, dist-init BASE+8.
Prints "ENV k=v ...", "DIFF <json: the keys changed from the run's own engine>", "ARGV <shell-quoted argv>"."""
import json
import os
import shlex
import sys

from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.backends.sglang_utils.server_args_utils import server_args_to_argv
from miles.ray.specs.inference import compute_inference_engine_env_vars
from miles.utils.arguments import parse_args

CHANGED = {"host", "port", "nccl_port", "dist_init_addr", "engine_info_bootstrap_port", "base_gpu_id",
           "gated_launch_port", "enable_memory_saver", "mem_fraction_static", "random_seed", "model_path"}

spec_path, host, port, base, seed_offset = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
assert not host.startswith("127.") and host != "localhost", f"an engine of another job must bind a routable address, got {host}"
spec = json.load(open(spec_path))
sys.argv = ["train.py"] + spec["argv"]
args = parse_args()
tp = args.rollout_num_gpus_per_engine


def server_kwargs(**kw):
    out = _compute_server_args(args, node_rank=0, disaggregation_bootstrap_port=None, sglang_overrides={"model_path": args.hf_checkpoint},
                               num_gpus_per_engine=tp, **kw)
    return {k: v for k, v in out.items() if v is not None}


own = server_kwargs(dist_init_addr="127.0.0.1:1", nccl_port=2, host="127.0.0.1", port=3, base_gpu_id=0,
                          engine_info_bootstrap_port=4, gated_launch_port=5, random_seed=args.seed)
ext = server_kwargs(dist_init_addr=f"{host}:{base + 8}", nccl_port=base, host=host, port=port, base_gpu_id=0,
                    engine_info_bootstrap_port=base + 1, gated_launch_port=None, random_seed=args.seed + seed_offset)
ext["enable_memory_saver"] = False
ext["mem_fraction_static"] = float(os.environ.get("MEM_FRACTION", "0.85"))
if os.environ.get("MODEL_PATH"):
    ext["model_path"] = os.environ["MODEL_PATH"]
diff = {k: (own.get(k), ext.get(k)) for k in sorted(set(own) | set(ext)) if own.get(k) != ext.get(k)}
unexpected = sorted(set(diff) - CHANGED)
assert not unexpected, f"the engine would differ from the run's own engine in {unexpected}: {diff}"
must = {"enable_return_routed_experts": True, "enable_fp32_lm_head": True, "tp_size": tp, "reasoning_parser": "flint",
        "tool_call_parser": "flint", "grammar_backend": "flint", "attention_backend": "fa3", "moe_runner_backend": "triton"}
bad = {k: (ext.get(k), v) for k, v in must.items() if ext.get(k) != v}
assert not bad, f"engine argv lacks the run's serving settings: {bad}"
env = compute_inference_engine_env_vars(args) | spec.get("env", {})
print("ENV " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
print("DIFF " + json.dumps({k: list(v) for k, v in diff.items()}, default=str))
print("ARGV " + shlex.join(["python3", "-m", "sglang.launch_server", *server_args_to_argv(ext)]))
