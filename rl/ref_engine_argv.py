"""The SGLang launch line of the reference server of a kl=B run (rl/rl_inner.sh start_ref_engine): the original Flint
on the run's own engine GPUs, beside the rollout engine, scoring every sample's response tokens for the KL loss
(flint/ref_scorer.py). Runs in miles-v0.1.3.sif with the run's miles on the path.

The arguments are the run's own engine's (miles' _compute_server_args on the trainer's argv in SPEC, as
rl/elastic_engine_argv.py), so the reference scores with the rollout engines' numerics (fa3 attention, Triton MoE,
the fp32 LM head, the levers' environment), with only these changes:
  host 127.0.0.1, port, nccl_port, dist_init_addr, engine_info_bootstrap_port   its own addresses, on this node only
  base_gpu_id 0, gated_launch_port None, enable_memory_saver False              its own GPUs, nothing gates or frees it
  random_seed seed + 7                                                          (it never samples for training)
  model_path                                                                    the original Flint (the spec's hf_checkpoint)
  mem_fraction_static REF_MEM_FRACTION, max_total_tokens REF_MAX_TOKENS         a small KV pool: one sample at a time
  swa_full_tokens_ratio 1.0                                                     a sample's sliding-window KV fits beside its full KV
  max_running_requests 1                                                        one prefill at a time (each a 4096-token chunk)
  cuda_graph_backend_decode / _prefill disabled, cuda_graph_bs_decode None      prefill only: no graph memory
  enable_return_routed_experts False                                            nothing to replay
The model path must be the original checkpoint: a reference to anything else would be a KL to the wrong model.

Usage: ref_engine_argv.py SPEC PORT BASE  (BASE: the first of 10 free ports: nccl BASE, bootstrap BASE+1, dist-init BASE+8)
Prints "ENV k=v ...", "DIFF <json>", "ARGV <shell-quoted argv>"."""
import json
import os
import shlex
import sys

from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.backends.sglang_utils.server_args_utils import server_args_to_argv
from miles.ray.specs.inference import compute_inference_engine_env_vars
from miles.utils.arguments import parse_args

CHANGED = {"host", "port", "nccl_port", "dist_init_addr", "engine_info_bootstrap_port", "base_gpu_id",
           "gated_launch_port", "enable_memory_saver", "mem_fraction_static", "random_seed", "max_total_tokens",
           "swa_full_tokens_ratio", "max_running_requests", "cuda_graph_backend_decode", "cuda_graph_backend_prefill",
           "cuda_graph_bs_decode", "enable_return_routed_experts"}

spec_path, port, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
spec = json.load(open(spec_path))
sys.argv = ["train.py"] + spec["argv"]
args = parse_args()
tp = args.rollout_num_gpus_per_engine
assert os.path.realpath(args.hf_checkpoint) == os.path.realpath(spec["flint"]), (args.hf_checkpoint, spec["flint"])


def server_kwargs(**kw):
    out = _compute_server_args(args, node_rank=0, disaggregation_bootstrap_port=None,
                               sglang_overrides={"model_path": args.hf_checkpoint}, num_gpus_per_engine=tp, **kw)
    return {k: v for k, v in out.items() if v is not None}


own = server_kwargs(dist_init_addr="127.0.0.1:1", nccl_port=2, host="127.0.0.1", port=3, base_gpu_id=0,
                    engine_info_bootstrap_port=4, gated_launch_port=5, random_seed=args.seed)
ref = server_kwargs(dist_init_addr=f"127.0.0.1:{base + 8}", nccl_port=base, host="127.0.0.1", port=port, base_gpu_id=0,
                    engine_info_bootstrap_port=base + 1, gated_launch_port=None, random_seed=args.seed + 7)
ref.update(enable_memory_saver=False, mem_fraction_static=float(os.environ.get("REF_MEM_FRACTION", "0.35")),
           max_total_tokens=int(os.environ.get("REF_MAX_TOKENS", "160000")), swa_full_tokens_ratio=1.0,
           max_running_requests=1, cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
           enable_return_routed_experts=False)
ref.pop("cuda_graph_bs_decode", None)
diff = {k: (own.get(k), ref.get(k)) for k in sorted(set(own) | set(ref)) if own.get(k) != ref.get(k)}
unexpected = sorted(set(diff) - CHANGED)
assert not unexpected, f"the reference server would differ from the run's own engine in {unexpected}: {diff}"
must = {"enable_fp32_lm_head": True, "tp_size": tp, "attention_backend": "fa3", "moe_runner_backend": "triton",
        "model_path": args.hf_checkpoint, "chunked_prefill_size": own.get("chunked_prefill_size")}
bad = {k: (ref.get(k), v) for k, v in must.items() if ref.get(k) != v}
assert not bad, f"reference argv lacks the run's serving settings: {bad}"
env = compute_inference_engine_env_vars(args) | spec.get("env", {})
print("ENV " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
print("DIFF " + json.dumps({k: list(v) for k, v in diff.items()}, default=str))
print("ARGV " + shlex.join(["python3", "-m", "sglang.launch_server", *server_args_to_argv(ref)]))
