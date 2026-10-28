"""Off-GPU check of the training schedule (rl/rl_inner.sh args): a recorded rollout (the memprobe batch: w562's 22
samples of 8 episodes, up to 145,930 tokens) through miles' rollout-side conversion and DP split as the run does it,
with the FSDP actor's TrainParallelConfig (dp 8; the fork's FSDP backend takes the rollout-side schedule): every rank
gets the same micro-batch count (w575 hung when they differed: 3 vs 2 steps), every sample one rank, one optimizer step
of 8 rollouts. Prints each rank's micro-batch tokens. Usage: schedule_check.py DUMP <train.py args...>"""
import sys
from miles.utils.arguments import parse_args
from miles.ray.rollout.debug_data import load_debug_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp
from miles.utils.dp_schedule import TrainParallelConfig

dump = sys.argv[1]
sys.argv = ["train.py"] + sys.argv[2:]
a = parse_args()
a.load_debug_rollout_data = dump
tp = TrainParallelConfig(dp_size=8, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=None,
                         independent_dp=False, supports_precomputed_schedule=True)
data, metadata = load_debug_rollout_data(a, rollout_id=0)
train = convert_samples_to_train_data(a, data, metadata=metadata, custom_convert_samples_to_train_data_func=None,
                                      custom_reward_post_process_func=None)
shards = split_train_data_by_dp(a, train, tp)
lens = [len(t) for t in train["tokens"]]
print("samples", len(lens), "rollouts", len(set(train["rollout_ids"])), "max len", max(lens))
seen = []
for r, sh in enumerate(shards):
    seen += list(sh["partition"])
    mbs = [[lens[sh["partition"][i]] for i in mb] for mb in sh["micro_batch_indices"]]
    print(f"rank {r}: num_microbatches {sh['num_microbatches']} num_rollouts {sh['num_rollouts']} "
          f"micro-batch tokens {[sum(m) for m in mbs]} ({[len(m) for m in mbs]} samples)")
ok = (len({tuple(sh["num_microbatches"]) for sh in shards}) == 1 and sorted(seen) == list(range(len(lens)))
      and all(sh["num_rollouts"] == [8] for sh in shards))
print("SCHEDULE_OK" if ok else "SCHEDULE_FAIL")
sys.exit(0 if ok else 1)
