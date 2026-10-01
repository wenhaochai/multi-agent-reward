"""Parity of the miles EasyPPO port (branch easyppo, worktree miles-easyppo) with EasyPPO's own code.

The reference functions are executed from the EasyPPO source itself (extracted with ast from
EasyPPO/verl/trainer/ppo/core_algos.py, ray_trainer.py and verl/utils/torch_functional.py), so the
comparison is against the released code, not a re-typed copy. Identical random inputs go to both sides:
  1. critic variance weights: EasyPPO compute_prompt_variance_loss_weights vs the miles rollout-side
     _compute_critic_loss_weights (Sample.group_index as the prompt uid);
  2. weighted value loss: EasyPPO compute_value_loss (0.5 * token-mean, loss_weights) vs the miles critic path
     (loss.loss_function -> value_loss_function, --calculate-per-token-loss, --value-loss-scale 0.5), divided by
     the token normalizer Megatron applies;
  3. actor loss with overlong rows masked: EasyPPO compute_policy_loss_vanilla (clip 0.2/0.28, dual clip 3.0) over a
     response mask with mask_overlong_response_rows applied vs miles apply_actor_overlong_mask +
     compute_policy_loss through loss.loss_function's per-token normalization.
Run inside miles.sif with the worktree on PYTHONPATH:
  PYTHONPATH=<miles-easyppo>:/root/Megatron-LM python3 tests/test_easyppo_parity.py
"""
import ast
import sys
import types
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

EASYPPO = Path("/scratch/gpfs/GROUP/USER/project/EasyPPO/verl")


def load_functions(path: Path, names: list[str], namespace: dict) -> dict:
    """Execute the named top-level functions of `path` (decorators dropped) in `namespace`."""
    src = path.read_text()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(found)
    assert not missing, f"{missing} not in {path}"
    for name in names:
        exec(found[name], namespace)
    return namespace


# ---- EasyPPO reference ------------------------------------------------------------------------------------------
verl_F = types.SimpleNamespace(
    **{
        k: v
        for k, v in load_functions(
            EASYPPO / "utils/torch_functional.py",
            ["clip_by_value", "masked_sum", "masked_mean"],
            {"torch": torch},
        ).items()
        if k in ("clip_by_value", "masked_sum", "masked_mean")
    }
)


class AlgoConfig:  # only for compute_policy_loss_vanilla's isinstance assert
    pass


REF = load_functions(
    EASYPPO / "trainer/ppo/core_algos.py",
    ["compute_prompt_variance_loss_weights", "agg_loss", "compute_value_loss", "compute_policy_loss_vanilla"],
    {"torch": torch, "np": np, "defaultdict": defaultdict, "Any": Any, "Optional": Optional, "verl_F": verl_F,
     "AlgoConfig": AlgoConfig, "ActorConfig": object},
)
REF["DataProto"] = object  # annotation only
load_functions(EASYPPO / "trainer/ppo/ray_trainer.py", ["mask_overlong_response_rows"], REF)


class ActorCfg(dict):
    """The fields compute_policy_loss_vanilla reads from verl's ActorConfig (easyppo_aime.yaml values)."""

    clip_ratio, clip_ratio_low, clip_ratio_high = 0.2, 0.2, 0.28
    global_batch_info: dict = {}

    def __init__(self):
        super().__init__(clip_ratio_c=3.0)


# ---- miles side ---------------------------------------------------------------------------------------------------
import miles.backends.training_utils.cp_utils as cp_utils  # noqa: E402
import miles.backends.training_utils.loss as miles_loss  # noqa: E402
import miles.backends.training_utils.loss_hub.losses as miles_losses  # noqa: E402
from miles.backends.training_utils.loss_hub.easyppo import apply_actor_overlong_mask  # noqa: E402
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss  # noqa: E402
from miles.ray.rollout.train_data_conversion import _compute_critic_loss_weights  # noqa: E402

_PS = types.SimpleNamespace(cp=types.SimpleNamespace(size=1), intra_dp=types.SimpleNamespace(size=1),
                            intra_dp_cp=types.SimpleNamespace(size=1), is_ulysses_cp=False)
cp_utils.get_parallel_state = lambda: _PS
miles_loss.get_parallel_state = lambda: _PS


def miles_args(**kw) -> Namespace:
    base = dict(
        calculate_per_token_loss=True, qkv_format="thd", recompute_loss_function=False,
        use_dynamic_global_batch_size=False, global_batch_size=0, value_clip=0.2, critic_variance_weighted_loss=False,
        value_loss_scale=1.0, loss_type="value_loss", true_on_policy_mode=False, actor_only_overlong_filter=False,
        allgather_cp=False, multi_lora=False, rollout_max_response_len=None, critic_variance_weight_beta=0.5,
        critic_variance_weight_min=0.25, critic_variance_weight_max=None,
    )
    base.update(kw)
    return Namespace(**base)


def make_batch(gen, n_prompts=4, n_per=4, max_len=12):
    """Ragged responses; EasyPPO-style padded matrices plus miles per-sample lists of the same numbers."""
    B = n_prompts * n_per
    lens = torch.randint(2, max_len + 1, (B,), generator=gen)
    lens[0] = max_len  # at least one overlong row
    lens[5] = max_len
    mask = torch.zeros(B, max_len)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    rewards = torch.rand(B, generator=gen).round(decimals=1)
    rewards[n_per : 2 * n_per] = 0.7  # a zero-variance prompt hits the w_min floor
    return B, lens, mask, rewards, [i // n_per for i in range(B)]


def test_variance_weights(gen):
    B, lens, mask, rewards, uids = make_batch(gen)
    ref_w, ref_var = REF["compute_prompt_variance_loss_weights"](
        sequence_rewards=rewards, sample_uids=np.asarray(uids, dtype=object), beta=0.5, w_min=0.25,
        valid_mask=mask.bool().any(-1), w_max=None,
    )
    samples = [types.SimpleNamespace(group_index=u) for u in uids]
    loss_masks = [[1] * int(l) for l in lens]
    got = torch.tensor(_compute_critic_loss_weights(miles_args(), samples, rewards.tolist(), loss_masks))
    assert torch.allclose(got, ref_w, atol=1e-6), (got, ref_w)
    # clipped variant
    ref_w2, _ = REF["compute_prompt_variance_loss_weights"](
        sequence_rewards=rewards, sample_uids=np.asarray(uids, dtype=object), beta=0.5, w_min=0.25,
        valid_mask=mask.bool().any(-1), w_max=1.5,
    )
    got2 = torch.tensor(_compute_critic_loss_weights(miles_args(critic_variance_weight_max=1.5), samples,
                                                     rewards.tolist(), loss_masks))
    assert torch.allclose(got2, ref_w2, atol=1e-6)
    return float((got - ref_w).abs().max())


def test_value_loss(gen):
    B, lens, mask, rewards, uids = make_batch(gen)
    T = mask.shape[1]
    vpreds = torch.randn(B, T, generator=gen)
    old = vpreds + 0.3 * torch.randn(B, T, generator=gen)  # some cross the 0.2 value clip
    returns = rewards[:, None].expand(B, T).clone()
    w, _ = REF["compute_prompt_variance_loss_weights"](
        sequence_rewards=rewards, sample_uids=np.asarray(uids, dtype=object), beta=0.5, w_min=0.25,
        valid_mask=mask.bool().any(-1), w_max=None,
    )
    ref, _ = REF["compute_value_loss"](vpreds=vpreds, returns=returns, values=old, response_mask=mask,
                                       cliprange_value=0.2, loss_agg_mode="token-mean", loss_weights=w)

    rl = [int(l) for l in lens]
    miles_losses.get_values = lambda logits, **kw: {"values": [vpreds[i, :l] for i, l in enumerate(rl)]}
    batch = {
        "values": [old[i, :l] for i, l in enumerate(rl)],
        "returns": [returns[i, :l] for i, l in enumerate(rl)],
        "loss_masks": [mask[i, :l] for i, l in enumerate(rl)],
        "critic_loss_weights": w.tolist(),
        "unconcat_tokens": None, "total_lengths": [l + 3 for l in rl], "response_lengths": rl,
    }
    args = miles_args(critic_variance_weighted_loss=True, value_loss_scale=0.5)
    miles_loss.get_loss_function = miles_losses.get_loss_function  # test_actor_overlong swaps it
    loss, num_tokens, _ = miles_loss.loss_function(args, batch, 1, torch.zeros(1), num_rollouts=B)
    got = loss / num_tokens  # Megatron's per-token normalization
    assert torch.allclose(got, ref, atol=1e-6), (got, ref)
    return float((got - ref).abs())


def test_actor_overlong(gen):
    B, lens, mask, rewards, uids = make_batch(gen)
    T = mask.shape[1]
    old_lp = -torch.rand(B, T, generator=gen) * 3
    new_lp = old_lp + 0.4 * torch.randn(B, T, generator=gen)  # ratios on both sides of 0.8 / 1.28 / dual clip 3
    adv = torch.randn(B, T, generator=gen) * 2

    data = types.SimpleNamespace(batch={"response_mask": mask.clone(), "overlong_filtered": (lens == T)})
    REF["mask_overlong_response_rows"](data)
    ref, _ = REF["compute_policy_loss_vanilla"](old_log_prob=old_lp, log_prob=new_lp, advantages=adv,
                                                response_mask=data.batch["response_mask"],
                                                loss_agg_mode="token-mean", config=ActorCfg())

    rl = [int(l) for l in lens]
    rollout_data = {"response_lengths": rl, "loss_masks": [mask[i, :l].clone() for i, l in enumerate(rl)],
                    "truncated": [0] * B}
    args = miles_args(actor_only_overlong_filter=True, rollout_max_response_len=T, loss_type="policy_loss")
    n = apply_actor_overlong_mask(args, rollout_data)
    assert n == int((lens == T).sum())

    def pg_only(args, batch, logits, sum_of_sample_mean):
        ppo_kl = torch.cat([old_lp[i, :l] - new_lp[i, :l] for i, l in enumerate(rl)])  # miles: old - new
        a = torch.cat([adv[i, :l] for i, l in enumerate(rl)])
        pg, _ = compute_policy_loss(ppo_kl, a, 0.2, 0.28, 3.0)
        return sum_of_sample_mean(pg), {}

    miles_loss.get_loss_function = lambda args, loss_fn=None: pg_only
    batch = {"loss_masks": rollout_data["loss_masks"], "total_lengths": [l + 3 for l in rl], "response_lengths": rl}
    loss, num_tokens, _ = miles_loss.loss_function(args, batch, 1, torch.zeros(1), num_rollouts=B)
    got = loss / num_tokens
    assert torch.allclose(got, ref, atol=1e-6), (got, ref)
    # without the filter the overlong rows would count: the two must differ on this batch
    full, _ = REF["compute_policy_loss_vanilla"](old_log_prob=old_lp, log_prob=new_lp, advantages=adv,
                                                 response_mask=mask, loss_agg_mode="token-mean", config=ActorCfg())
    assert not torch.allclose(full, ref)
    return float((got - ref).abs())


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    worst = defaultdict(float)
    for trial in range(50):
        worst["variance_weights"] = max(worst["variance_weights"], test_variance_weights(gen))
        worst["value_loss"] = max(worst["value_loss"], test_value_loss(gen))
        worst["actor_overlong"] = max(worst["actor_overlong"], test_actor_overlong(gen))
    for k, v in worst.items():
        print(f"PASS {k}: 50 random batches, max |miles - EasyPPO| = {v:.2e}")
    sys.exit(0)
