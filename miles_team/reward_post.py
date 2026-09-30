"""--custom-reward-post-process-path miles_team.reward_post.flash_reinforce

molt's FlashREINFORCE baseline (molt/trainer/algorithm/advantage.py flash_reinforce): A_i = R_i - mean(R) over every
sample of the rollout batch (one session segment = one sample; n_samples_per_prompt = 1 has no prompt group), no
whitening. Under --advantage-estimator grpo miles broadcasts the returned value over the sample's tokens, so this is
the advantage. The team rollout's padding samples (metadata tm_pad) are left out of the mean and get 0.
Returning per-sample values also lets teammate segments keep R - collab while the lead keeps R (miles' own normalizer
insists that every sibling of a compact rollout share one reward).
"""


def flash_reinforce(args, samples):
    raw = [float(s.get_reward_value(args)) for s in samples]
    real = [r for r, s in zip(raw, samples) if not (s.metadata or {}).get("tm_pad")]
    mean = sum(real) / len(real) if real else 0.0
    adv = [0.0 if (s.metadata or {}).get("tm_pad") else r - mean for r, s in zip(raw, samples)]
    return raw, adv
