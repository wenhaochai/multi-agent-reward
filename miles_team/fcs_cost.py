"""Cost-steered team reward for game G1 (labs-molt-docs docs/fcs_multiagent_reward.md, "Cost-steered rewards"; after
https://anishlk.com/swe-2-extended/, which extends Cognition's SWE-2 reward R = S - lambda^(e) C).

Effort e = team size K, drawn per episode from MA_FT_K_SET (fcs_team.py). Cost C = tokens the episode generated (all
agents). Every real sample of the episode gets R = r - lambda_K C / c_K0, where r is the arm's reward (fcs_team.rewards).
pi_0 is the actor of the critic-only rollouts (the actor does not move there): s_K0 and c_K0 are the per-K means of
the lead score S and of C over those rollouts. After them, once per rollout, with EMAs of the per-K means:
    u_K = (ema_s_K - s_K0) / s_K0,  v_K = (c_K0 - ema_c_K) / c_K0,
    log lambda_K += eta [(1 - alpha) u_K - alpha v_K],  lambda_K clamped to [LAM_MIN, LAM_MAX].
alpha = 1 raises the score with the cost held at pi_0's level; alpha = 0 cuts the cost with the score held.

Hooks (both run in miles' RolloutExecutor, in this order, once per training rollout):
  observe(rollout_id, args, samples)  from fcs_team.log_rollout: stores this rollout's per-K sums and returns the
                                      fc/* metrics.
  apply_cost(samples, rewards)        from fcs_team.post_process (--custom-reward-post-process-path), after the arm's
                                      relabel: the cost term (MA_FC_ON=1) or the rewards unchanged (MA_FC_ON=0: the
                                      mixed-K control, which still logs fc/*).
lambda is never stored: it is replayed from the per-rollout sums (MA_FC_STATE json), so a run resumed from an older
checkpoint recomputes exactly the lambdas its rollouts used (sums of rollouts it regenerates are overwritten).
"""
from __future__ import annotations

import json
import math
import os

ON = os.environ.get("MA_FC_ON", "0") == "1"
ALPHA = float(os.environ.get("MA_FC_ALPHA", "1.0"))
ETA = float(os.environ.get("MA_FC_ETA", "0.5"))
LAM0 = float(os.environ.get("MA_FC_LAMBDA0", "0.02"))
EMA = float(os.environ.get("MA_FC_EMA", "0.7"))
LAM_MIN, LAM_MAX = 1e-3, 1.0
S0_FLOOR = 1e-3
STATE = os.environ.get("MA_FC_STATE", "")

_obs: dict[int, dict[int, list[float]]] = {}  # rollout id -> {K: [sum S, sum C, episodes]}
_cur: dict | None = None  # replay result for the rollout being converted


def _load() -> None:
    if STATE and os.path.exists(STATE):
        with open(STATE) as f:
            raw = json.load(f)["obs"]
        _obs.update({int(r): {int(k): v for k, v in d.items()} for r, d in raw.items()})


def _save() -> None:
    if not STATE:
        return
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE + ".new", "w") as f:
        json.dump({"obs": _obs}, f)
    os.replace(STATE + ".new", STATE)


_load()


def episodes(samples) -> list[tuple[int, float, float]]:
    """(K, lead score S, episode tokens C) per episode: one lead sample per episode carries fc_lead."""
    out = []
    for s in samples:
        m = s.metadata or {}
        if m.get("fc_lead") and not m.get("fc_pad"):
            out.append((int(m["fc_k"]), float(m["fc_score"]), float(m["fc_cost"])))
    return out


def replay(obs: dict, upto: int, critic_only: int, alpha: float = ALPHA, eta: float = ETA, lam0: float = LAM0,
           ema: float = EMA) -> dict:
    """lambda_K, pi_0 means and EMAs after rollouts 0..upto. Rollouts < critic_only define pi_0 (and keep lam0)."""
    acc0: dict[int, list[float]] = {}
    lam: dict[int, float] = {}
    es: dict[int, float] = {}
    ec: dict[int, float] = {}
    uv: dict[int, tuple[float, float]] = {}
    for r in sorted(x for x in obs if x <= upto):
        for k, (ss, cc, n) in sorted(obs[r].items()):
            lam.setdefault(k, lam0)
            if r < critic_only or k not in acc0:
                a = acc0.setdefault(k, [0.0, 0.0, 0.0])
                a[0], a[1], a[2] = a[0] + ss, a[1] + cc, a[2] + n
                continue
            s0, c0 = acc0[k][0] / acc0[k][2], acc0[k][1] / acc0[k][2]
            es[k] = ema * es.get(k, s0) + (1 - ema) * ss / n
            ec[k] = ema * ec.get(k, c0) + (1 - ema) * cc / n
            u, v = (es[k] - s0) / max(s0, S0_FLOOR), (c0 - ec[k]) / c0
            step = math.log(lam[k]) + eta * ((1 - alpha) * u - alpha * v)
            lam[k] = math.exp(min(math.log(LAM_MAX), max(math.log(LAM_MIN), step)))  # clamp in log space: no overflow
            uv[k] = (u, v)
    return {"lam": lam, "s0": {k: a[0] / a[2] for k, a in acc0.items()},
            "c0": {k: a[1] / a[2] for k, a in acc0.items()}, "ema_s": es, "ema_c": ec, "uv": uv}


def observe(rollout_id: int, args, samples) -> dict:
    """Store this rollout's per-K sums, replay lambda through it, return fc/* metrics."""
    global _cur
    d: dict[int, list[float]] = {}
    for k, S, C in episodes(samples):
        a = d.setdefault(k, [0.0, 0.0, 0.0])
        a[0], a[1], a[2] = a[0] + S, a[1] + C, a[2] + 1
    for r in [r for r in _obs if r >= rollout_id]:  # regenerated after a resume: the old sums are stale
        del _obs[r]
    _obs[rollout_id] = d
    _save()
    _cur = replay(_obs, rollout_id, args.num_critic_only_steps or 0)
    m = {}
    for k, (ss, cc, n) in d.items():
        m |= {f"fc/s_K{k}": ss / n, f"fc/c_K{k}": cc / n, f"fc/n_K{k}": n, f"fc/lambda_K{k}": _cur["lam"][k],
              f"fc/s0_K{k}": _cur["s0"][k], f"fc/c0_K{k}": _cur["c0"][k]}
        if k in _cur["uv"]:
            m |= {f"fc/u_K{k}": _cur["uv"][k][0], f"fc/v_K{k}": _cur["uv"][k][1]}
    return m


def apply_cost(samples, rewards: list[float]) -> list[float]:
    """rewards (the arm's, per sample) with the cost term on every real sample if ON (fcs_team.post_process)."""
    out = []
    for s, r in zip(samples, rewards):
        m = s.metadata or {}
        if ON and not m.get("fc_pad") and "fc_k" in m:
            assert _cur is not None, "fcs_cost.observe must run first (fcs_team.log_rollout)"
            k = int(m["fc_k"])
            r -= _cur["lam"][k] * float(m["fc_cost"]) / _cur["c0"][k]
        out.append(r)
    return out


def post_process(args, samples):
    """(raw, rewards) with the cost term, without relabeling (kept for direct use; runs use fcs_team.post_process)."""
    out = apply_cost(samples, [float(s.get_reward_value(args)) for s in samples])
    return out, list(out)
