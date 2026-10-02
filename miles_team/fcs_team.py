"""Game G1 on Frontier-CS (labs-molt-docs docs/fcs_multiagent_reward.md): propose, then synthesize.
--custom-generate-function-path miles_team.fcs_team.generate

Round 1: K = MA_FT_MATES teammates each write a complete solution from the solo prompt (the FrontierSmith prompt the
solo EasyPPO run trains on), thinking on, MA_FT_MATE_BUDGET new tokens; each program is judged (score s_j in [0, 1]
and per-case ratios s_j(c)), exactly as the solo reward judges. Round 2: the lead gets the problem plus the
teammates' extracted C++ (MA_FT_CODE_CHARS chars each, no scores or test results; a teammate cut off at its budget
shows as "no code" and cannot be adopted) and writes the final program in ```cpp```, or `<adopt j/>` to submit teammate
j's code unchanged (MA_FT_LEAD_BUDGET tokens); judged: S, S(c). MA_FT_MATES=0 is plain solo (one agent, the solo prompt).

Rewards per agent (MA_FT_REWARD; a = MA_FT_ALPHA, b = MA_FT_BETA):
  shared  mate S                                            lead S
  indiv   mate s_j                                          lead S
  mix     mate a s_j + (1-a) S                              lead S
  unique  mate mean_c max(0, s_j(c) - max_{i!=j} s_i(c)) + b S   lead S
  adopt   mate S if the lead adopted j (tag or identical code) else 0   lead S
  synth   mate a s_j + (1-a) S                              lead S - max_j s_j
Each agent is one session with one turn, so one sample; an episode always yields exactly 1 + K samples (a session
that generated nothing gets a 1-token masked stand-in), which keeps the samples per step fixed (16 prompts x 8
episodes x 4 = 512, the solo run's 16 x 32) for miles' sample-counted global batch.

Grouping (EasyPPO noise-normalized critic weights read Sample.group_index): group_index = 2 * prompt group + role,
role 0 = lead, 1 = teammate (the K teammates are exchangeable, so they share one group per prompt). Every
sample of an episode shares the episode's rollout_id; with --calculate-per-token-loss the loss is a token mean over
all samples, so the team's tokens weigh the same as solo tokens.

Eval: MA_FT_EVAL_MATES teammates (default MA_FT_MATES; a solo-trained run sets 3 to score its weights in the game);
sample.metadata["ft_mode"] == "solo" plays one agent on the solo prompt (reward s: the weights as a single
agent); otherwise the full game, returning the lead's sample with reward S. Episode stats (ft_*) ride on the lead's
sample metadata; log_rollout / log_eval average them into rollout/ft_* and eval/<set>/ft_*.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

from .fcs_rm import extract_cpp, judge_code, strip_think
from .team_rollout import Aborted, Session, _pad_sample

MATES = int(os.environ.get("MA_FT_MATES", "3"))
EVAL_MATES = int(os.environ.get("MA_FT_EVAL_MATES", str(MATES)))  # team eval of solo-trained weights: set 3 with MATES=0
MATE_BUDGET = int(os.environ.get("MA_FT_MATE_BUDGET", "16384"))
LEAD_BUDGET = int(os.environ.get("MA_FT_LEAD_BUDGET", "16384"))
CODE_CHARS = int(os.environ.get("MA_FT_CODE_CHARS", "12000"))
MAX_LEN = int(os.environ.get("MA_FT_MAX_LEN", "65536"))  # session context cap (the lead's prompt holds K programs)
REWARD = os.environ.get("MA_FT_REWARD", "shared")
ALPHA = float(os.environ.get("MA_FT_ALPHA", "0.5"))
BETA = float(os.environ.get("MA_FT_BETA", "0.0"))
THINK = os.environ.get("MA_FT_THINK", "1") == "1"
TRACE_DIR = os.environ.get("MA_FT_TRACE_DIR", "")
TRACE_EVERY = int(os.environ.get("MA_FT_TRACE_EVERY", "32"))
REWARDS = ("shared", "indiv", "mix", "unique", "adopt", "synth")
assert REWARD in REWARDS, f"MA_FT_REWARD must be one of {REWARDS}"
assert MATES >= 0

LEAD_SUFFIX = (
    "\n\nYou lead a team. {k} teammates have each written a solution to this problem independently. Their code is "
    "below; it has not been compiled, run or scored.\n\n{codes}\n\nWrite the final solution: improve, fix or combine "
    "their ideas as you see fit. Output ONLY the final C++ code wrapped in ```cpp and ```. To submit one teammate's "
    "code unchanged instead, output only <adopt j/> with j its number."
)
CODE_BLOCK = "### Teammate {j}\n```cpp\n{code}\n```"
NO_CODE = "### Teammate {j}\n(no code: the teammate's answer was cut off or contained no program)"
_ADOPT_RE = re.compile(r"<adopt\s*(?:j\s*=\s*)?\"?(\d+)\"?\s*/?>", re.IGNORECASE)
_trace_count = 0


def parse_adopt(text: str, k: int) -> int | None:
    """Teammate number from an `<adopt j/>` tag in the visible answer (after the thinking), if valid."""
    m = _ADOPT_RE.search(strip_think(text) or "")
    if not m:
        return None
    j = int(m.group(1))
    return j if 1 <= j <= k else None


def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", code or "").strip()


def unique_credit(cases: list[list[float]], j: int) -> float:
    """mean over cases of max(0, s_j(c) - max_{i != j} s_i(c)): what teammate j alone contributes."""
    mine = cases[j]
    if not mine:
        return 0.0
    others = [c for i, c in enumerate(cases) if i != j]
    tot = 0.0
    for c, v in enumerate(mine):
        best = max((o[c] for o in others if c < len(o)), default=0.0)
        tot += max(0.0, v - best)
    return tot / len(mine)


def rewards(arm: str, S: float, s: list[float], cases: list[list[float]], adopted: int | None,
            alpha: float = ALPHA, beta: float = BETA) -> tuple[float, list[float]]:
    """(lead reward, teammate rewards) for one episode; adopted is 1-based or None."""
    k = len(s)
    if arm == "shared":
        return S, [S] * k
    if arm == "indiv":
        return S, list(s)
    if arm == "mix":
        return S, [alpha * x + (1 - alpha) * S for x in s]
    if arm == "unique":
        return S, [unique_credit(cases, j) + beta * S for j in range(k)]
    if arm == "adopt":
        return S, [S if adopted == j + 1 else 0.0 for j in range(k)]
    if arm == "synth":
        return S - (max(s) if s else 0.0), [alpha * x + (1 - alpha) * S for x in s]
    raise ValueError(arm)


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


async def _agent(input, name: str, messages: list, budget: int) -> tuple[Session, str, bool]:
    sess = Session(input, name, THINK, max_len=MAX_LEN)
    text, cut = await sess.turn(messages, budget)
    return sess, text, cut


def _with_suffix(messages: list, suffix: str) -> list:
    out = deepcopy(messages)
    out[-1] = {**out[-1], "content": out[-1]["content"] + suffix}
    return out


def _bad(j_res: dict) -> float:
    return 1.0 if j_res["status"] in ("compile error", "no code") else 0.0


def _write_trace(rec: dict) -> None:
    global _trace_count
    _trace_count += 1
    if not TRACE_DIR or (_trace_count - 1) % TRACE_EVERY:
        return
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(os.path.join(TRACE_DIR, f"traces_{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    try:
        return await _play(input)
    except Aborted:
        s = deepcopy(input.sample)
        s.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=s)


async def _play(input: GenerateFnInput) -> GenerateFnOutput:
    meta = input.sample.metadata or {}
    messages = meta.get("messages")
    assert messages, "fcs_team needs the raw chat in sample.metadata['messages'] (tools/make_fcs_team_data.py)"
    label = input.sample.label
    solo_eval = input.evaluation and meta.get("ft_mode") == "solo"
    k = 0 if solo_eval else (EVAL_MATES if input.evaluation else MATES)

    if k == 0:  # plain solo: one agent on the solo prompt
        sess, text, cut = await _agent(input, "solo", messages, LEAD_BUDGET)
        res = await judge_code(label, extract_cpp(text))
        info = {"ft_solo": 1.0, "ft_lead_score": res["score"], "ft_lead_bad": _bad(res),
                "ft_tokens": sum(sg.response_length for sg in sess.segments), "ft_judge_infra_error": float(res["infra_error"])}
        return _single(input, sess, res["score"], info)

    mate_msgs = messages
    mate_runs = await asyncio.gather(*[_agent(input, f"mate{j + 1}", mate_msgs, MATE_BUDGET) for j in range(k)])
    codes = [extract_cpp(text) for _, text, _ in mate_runs]
    mate_res = await asyncio.gather(*[judge_code(label, c) for c in codes])  # judged like solo (cut -> usually 0)
    # the lead sees (and can adopt) only finished answers: a cut-off answer's "code" is FrontierSmith's fallback,
    # i.e. the raw reasoning text
    shown = ["" if cut else c for c, (_, _, cut) in zip(codes, mate_runs)]

    blocks = [CODE_BLOCK.format(j=j + 1, code=c[:CODE_CHARS]) if c else NO_CODE.format(j=j + 1)
              for j, c in enumerate(shown)]
    lead_msgs = _with_suffix(messages, LEAD_SUFFIX.format(k=k, codes="\n\n".join(blocks)))
    lead, ltext, lcut = await _agent(input, "lead", lead_msgs, LEAD_BUDGET)
    adopted = None if lcut else parse_adopt(ltext, k)
    if adopted is not None and shown[adopted - 1]:
        final = shown[adopted - 1]
    else:
        adopted = None
        final = extract_cpp(ltext)
        same = [j + 1 for j, c in enumerate(shown) if c and _norm(c) == _norm(final)]
        adopted = same[0] if same else None
    lead_res = await judge_code(label, final)

    S = lead_res["score"]
    s = [r["score"] for r in mate_res]
    cases = [r["cases"] for r in mate_res]
    r_lead, r_mates = rewards(REWARD, S, s, cases, adopted)
    corr = [c for i in range(k) for j in range(i + 1, k) if (c := _pearson(cases[i], cases[j])) is not None]
    sessions = [lead] + [m for m, _, _ in mate_runs]
    info = {
        "ft_solo": 0.0, "ft_lead_score": S, "ft_mate_score": sum(s) / k, "ft_best_mate": max(s),
        "ft_synth_gain": S - max(s), "ft_adopt": 1.0 if adopted is not None else 0.0,
        "ft_adopt_tag": 1.0 if (not lcut and parse_adopt(ltext, k)) else 0.0,
        "ft_mate_corr": sum(corr) / len(corr) if corr else float("nan"),
        "ft_lead_bad": _bad(lead_res), "ft_mate_bad": sum(_bad(r) for r in mate_res) / k,
        "ft_lead_cut": 1.0 if lcut else 0.0, "ft_mate_cut": sum(1.0 for _, _, c in mate_runs if c) / k,
        "ft_tokens": sum(sg.response_length for se in sessions for sg in se.segments),
        "ft_r_lead": r_lead, "ft_r_mate": sum(r_mates) / k,
        "ft_judge_infra_error": float(lead_res["infra_error"] or any(r["infra_error"] for r in mate_res)),
    }
    _write_trace({"time": round(time.time(), 1), "eval": input.evaluation, "arm": REWARD, "label": label,
                  "S": S, "s": s, "adopted": adopted, "r_lead": r_lead, "r_mates": r_mates,
                  "mate_status": [r["status"] for r in mate_res], "lead_status": lead_res["status"],
                  "lead_visible": (strip_think(ltext) or "")[-3000:], "codes": [c[:3000] for c in codes]})

    if input.evaluation:
        return _single(input, lead, S, info)

    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    g = input.sample.group_index if input.sample.group_index is not None else input.sample.index
    real = [sg for se in sessions for sg in se.segments if any(sg.loss_mask)]
    if not real:  # nothing generated by anyone: retry the episode
        raise Aborted("empty episode")
    samples = []
    for role, (se, r) in enumerate([(lead, r_lead)] + [(m, rm) for (m, _, _), rm in zip(mate_runs, r_mates)]):
        segs = [sg for sg in se.segments if any(sg.loss_mask)]
        seg = segs[-1] if segs else _pad_sample(real[0])  # one sample per agent (a session has one turn)
        seg.reward, seg.rollout_id = float(r), rollout_id
        seg.group_index = 2 * g + (0 if role == 0 else 1)
        seg.metadata = {**(seg.metadata or {}), "ft_role": "lead" if role == 0 else f"mate{role}"}
        samples.append(seg)
    samples[0].metadata = {**samples[0].metadata, **info}
    return GenerateFnOutput(samples=samples)


def _single(input, sess: Session, reward: float, info: dict) -> GenerateFnOutput:
    """One sample (eval, or solo training): the agent's segment with its reward and the episode stats."""
    if sess.segments:
        out = sess.segments[-1]
    else:
        out = deepcopy(input.sample)
        out.status = Sample.Status.TRUNCATED
    out.reward = float(reward)
    out.metadata = {**(out.metadata or {}), **info}
    return GenerateFnOutput(samples=out)


def _ft_means(samples) -> dict:
    acc: dict[str, list[float]] = {}
    for s in samples:
        for key, v in (s.metadata or {}).items():
            if key.startswith("ft_") and isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                acc.setdefault(key, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """--custom-rollout-log-function-path: add rollout/ft_* means to miles' own rollout log (returns False)."""
    m = {f"rollout/{k}": v for k, v in _ft_means(samples).items()}
    print(f"[fcs_team] rollout {rollout_id}: {json.dumps({k: round(v, 4) for k, v in m.items()})}", flush=True)
    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(m)
    return False


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    """--custom-eval-rollout-log-function-path: print eval/<set>/ft_* means; miles' default eval log still runs."""
    for key, d in data.items():
        samples = d.get("samples")
        if samples:
            m = {f"eval/{key}/{k}": v for k, v in _ft_means(samples).items()}
            print(f"[fcs_team] eval {rollout_id} {key}: {json.dumps({k: round(v, 4) for k, v in m.items()})}",
                  flush=True)
            if isinstance(extra_metrics, dict):
                extra_metrics.update(m)
    return False
