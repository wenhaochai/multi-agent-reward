"""Frontier-CS games with an oracle (labs-molt-docs docs/fcs_multiagent_reward.md, user decisions 2026-10-03):
every submission is judged and its total score (0-100) and per-test-case scores come back to the agent that submitted
it or to its lead. An episode keeps its best submission: V = max over its submissions.
--custom-generate-function-path miles_team.fcs_oracle.generate

Games (MA_FO_GAME for training; an eval set picks its game with metadata.fo_game), each with 5 submissions:
  team  1 lead + 4 subagents, after DeepSeek-V4.1-Flash's Agent Team mode (arXiv 2609.19969 sec. 5.3.5: the lead
        delegates a task to each teammate, teammates start fresh, the lead reviews the work and produces the final
        answer). Turn 1: the lead reads the problem and writes one task per subagent (<task j>...</task>, at most
        MA_FO_PLAN_BUDGET tokens). The subagents (fresh sessions: the problem plus their task) each submit a program.
        Turn 2: the lead sees every subagent's task, code, total and per-case scores, and submits the final program
        (or <adopt j/>). Submissions: 4 subagents + the lead's final.
  seq   one agent, MA_FO_ROUNDS rounds in one chat: round t sees its earlier programs with their scores and submits
        a new one (sequential self-revision with the same feedback).
  par   MA_FO_ROUNDS independent solo attempts (best-of-n).

Rewards (MA_FO_REWARD; s = a submission's score in [0, 1], V = the episode's best):
          team: plan / final / subagent j                     seq: round t                par: attempt t
  shared  V / V / V                                            V                           V
  indiv   max_j s_j / S / s_j                                  s_t                         s_t
  gain    V / S - max_j s_j / V                                -                           -
  diff    V / max(0, S - max_j s_j) / V - max(S, s_-j)         max(0, s_t - max_{i<t} s_i)  V - max_{i!=t} s_i
          (team diff: S leaves the counterfactual of a subagent the lead adopted)
indiv on par is plain single-agent RL (each attempt its own score); shared on seq is the sequential baseline.

Training returns one sample per agent turn (a turn that string-extends its session's previous turn shares that
segment, which then carries the later turn's reward), padded with masked stand-ins to a fixed count per episode (team 6, seq/par MA_FO_ROUNDS), so the samples
per step stay fixed. Critic groups: 8 g + role (team: plan 0, final 1, subagent 2; seq: round t; par: 0; padding 7).
A training episode in which any judge call failed for infrastructure reasons is aborted (miles resubmits the group).
Every real sample carries fo = the episode record (game, scores, role) so post_process recomputes the arm's reward
(identity live; relabels rollouts replayed by miles --replay-rollout-data). Eval returns the episode's last
submission's sample with reward V. Latency = generated tokens on the critical path (team: plan + the longest
subagent + final; seq: sum; par: max), as DeepSeek's derived latency without the prefill and tool terms.
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

GAME = os.environ.get("MA_FO_GAME", "team")
REWARD = os.environ.get("MA_FO_REWARD", "shared")
SUBS = int(os.environ.get("MA_FO_SUBS", "4"))
ROUNDS = int(os.environ.get("MA_FO_ROUNDS", "5"))
BUDGET = int(os.environ.get("MA_FO_BUDGET", "32768"))
PLAN_BUDGET = int(os.environ.get("MA_FO_PLAN_BUDGET", "8192"))
CODE_CHARS = int(os.environ.get("MA_FO_CODE_CHARS", "8000"))
MAX_CASES = int(os.environ.get("MA_FO_MAX_CASES", "100"))
MAX_LEN = int(os.environ.get("MA_FO_MAX_LEN", "65536"))
THINK = os.environ.get("MA_FO_THINK", "1") == "1"
TRACE_DIR = os.environ.get("MA_FO_TRACE_DIR", "")
TRACE_EVERY = int(os.environ.get("MA_FO_TRACE_EVERY", "32"))
GAMES = ("team", "seq", "par")
ARMS = {"team": ("shared", "indiv", "gain", "diff"), "seq": ("shared", "indiv", "diff"),
        "par": ("shared", "indiv", "diff")}
assert GAME in GAMES and REWARD in ARMS[GAME], (GAME, REWARD)

PLAN = (
    "\n\nYou lead a team of {n} subagents. Each subagent gets this problem plus one task you write, and submits a "
    "complete C++ solution. Every submission is judged: you will see each subagent's code with its total score "
    "(0-100) and its per-test-case scores, and then you write the team's final solution. The team keeps its "
    "best-scoring submission. Write the {n} tasks now: the approach each subagent should take and what to watch out "
    "for. Output exactly {n} blocks, <task 1>...</task> through <task {n}>...</task>. Do not write the solution yet."
)
SUB = "\n\nYour team lead gave you this task:\n{task}\n\nFollow it. Output ONLY the C++ code wrapped in ```cpp and ```."
NO_TASK = "(no task was given: solve the problem your own way)"
FEEDBACK = (
    "The subagents' submissions were judged (total score 0-100; per-test-case scores are fractions of full marks)."
    "\n\n{blocks}\n\nThe team keeps its best submission, currently {best:.2f}/100. Write the final solution: improve "
    "on, fix or combine theirs to score higher. Output ONLY the final C++ code wrapped in ```cpp and ```, or output "
    "only <adopt j/> to resubmit subagent j's code unchanged."
)
SUB_BLOCK = "### Subagent {j}\nTask: {task}\nResult: {result}\nPer-test-case scores: {cases}\n```cpp\n{code}\n```"
SUB_NONE = "### Subagent {j}\nTask: {task}\nResult: no code (the answer was cut off or contained no program), score 0"
REVISE = (
    "Your submission was judged: {result}\nPer-test-case scores (fractions of full marks): {cases}\nYour best score "
    "so far is {best:.2f}/100. Write an improved solution. Output ONLY the C++ code wrapped in ```cpp and ```."
)
_SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")  # Session decodes with special tokens kept
_TASK_RE = re.compile(r"<task\s*(\d+)\s*>(.*?)</task\s*>", re.IGNORECASE | re.DOTALL)
_ADOPT_RE = re.compile(r"<adopt\s*(?:j\s*=\s*)?\"?(\d+)\"?\s*/?>", re.IGNORECASE)
_trace_count = 0


def parse_tasks(text: str, n: int) -> list[str]:
    """The lead's tasks from its visible answer; missing ones become NO_TASK."""
    tasks = {}
    for m in _TASK_RE.finditer(strip_think(text) or ""):
        j = int(m.group(1))
        if 1 <= j <= n and j not in tasks and m.group(2).strip():
            tasks[j] = m.group(2).strip()
    return [tasks.get(j, NO_TASK) for j in range(1, n + 1)]


def parse_adopt(text: str, k: int) -> int | None:
    m = _ADOPT_RE.search(strip_think(text) or "")
    if m and 1 <= int(m.group(1)) <= k:
        return int(m.group(1))
    return None


def fmt_cases(cases: list[float]) -> str:
    s = " ".join(f"{c:.2f}" for c in cases[:MAX_CASES])
    return s + (f" ... ({len(cases)} cases)" if len(cases) > MAX_CASES else "") if cases else "(none)"


_PATH_RE = re.compile(r"^\S*?sol\.cpp:")


def compile_errors(msg: str) -> str:
    """Every error of a g++/ld run, as an online judge shows it: each `error:` line (path shortened to sol.cpp) with
    the source line and caret g++ prints under it; `note:` lines and template-instantiation context are dropped (the
    whole output reaches 133k tokens, the errors alone at most ~1.5k on the SFT model's programs)."""
    out, lines = [], (msg or "").splitlines()
    for i, l in enumerate(lines):
        if " error: " in l or "fatal error:" in l or "undefined reference" in l:
            out.append(_PATH_RE.sub("sol.cpp:", l))
            for nxt in lines[i + 1:i + 3]:  # "   12 |     code" and "      |     ^~~"
                if re.match(r"^\s*\d*\s*\|", nxt):
                    out.append(nxt)
                else:
                    break
    return "\n".join(out)


def fmt_result(res: dict) -> str:
    head = f"{res['status']}, score {100 * res['score']:.2f}/100"
    if res["status"] == "compile error":
        errs = compile_errors(res.get("msg", ""))
        return head + ("\nCompiler errors:\n" + errs if errs else "")
    return head


def rewards(game: str, arm: str, rec: dict) -> list[float]:
    """Rewards in the order of rec's submissions. team: [plan, final, sub_1..sub_n]; seq/par: one per round."""
    s = rec["s"]
    if game == "team":
        S, best = rec["S"], max(s)
        V = max(S, best)
        if arm == "shared":
            return [V, V] + [V] * len(s)
        if arm == "indiv":
            return [best, S] + list(s)
        if arm == "gain":
            return [V, S - best] + [V] * len(s)
        if arm == "diff":  # without j the lead could not have adopted j's code, so S leaves j's counterfactual then
            adopted = rec.get("adopted")
            return [V, max(0.0, S - best)] + [V - max(([] if adopted == j + 1 else [S]) + s[:j] + s[j + 1:])
                                              for j in range(len(s))]
    else:
        V = max(s)
        if arm == "shared":
            return [V] * len(s)
        if arm == "indiv":
            return list(s)
        if arm == "diff" and game == "seq":
            return [max(0.0, s[t] - max(s[:t], default=0.0)) for t in range(len(s))]
        if arm == "diff" and game == "par":
            return [V - max(s[:t] + s[t + 1:], default=0.0) for t in range(len(s))]
    raise ValueError((game, arm))


def relabel(fo: dict, arm: str | None = None) -> float:
    return rewards(fo["game"], arm or REWARD, fo)[fo["role"]]


def _bad(res: dict) -> float:
    return 1.0 if res["status"] in ("compile error", "no code") else 0.0


def _gen(sess: Session) -> int:
    return sum(sum(sg.loss_mask) for sg in sess.segments)


def _write_trace(rec: dict) -> None:
    global _trace_count
    _trace_count += 1
    if not TRACE_DIR or (_trace_count - 1) % TRACE_EVERY:
        return
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(os.path.join(TRACE_DIR, f"traces_{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _with_suffix(messages: list, suffix: str) -> list:
    out = deepcopy(messages)
    out[-1] = {**out[-1], "content": out[-1]["content"] + suffix}
    return out


async def _turn(sess: Session, messages: list, budget: int) -> tuple[str, bool, int]:
    """(text, cut, index of the segment this turn generated into, or -1 when it generated nothing)."""
    before = _gen(sess)
    text, cut = await sess.turn(messages, budget)
    for tok in _SPECIAL:  # e.g. the closing <|im_end|> would otherwise end up in fence-less code and in later prompts
        text = text.replace(tok, "")
    return text, cut, (len(sess.segments) - 1 if _gen(sess) > before else -1)


def _tt(role: str, text: str, cut: bool, res: dict | None, **extra) -> dict:
    """One turn for the trace file: the full text (reasoning included), whether it was cut, and how it was judged."""
    return {"role": role, "cut": bool(cut), "closed_think": "</think>" in (text or ""),
            "status": res["status"] if res else None, "score": res["score"] if res else None,
            "text": text or "", **extra}


def _visible(text: str, cut: bool) -> str:
    """What an agent's earlier turn shows in later context: its answer without the reasoning."""
    return "(cut off: the answer hit the token limit)" if cut else (strip_think(text) or "").strip() or "(empty)"


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
    assert messages, "fcs_oracle needs the raw chat in sample.metadata['messages'] (tools/make_fcs_team_data.py)"
    game = meta.get("fo_game", GAME) if input.evaluation else GAME
    label = input.sample.label
    play = {"team": _team, "seq": _seq, "par": _par}[game]
    turns, rec, info, texts = await play(input, messages, label)
    if info.get("fo_judge_infra_error") and not input.evaluation:
        # a submission could not be judged (fork / disk / sandbox): its 0 is not the policy's, so do not train on the
        # episode; miles resubmits the group (eval keeps the 0, flagged in fo_judge_infra_error)
        raise Aborted("judge infrastructure error")
    V = max(rec["s"] + ([rec["S"]] if game == "team" else []))
    info = {**info, "fo_V": V, "fo_game_" + game: 1.0}
    _write_trace({"time": round(time.time(), 1), "eval": input.evaluation, "game": game, "arm": REWARD,
                  "label": label, **rec, "V": V, "turns": texts})
    if input.evaluation:  # the episode's last submission (team: the lead's final), scored V
        sess, seg_i = turns[1] if game == "team" else turns[-1]
        if seg_i >= 0:
            out = sess.segments[seg_i]
        else:
            out = deepcopy(input.sample)
            out.status = Sample.Status.TRUNCATED
        out.reward = float(V)
        out.metadata = {**(out.metadata or {}), **info}
        return GenerateFnOutput(samples=out)
    return _pack(input, turns, game, rec, info)


async def _team(input, messages, label):
    n = SUBS
    lead = Session(input, "lead", THINK, max_len=MAX_LEN)
    plan_msgs = _with_suffix(messages, PLAN.format(n=n))
    ptext, pcut, pseg = await _turn(lead, plan_msgs, PLAN_BUDGET)
    tasks = parse_tasks("" if pcut else ptext, n)
    subs = [Session(input, f"sub{j + 1}", THINK, max_len=MAX_LEN) for j in range(n)]
    runs = await asyncio.gather(*[_turn(subs[j], _with_suffix(messages, SUB.format(task=tasks[j])), BUDGET)
                                  for j in range(n)])
    codes = ["" if cut else extract_cpp(text) for text, cut, _ in runs]
    res = await asyncio.gather(*[judge_code(label, c) for c in codes])
    blocks = [SUB_BLOCK.format(j=j + 1, task=tasks[j], result=fmt_result(r), cases=fmt_cases(r["cases"]),
                               code=codes[j][:CODE_CHARS]) if codes[j] else
              SUB_NONE.format(j=j + 1, task=tasks[j]) for j, r in enumerate(res)]
    s = [r["score"] for r in res]
    best = max(s)
    final_msgs = plan_msgs + [{"role": "assistant", "content": _visible(ptext, pcut)},
                              {"role": "user", "content": FEEDBACK.format(blocks="\n\n".join(blocks),
                                                                          best=100 * best)}]
    ftext, fcut, fseg = await _turn(lead, final_msgs, BUDGET)
    adopted = None if fcut else parse_adopt(ftext, n)
    if adopted is not None and codes[adopted - 1]:
        fres = res[adopted - 1]  # the same program: reuse its score (time-limited heuristics vary run to run)
    else:
        adopted = None
        fres = await judge_code(label, "" if fcut else extract_cpp(ftext))
    S = fres["score"]
    gen_sub = [_gen(x) for x in subs]
    plan_gen = sum(lead.segments[pseg].loss_mask) if pseg >= 0 else 0
    lead_gen = _gen(lead)
    rec = {"game": "team", "s": s, "S": S, "adopted": adopted}
    corr = [c for i in range(n) for j in range(i + 1, n)
            if (c := _pearson(res[i]["cases"], res[j]["cases"])) is not None]
    info = {"fo_sub_corr": sum(corr) / len(corr) if corr else float("nan"),"fo_S": S, "fo_best_sub": best, "fo_mean_sub": sum(s) / n, "fo_beat": float(S > best),
            "fo_gain": max(0.0, S - best), "fo_adopt": float(adopted is not None),
            "fo_tasks_ok": sum(t != NO_TASK for t in tasks) / n, "fo_plan_cut": float(pcut),
            "fo_final_cut": float(fcut), "fo_sub_cut": sum(c for _, c, _ in runs) / n,
            "fo_final_bad": _bad(fres), "fo_sub_bad": sum(_bad(r) for r in res) / n,
            "fo_tokens": lead_gen + sum(gen_sub), "fo_latency": lead_gen + max(gen_sub),
            "fo_plan_tokens": plan_gen,
            "fo_judge_infra_error": float(fres["infra_error"] or any(r["infra_error"] for r in res))}
    # turns in reward order: plan, final, sub_1..sub_n (session, segment index or -1)
    turns = [(lead, pseg), (lead, fseg)] + [(subs[j], runs[j][2]) for j in range(n)]
    trace = ([_tt("plan", ptext, pcut, None, tasks=tasks)]
             + [_tt(f"sub{j + 1}", runs[j][0], runs[j][1], res[j]) for j in range(n)]
             + [_tt("final", ftext, fcut, fres)])
    return turns, rec, info, trace


async def _seq(input, messages, label):
    sess = Session(input, "solo", THINK, max_len=MAX_LEN)
    chat, s, turns, texts, v_at = deepcopy(messages), [], [], [], []
    cut_n, bad, infra, best = 0, 0.0, False, 0.0
    for t in range(ROUNDS):
        text, cut, seg = await _turn(sess, chat, BUDGET)
        code = "" if cut else extract_cpp(text)
        res = await judge_code(label, code)
        s.append(res["score"])
        turns.append((sess, seg))
        texts.append(_tt(f"round{t + 1}", text, cut, res))
        cut_n, bad, infra = cut_n + cut, bad + _bad(res), infra or res["infra_error"]
        best = max(best, res["score"])
        v_at.append(best)
        if t + 1 < ROUNDS:
            shown = f"```cpp\n{code[:CODE_CHARS]}\n```" if code else "(no code: the answer was cut off or had no program)"
            chat = chat + [{"role": "assistant", "content": shown},
                           {"role": "user", "content": REVISE.format(result=fmt_result(res),
                                                                     cases=fmt_cases(res["cases"]), best=100 * best)}]
    gen = _gen(sess)
    rec = {"game": "seq", "s": s}
    info = {"fo_S": s[-1], "fo_first": s[0], "fo_cut": cut_n / ROUNDS, "fo_bad": bad / ROUNDS, "fo_tokens": gen,
            "fo_latency": gen, **{f"fo_V_at{t + 1}": v for t, v in enumerate(v_at)},
            "fo_judge_infra_error": float(infra)}
    return turns, rec, info, texts


async def _par(input, messages, label):
    sessions = [Session(input, f"try{t + 1}", THINK, max_len=MAX_LEN) for t in range(ROUNDS)]
    runs = await asyncio.gather(*[_turn(x, messages, BUDGET) for x in sessions])
    res = await asyncio.gather(*[judge_code(label, "" if cut else extract_cpp(text)) for text, cut, _ in runs])
    s = [r["score"] for r in res]
    gen = [_gen(x) for x in sessions]
    rec = {"game": "par", "s": s}
    info = {"fo_S": s[-1], "fo_mean": sum(s) / len(s), "fo_cut": sum(c for _, c, _ in runs) / ROUNDS,
            "fo_bad": sum(_bad(r) for r in res) / ROUNDS, "fo_tokens": sum(gen), "fo_latency": max(gen),
            **{f"fo_V_at{t + 1}": max(s[: t + 1]) for t in range(ROUNDS)},
            "fo_judge_infra_error": float(any(r["infra_error"] for r in res))}
    trace = [_tt(f"try{t + 1}", runs[t][0], runs[t][1], res[t]) for t in range(ROUNDS)]
    return [(sessions[t], runs[t][2]) for t in range(ROUNDS)], rec, info, trace


def n_samples(game: str) -> int:
    return 2 + SUBS if game == "team" else ROUNDS


def _role_group(game: str, role: int) -> int:
    if game == "team":
        return min(role, 2)  # plan 0, final 1, subagents 2
    return role if game == "seq" else 0


def _pack(input, turns, game: str, rec: dict, info: dict) -> GenerateFnOutput:
    """One training sample per agent turn in reward order, padded with masked stand-ins to n_samples(game)."""
    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    g = input.sample.group_index if input.sample.group_index is not None else input.sample.index
    r = rewards(game, REWARD, rec)
    real = [sg for sess, _ in turns for sg in sess.segments if any(sg.loss_mask)]
    if not real:  # nothing generated by anyone: retry the episode
        raise Aborted("empty episode")
    samples, at = [], {}
    for role, (sess, seg_i) in enumerate(turns):
        if seg_i < 0:
            continue  # this turn generated nothing
        seg = sess.segments[seg_i]
        if id(seg) not in at:  # a segment several turns generated into (string-extending chat) carries the last
            at[id(seg)] = len(samples)
            samples.append(seg)
        seg.reward, seg.rollout_id, seg.group_index = float(r[role]), rollout_id, 8 * g + _role_group(game, role)
        seg.metadata = {**(seg.metadata or {}), "fo_role": role, "fo": {**rec, "role": role}}
    while len(samples) < n_samples(game):
        p = _pad_sample(real[0])
        p.reward, p.rollout_id, p.group_index = 0.0, rollout_id, 8 * g + 7
        p.metadata = {k: v for k, v in p.metadata.items() if not k.startswith("fo")} | {"fo_pad": True}
        samples.append(p)
    samples[0].metadata = {**samples[0].metadata, **info}
    return GenerateFnOutput(samples=samples)


def post_process(args, samples):
    """--custom-reward-post-process-path: the arm's reward from each real sample's episode record (identity on live
    rollouts; relabels replayed ones)."""
    out = []
    for s in samples:
        fo = (s.metadata or {}).get("fo")
        out.append(relabel(fo) if fo is not None else float(s.get_reward_value(args)))
    return out, list(out)


def _means(samples) -> dict:
    acc: dict[str, list[float]] = {}
    for s in samples:
        for k, v in (s.metadata or {}).items():
            if k.startswith("fo_") and isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and not (isinstance(v, float) and math.isnan(v)) and k != "fo_role":
                acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    m = {f"rollout/{k}": v for k, v in _means(samples).items()}
    print(f"[fcs_oracle] rollout {rollout_id}: {json.dumps({k: round(v, 4) for k, v in m.items()})}", flush=True)
    if isinstance(rollout_extra_metrics, dict):
        rollout_extra_metrics.update(m)
    return False


def log_eval(rollout_id, args, data, extra_metrics) -> bool:
    for key, d in data.items():
        if d.get("samples"):
            m = {f"eval/{key}/{k}": v for k, v in _means(d["samples"]).items()}
            print(f"[fcs_oracle] eval {rollout_id} {key}: {json.dumps({k: round(v, 4) for k, v in m.items()})}",
                  flush=True)
            if isinstance(extra_metrics, dict):
                extra_metrics.update(m)
    return False
