"""Frontier-CS games with an oracle (labs-molt-docs docs/fcs_multiagent_reward.md, user decisions 2026-10-03/04):
every submitted program is judged and its total score (0-100), per-test-case scores and compiler errors come back.
The judged program of a reply is its LAST ```cpp block; every prompt says so.
--custom-generate-function-path miles_team.fcs_oracle.generate

Games (MA_FO_GAME for training; an eval set picks its game with metadata.fo_game):
  team  1 lead + 4 subagents, after DeepSeek-V4.1-Flash's Agent Team mode (arXiv 2609.19969 sec. 5.3.5: the lead
        delegates a task to each teammate, teammates start fresh, the lead reviews their work and produces the final
        answer). Turn 1: the lead writes one task per subagent (<task j>...</task>). Each subagent (a fresh session:
        the problem plus its task) is a helper, not a forced submitter: it may test up to MA_FO_SUB_TESTS programs
        against the judge (a reply with a cpp block and no <report>), seeing each result, and then reports to the
        lead (a <report>...</report> reply, never judged; a reply without a cpp block is its report too). Turn 2: the
        lead reads every subagent's task, test scores, best-scoring tested program with its result, and report, and
        submits the final program (or <adopt j/> for subagent j's best tested program). Code blocks are parsed line
        by line (fcs_rm.code_blocks); a subagent test needs a cpp-tagged block. The team's outcome is the lead's
        final submission only: V = S.
  seq   one agent, MA_FO_ROUNDS rounds in one chat, each round seeing its earlier programs and their results; it is
        told the round number and that only the last round counts; the outcome is the last round's submission.
  par   MA_FO_ROUNDS independent solo attempts; the outcome is the best of them (an oracle-selected reference).

Rewards (MA_FO_REWARD; s = a judged score in [0, 1]):
  team  shared  every turn gets S (plan, final, every subagent turn)
        bonus   lead S; subagent j: S + MA_FO_BONUS x (its best test score; 0 if it tested none)
        diff    lead S; subagent j: S - S_-j, where S_-j is the score of a counterfactual final the lead writes from
                the same context with subagent j's block replaced by "not available" and its programs not adoptable
                (MA_FO_SUBS extra lead turns,
                generated and judged but never trained on; also computed when MA_FO_CF=1, e.g. by the shared-value
                producer, so a diff run can relabel replayed rollouts)
  seq   shared  every round gets s_last;  indiv  s_t;  diff  max(0, s_t - max_{i<t} s_i)
  par   shared  max_t s_t;  indiv  s_t (plain single-agent RL);  diff  V - max_{i!=t} s_i

Training returns one sample per agent turn (a turn that string-extends its session's previous turn shares that
segment). With miles --variable-rollout-samples the trainers take any count; without it the episode is padded with
masked 2-token stand-ins to a fixed count (team 2 + SUBS x (SUB_TESTS + 1), seq/par MA_FO_ROUNDS). Critic groups: 8 g + role (team: plan 0, final 1, every subagent turn 2; seq: round t; par: 0;
padding 7). A training episode in which a judge call failed for infrastructure reasons is aborted (miles resubmits
the group). Every real sample carries fo = the episode record (game, scores, role) so post_process recomputes the
arm's reward (identity live; relabels rollouts replayed by miles --replay-rollout-data). Eval returns the sample of
the episode's outcome turn with reward V. Latency = generated tokens on the critical path (team: plan + the longest
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

from .fcs_rm import code_blocks, extract_cpp, judge_code, strip_think
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
SUB_TESTS = int(os.environ.get("MA_FO_SUB_TESTS", "3"))
REPORT_BUDGET = int(os.environ.get("MA_FO_REPORT_BUDGET", "16384"))
REPORT_CHARS = int(os.environ.get("MA_FO_REPORT_CHARS", "6000"))
BONUS = float(os.environ.get("MA_FO_BONUS", "0.5"))
CF = os.environ.get("MA_FO_CF", "0") == "1"
TRACE_DIR = os.environ.get("MA_FO_TRACE_DIR", "")
TRACE_EVERY = int(os.environ.get("MA_FO_TRACE_EVERY", "32"))
GAMES = ("team", "seq", "par")
ARMS = {"team": ("shared", "bonus", "diff"), "seq": ("shared", "indiv", "diff"),
        "par": ("shared", "indiv", "diff")}
assert GAME in GAMES and REWARD in ARMS[GAME], (GAME, REWARD)

# the problem prompt's output rule (data/*.jsonl), dropped for the lead's plan and the subagents, who need not answer
# with code only
OUTPUT_RULE = ("Output ONLY the C++ code wrapped in ```cpp and ```. No explanation. If you write more than one ```cpp "
               "block, only the last one is submitted.")
OLD_OUTPUT_RULE = "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation."
PLAN = (
    "\n\nYou lead a team of {n} subagents. Each subagent gets this problem plus one task you write. A subagent may "
    "test up to {k} C++ programs against the judge, seeing each program's total score (0-100), per-test-case scores "
    "and compiler errors, and then reports back to you: ideas, analysis, test cases or code. Then you write the "
    "team's final solution; only your final submission counts. Write the {n} tasks now: what each subagent should "
    "explore, try or check. Output exactly {n} blocks, <task 1>...</task> through <task {n}>...</task>. Do not write "
    "the solution yet."
)
SUB = (
    "\n\nYou are a subagent. Your team lead gave you this task:\n{task}\n\nYou do not have to write a full "
    "solution: your report to the lead can hold ideas, analysis, test cases, partial code or a complete program, "
    "whatever helps the lead most. To test a C++ program, reply with it in a ```cpp block (and no <report>): the "
    "last ```cpp block of such a reply is judged, and you will see its total score (0-100), per-test-case scores and "
    "compiler errors. You can test up to {k} programs. When you are done, reply with your report for the lead inside "
    "<report>...</report>; nothing in a report reply is judged, and a reply without a ```cpp block also ends your "
    "work and goes to the lead as your report."
)
SUB_FEEDBACK = (
    "Your program was judged: {result}\nPer-test-case scores (fractions of full marks): {cases}\nYou can test {left} "
    "more program(s). Reply with another ```cpp block to test it, or finish with <report>...</report>."
)
SUB_LAST = (
    "Your program was judged: {result}\nPer-test-case scores (fractions of full marks): {cases}\nYou have no tests "
    "left. Write your report for the lead inside <report>...</report>."
)
NO_TASK = "(no task was given: help with the problem your own way)"
FEEDBACK = (
    "Your subagents reported back. For each: its task, the scores of the programs it tested, its best-scoring "
    "tested program with its result (total score 0-100; per-test-case scores are fractions of full marks), and its "
    "report.\n\n{blocks}\n\nWrite the team's final solution; only your final submission counts. Output the final "
    "C++ code in a ```cpp block (if you write more than one, only the last one is submitted), or output only "
    "<adopt j/> to submit subagent j's best-scoring tested program, unchanged."
)
NOT_AVAILABLE = "### Subagent {j}\n(this subagent's work is not available)"
SEQ_INTRO = ("\n\nYou have {r} rounds. After each round your program is judged and you see its total score (0-100), "
             "per-test-case scores and compiler errors. Only your round-{r} submission counts.")
REVISE = (
    "Your submission was judged: {result}\nPer-test-case scores (fractions of full marks): {cases}\nYour best score "
    "so far is {best:.2f}/100. This is round {t} of {r}; only your round-{r} submission counts. Write your solution "
    "for this round. Output ONLY the C++ code wrapped in ```cpp and ```; only the last ```cpp block is submitted."
)
_SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")  # Session decodes with special tokens kept
_TASK_RE = re.compile(r"<task\s*(\d+)\s*>(.*?)</task\s*\d*\s*>", re.IGNORECASE | re.DOTALL)
_ADOPT_RE = re.compile(r"<adopt\s*(?:j\s*=\s*)?\"?(\d+)\"?\s*/?>", re.IGNORECASE)
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.IGNORECASE | re.DOTALL)
_REPORT_OPEN_RE = re.compile(r"<report>(.*)$", re.IGNORECASE | re.DOTALL)
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
    """Rewards by reward index. team: [plan, final, sub_1..sub_n] (every turn of subagent j gets sub_j); seq/par: one
    per round."""
    if game == "team":
        S, top = rec["S"], [max(t) if t else 0.0 for t in rec["tests"]]
        if arm == "shared":
            return [S, S] + [S] * len(top)
        if arm == "bonus":
            return [S, S] + [S + BONUS * x for x in top]
        if arm == "diff":
            assert rec.get("S_minus") is not None, "diff needs the counterfactual finals (MA_FO_CF=1 when generating)"
            return [S, S] + [S - m for m in rec["S_minus"]]
    else:
        s = rec["s"]
        V = s[-1] if game == "seq" else max(s)
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
    V = rec["S"] if game == "team" else (rec["s"][-1] if game == "seq" else max(rec["s"]))
    info = {**info, "fo_V": V, "fo_game_" + game: 1.0}
    _write_trace({"time": round(time.time(), 1), "eval": input.evaluation, "game": game, "arm": REWARD,
                  "label": label, **rec, "V": V, "turns": texts})
    if input.evaluation:  # the episode's last submission (team: the lead's final), scored V
        sess, seg_i = turns[1][:2] if game == "team" else turns[-1][:2]
        if seg_i >= 0:
            out = sess.segments[seg_i]
        else:
            out = deepcopy(input.sample)
            out.status = Sample.Status.TRUNCATED
        out.reward = float(V)
        out.metadata = {**(out.metadata or {}), **info}
        return GenerateFnOutput(samples=out)
    return _pack(input, turns, game, rec, info)


def last_block(text: str) -> str:
    """The last cpp-tagged block of a reply's visible part, or "" (a reply without one submits nothing; an untagged
    block, e.g. a test case, is not a program)."""
    m = code_blocks(strip_think(text) or "", untagged=False)
    return m[-1] if m else ""


def _clip(text: str, n: int, what: str) -> str:
    """At most n characters, marked when cut, with an open code fence closed so it cannot swallow what follows."""
    if len(text) <= n:
        return text
    t = text[:n] + f"\n... ({what} truncated)"
    return t + "\n```" if t.count("```") % 2 else t


async def _gather(*aws):
    """asyncio.gather that cancels the siblings when one fails (an Aborted episode stops its other agents)."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise


def _check_infra(input, res: dict) -> dict:
    if res["infra_error"] and not input.evaluation:  # its 0 is not the policy's: stop the episode now
        raise Aborted("judge infrastructure error")
    return res


def _prompt_len(sess: Session, seg_i: int) -> int:
    if seg_i < 0:
        return 0
    sg = sess.segments[seg_i]
    return len(sg.tokens) - sg.response_length


def _without_output_rule(messages: list) -> list:
    out = deepcopy(messages)
    c = out[-1]["content"]
    out[-1] = {**out[-1], "content": c.replace(OUTPUT_RULE, "").replace(OLD_OUTPUT_RULE, "")}
    return out


async def _subagent(input, j: int, messages: list, task: str, label: str) -> dict:
    """One subagent: up to SUB_TESTS judged programs, each result shown back, then its report for the lead. A reply
    with <report> ends the work and is never judged; a reply without a cpp block is the report too."""
    sess = Session(input, f"sub{j + 1}", THINK, max_len=MAX_LEN)
    chat = _with_suffix(_without_output_rule(messages), SUB.format(task=task, k=SUB_TESTS))
    tests, segs, trace, report, cut_any, clamped = [], [], [], "", False, False
    while True:
        last_turn = len(tests) >= SUB_TESTS  # tests used up: this turn only reports
        budget = REPORT_BUDGET if last_turn else BUDGET
        text, cut, seg = await _turn(sess, chat, budget)
        clamped = clamped or (cut and seg < 0) or _prompt_len(sess, seg) + budget > MAX_LEN
        if seg >= 0 and seg not in segs:
            segs.append(seg)
        vis = "" if cut else (strip_think(text) or "").strip()
        code = "" if (cut or last_turn) else last_block(text)
        # a report: a closed <report>...</report>, or an open <report> in a reply with no cpp block (a mention of the
        # tag in a test reply does not stop its program from being judged)
        m = None if cut else (_REPORT_RE.search(vis) or (None if code else _REPORT_OPEN_RE.search(vis)))
        code = "" if m else code
        res = _check_infra(input, await judge_code(label, code)) if code else None
        if code:
            tests.append((code, res))
        trace.append(_tt(f"sub{j + 1}", text, cut, res, test=len(tests) if code else 0))
        if cut:
            cut_any, report = True, "(cut off: the subagent's reply hit the token limit)"
            break
        if m or not code or last_turn:
            report = (m.group(1) if m else vis).strip() or "(empty report)"
            break
        left = SUB_TESTS - len(tests)
        chat = chat + [{"role": "assistant", "content": vis},
                       {"role": "user", "content": (SUB_FEEDBACK if left else SUB_LAST).format(
                           result=fmt_result(res), cases=fmt_cases(res["cases"]), left=left)}]
    best = max(range(len(tests)), key=lambda i: tests[i][1]["score"]) if tests else None  # first of the ties
    return {"sess": sess, "segs": segs, "tests": tests, "best": best, "report": _clip(report, REPORT_CHARS, "report"),
            "cut": cut_any, "clamped": clamped, "trace": trace}


def _sub_block(j: int, task: str, r: dict) -> str:
    out = f"### Subagent {j + 1}\nTask: {task}\n"
    if r["tests"]:
        code, res = r["tests"][r["best"]]
        scores = ", ".join(f"{100 * x['score']:.2f}" for _, x in r["tests"])
        out += (f"Programs tested: {len(r['tests'])} (scores in order: {scores})\n"
                f"Best tested program (test {r['best'] + 1}): {fmt_result(res)}\n"
                f"Per-test-case scores: {fmt_cases(res['cases'])}\n```cpp\n{_clip(code, CODE_CHARS, 'code')}\n```\n")
    else:
        out += "Programs tested: 0\n"
    return out + "Report:\n" + r["report"]


async def _final(input, name, msgs: list, subr: list[dict], label: str):
    """A lead final turn and its judged result: the last cpp block, else <adopt j/> of subagent j's best tested
    program (its judged result reused), else the fence-less fallback of a solo answer. An adopt tag that names no
    tested program submits nothing."""
    sess = Session(input, name, THINK, max_len=MAX_LEN) if isinstance(name, str) else name
    text, cut, seg = await _turn(sess, msgs, BUDGET)
    code = "" if cut else last_block(text)
    tag = None if cut else _ADOPT_RE.search(strip_think(text) or "")
    adopted = parse_adopt(text, len(subr)) if (tag and not code) else None
    if adopted is not None and subr[adopted - 1]["tests"]:
        r = subr[adopted - 1]
        return sess, text, cut, seg, adopted, r["tests"][r["best"]][1]
    if not code and not cut and not tag:
        code = extract_cpp(text)  # no fenced cpp block and no adopt tag: the fallback, as for a solo answer
    return sess, text, cut, seg, None, _check_infra(input, await judge_code(label, code))


async def _team(input, messages, label):
    n = SUBS
    lead = Session(input, "lead", THINK, max_len=MAX_LEN)
    plan_msgs = _with_suffix(_without_output_rule(messages), PLAN.format(n=n, k=SUB_TESTS))
    ptext, pcut, pseg = await _turn(lead, plan_msgs, PLAN_BUDGET)
    tasks = parse_tasks("" if pcut else ptext, n)
    subr = await _gather(*[_subagent(input, j, messages, tasks[j], label) for j in range(n)])
    blocks = [_sub_block(j, tasks[j], r) for j, r in enumerate(subr)]
    head = plan_msgs + [{"role": "assistant", "content": _visible(ptext, pcut)}]
    final_msgs = head + [{"role": "user", "content": FEEDBACK.format(blocks="\n\n".join(blocks))}]
    _, ftext, fcut, fseg, adopted, fres = await _final(input, lead, final_msgs, subr, label)
    S = fres["score"]
    S_minus, cf_gen, cf_trace = None, 0, []
    if (REWARD == "diff" or CF) and not input.evaluation:
        # subagent j's work is gone from the counterfactual: its block says so and its programs cannot be adopted
        cfs = await _gather(*[_final(
            input, f"cf{j + 1}",
            head + [{"role": "user", "content": FEEDBACK.format(blocks="\n\n".join(
                NOT_AVAILABLE.format(j=j + 1) if i == j else b for i, b in enumerate(blocks)))}],
            [{**r, "tests": []} if i == j else r for i, r in enumerate(subr)], label) for j in range(n)])
        S_minus = [c[5]["score"] for c in cfs]
        cf_gen = sum(_gen(c[0]) for c in cfs)
        cf_trace = [_tt(f"cf{j + 1}", c[1], c[2], c[5]) for j, c in enumerate(cfs)]
    tests = [[x["score"] for _, x in r["tests"]] for r in subr]
    best = max((x for t in tests for x in t), default=0.0)
    gen_sub = [_gen(r["sess"]) for r in subr]
    plan_gen = sum(lead.segments[pseg].loss_mask) if pseg >= 0 else 0
    lead_gen = _gen(lead)
    final_prompt = _prompt_len(lead, fseg)
    rec = {"game": "team", "S": S, "tests": tests, "adopted": adopted, "S_minus": S_minus}
    info = {"fo_S": S, "fo_best_test": best, "fo_beat": float(S > best), "fo_adopt": float(adopted is not None),
            "fo_sub_tests": sum(map(len, tests)) / n, "fo_sub_notest": sum(not t for t in tests) / n,
            "fo_tasks_ok": sum(t != NO_TASK for t in tasks) / n, "fo_plan_cut": float(pcut),
            "fo_final_cut": float(fcut), "fo_sub_cut": sum(r["cut"] for r in subr) / n,
            "fo_final_bad": _bad(fres), "fo_report_chars": sum(len(r["report"]) for r in subr) / n,
            "fo_final_prompt_tokens": final_prompt,
            "fo_final_clamped": float((fcut and fseg < 0) or final_prompt + BUDGET > MAX_LEN),
            "fo_adopt_invalid": float(bool(not fcut and _ADOPT_RE.search(strip_think(ftext) or "")) and adopted is None
                                      and not last_block(ftext)),
            "fo_sub_clamped": sum(r["clamped"] for r in subr) / n,
            "fo_tokens": lead_gen + sum(gen_sub), "fo_latency": lead_gen + max(gen_sub), "fo_plan_tokens": plan_gen,
            "fo_cf_tokens": cf_gen,
            "fo_judge_infra_error": float(fres["infra_error"] or any(x["infra_error"] for r in subr for _, x in r["tests"]))}
    if S_minus is not None:
        info["fo_sub_marginal"] = sum(S - m for m in S_minus) / n
    # turns with their reward index: plan 0, final 1, every turn of subagent j 2 + j
    turns = [(lead, pseg, 0), (lead, fseg, 1)] + [(r["sess"], sg, 2 + j) for j, r in enumerate(subr) for sg in r["segs"]]
    trace = ([_tt("plan", ptext, pcut, None, tasks=tasks)] + [t for r in subr for t in r["trace"]]
             + [_tt("final", ftext, fcut, fres, adopted=adopted)] + cf_trace)
    return turns, rec, info, trace


async def _seq(input, messages, label):
    sess = Session(input, "solo", THINK, max_len=MAX_LEN)
    chat, s, turns, texts, v_at = _with_suffix(messages, SEQ_INTRO.format(r=ROUNDS)), [], [], [], []
    cut_n, bad, infra, best = 0, 0.0, False, 0.0
    for t in range(ROUNDS):
        text, cut, seg = await _turn(sess, chat, BUDGET)
        code = "" if cut else extract_cpp(text)
        res = _check_infra(input, await judge_code(label, code))
        s.append(res["score"])
        turns.append((sess, seg))
        texts.append(_tt(f"round{t + 1}", text, cut, res))
        cut_n, bad, infra = cut_n + cut, bad + _bad(res), infra or res["infra_error"]
        best = max(best, res["score"])
        v_at.append(best)
        if t + 1 < ROUNDS:
            shown = (f"```cpp\n{_clip(code, CODE_CHARS, 'code')}\n```" if code
                     else "(no code: the answer was cut off or had no program)")
            chat = chat + [{"role": "assistant", "content": shown},
                           {"role": "user", "content": REVISE.format(result=fmt_result(res), cases=fmt_cases(res["cases"]),
                                                                     best=100 * best, t=t + 2, r=ROUNDS)}]
    gen = _gen(sess)
    rec = {"game": "seq", "s": s}
    info = {"fo_S": s[-1], "fo_first": s[0], "fo_cut": cut_n / ROUNDS, "fo_bad": bad / ROUNDS, "fo_tokens": gen,
            "fo_latency": gen, **{f"fo_best_at{t + 1}": v for t, v in enumerate(v_at)},
            "fo_judge_infra_error": float(infra)}
    return turns, rec, info, texts


async def _par(input, messages, label):
    sessions = [Session(input, f"try{t + 1}", THINK, max_len=MAX_LEN) for t in range(ROUNDS)]
    runs = await _gather(*[_turn(x, messages, BUDGET) for x in sessions])
    res = await _gather(*[judge_code(label, "" if cut else extract_cpp(text)) for text, cut, _ in runs])
    s = [r["score"] for r in res]
    gen = [_gen(x) for x in sessions]
    rec = {"game": "par", "s": s}
    info = {"fo_S": s[-1], "fo_mean": sum(s) / len(s), "fo_cut": sum(c for _, c, _ in runs) / ROUNDS,
            "fo_bad": sum(_bad(r) for r in res) / ROUNDS, "fo_tokens": sum(gen), "fo_latency": max(gen),
            **{f"fo_best_at{t + 1}": max(s[: t + 1]) for t in range(ROUNDS)},
            "fo_judge_infra_error": float(any(r["infra_error"] for r in res))}
    trace = [_tt(f"try{t + 1}", runs[t][0], runs[t][1], res[t]) for t in range(ROUNDS)]
    return [(sessions[t], runs[t][2]) for t in range(ROUNDS)], rec, info, trace


def n_samples(game: str) -> int:
    return 2 + SUBS * (SUB_TESTS + 1) if game == "team" else ROUNDS


def _role_group(game: str, role: int) -> int:
    if game == "team":
        return min(role, 2)  # plan 0, final 1, subagents 2
    return role if game == "seq" else 0


def _pack(input, turns, game: str, rec: dict, info: dict) -> GenerateFnOutput:
    """One training sample per agent turn in reward order, padded with masked stand-ins to n_samples(game)."""
    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    g = input.sample.group_index if input.sample.group_index is not None else input.sample.index
    r = rewards(game, REWARD, rec)
    real = [sg for item in turns for sg in item[0].segments if any(sg.loss_mask)]
    if not real:  # nothing generated by anyone: retry the episode
        raise Aborted("empty episode")
    samples, at = [], {}
    for k, item in enumerate(turns):
        sess, seg_i, role = item if len(item) == 3 else (*item, k)
        if seg_i < 0:
            continue  # this turn generated nothing
        seg = sess.segments[seg_i]
        if id(seg) not in at:  # a segment several turns generated into (string-extending chat) carries the last
            at[id(seg)] = len(samples)
            samples.append(seg)
        seg.reward, seg.rollout_id, seg.group_index = float(r[role]), rollout_id, 8 * g + _role_group(game, role)
        seg.metadata = {**(seg.metadata or {}), "fo_role": role, "fo": {**rec, "role": role}}
    assert len(samples) <= n_samples(game), (len(samples), n_samples(game))
    # with miles --variable-rollout-samples the trainers split whatever a rollout returns into fixed steps: no pads
    target = len(samples) if getattr(input.state.args, "variable_rollout_samples", None) else n_samples(game)
    while len(samples) < target:
        p = _pad_sample(real[0])
        p.tokens = [p.tokens[0], p.tokens[-1]]  # a 2-token stand-in: one prompt token and one masked response token
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
