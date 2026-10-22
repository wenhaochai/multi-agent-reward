"""Long-horizon agents on Frontier-CS (user 2026-10-21: "these agents never work long enough; they need to submit again
and again and talk all the time"; first step by popup: evaluate before any training, about 500k generated tokens per
agent). N mini-swe-agents (fcs_agent's bash-only scaffold) share one git repository and work in lockstep rounds; with
N = 1 the same code is the lone-agent control at the same per-agent budget, and MA_LG_TOKENS sets an equal-total one.

What changes against fcs_agent_team (one submission, no feedback, no messages):
  submit [file]   (the whole action) judges the file (default solution.cpp) at once. The agent sees its PROVISIONAL score:
                  the mean over the even-numbered test cases, with per-case scores and compiler errors, like a contest's
                  provisional board. Every submission and its provisional score goes to SCORES.md and to every agent.
                  An agent may submit as often as it likes.
  post <message>  (the whole action) sends a message to every other agent. New messages and new submissions of the
                  others are shown at the top of each agent's next observation, so the agents keep talking without
                  having to poll files; BOARD.md keeps the whole log.
  the end         an agent stops when it runs `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, uses up MA_LG_TOKENS
                  generated tokens, or makes three format errors in a row. When every agent has stopped, the submission
                  with the highest provisional score (the earliest on a tie) is the team's answer, and its FINAL score
                  (the mean over the odd-numbered cases, never shown to the agents) is the episode's score. A problem
                  with a single test case uses it for both (flagged "shared_case" in the trace).
  context         a long episode overflows the 65k-token context, so before each step the oldest steps of an agent's
                  chat are dropped (MA_LG_CTX_CHARS of history kept) and its instance message says how many; the agents
                  are told to keep what matters in files (notes, SCORES.md, BOARD.md).
  memory          in evaluation only the last turn's segment is kept (every turn re-renders the whole chat, so keeping
                  every segment would hold hundreds of full contexts per agent); generated tokens are counted apart.

Env: MA_LG_N (4), MA_LG_TOKENS (500000 per agent), MA_LG_CTX_CHARS (110000), MA_LG_MAX_ROUNDS (2000, a safety cap),
fcs_agent's MA_AG_TURN / MA_AG_TIMEOUT / MA_AG_THINK / MA_AG_MAX_LEN / MA_AG_TRACE_DIR.
--custom-generate-function-path miles_team.fcs_agent_long.generate   (evaluation; training is not supported yet)
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import random
import re
import shutil
import time
import uuid
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

from . import fcs_agent as A
from . import fcs_agent_team as T
from .fcs_rm import answer, judge_code
from .team_rollout import Aborted, Session

N = int(os.environ.get("MA_LG_N", "4"))
TOKENS = int(os.environ.get("MA_LG_TOKENS", "500000"))
CTX_CHARS = int(os.environ.get("MA_LG_CTX_CHARS", "110000"))
MAX_ROUNDS = int(os.environ.get("MA_LG_MAX_ROUNDS", "2000"))
MAX_FILE = 256 << 10          # bytes of a submitted file
MAX_POST = 2000               # characters of one message
SUBMIT_RE = re.compile(r"\s*submit(?:\s+(\S+))?\s*")
POST_RE = re.compile(r"\s*post\s+(.+?)\s*", re.S)

_LONE_ENV = ("You work in an empty directory with a Linux shell (g++, python3 and the usual command-line tools; no "
             "internet).")
_LONE_SUBMIT = ("6. Submit your solution and finish your work by issuing the following command: "
                "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat solution.cpp`.\n   The output of that command after "
                "its first line is your submission. Do not combine it with any other command. <important>After this "
                "command, you cannot continue working on this task.</important>")
_FLAT = A.INSTANCE.replace("\\\n", "")
assert _LONE_ENV in _FLAT and _LONE_SUBMIT in _FLAT, "fcs_agent.INSTANCE changed"

_TEAM_ENV = ("You are {name}, one of {n} agents working at the same time in this shared git repository, with a Linux "
             "shell (g++, python3, git and the usual command-line tools; no internet). The agents act in rounds: in "
             "each round every agent runs one command, in a random order. After every command the repository's "
             "source files are committed under the name of the agent that ran it, so you see the others' files and "
             "commits.")
_SOLO_ENV = ("You work alone in a git repository with a Linux shell (g++, python3, git and the usual command-line "
             "tools; no internet).")
_TOOLS = """

## Long work, submissions and messages

You have a budget of about {budget} generated tokens, and you should use it: keep improving your program for as long \
as you can make it better. Two commands are provided by the harness; each must be the whole action, alone:

- `submit` or `submit <file>`: judges the file (default solution.cpp) right away and shows its PROVISIONAL score \
(0-100): the mean over half of the hidden tests, with the score of each of those tests and any compiler errors. You \
may submit as often as you like.{team_submit}
- `post <message>`: {post_text}

When {when_end}, the submission with the highest provisional score is judged on the OTHER half of the hidden tests; \
that final score is the result. Older steps of this conversation are removed when it gets long, so keep what you need \
to remember in a notes file and re-read it; SCORES.md lists every submission and its provisional score."""
_STEP6 = ("6. Submit with `submit` whenever you have something better, read the provisional score, and keep "
          "improving.\n7. When you cannot improve any more, finish your work with `echo "
          "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. This ends only your own work and submits nothing by itself.")


def instance(problem: str, name: str) -> str:
    team = N > 1
    tools = _TOOLS.format(
        budget=f"{TOKENS:,}",
        team_submit=(" Every submission of every agent and its provisional score is shown to all agents." if team
                     else ""),
        post_text=("sends the message to every other agent; their messages appear at the top of your next "
                   "observations, and BOARD.md keeps them all. Use it to split the work, report what you found and "
                   "agree on the next step." if team else "there are no other agents, so this only writes BOARD.md."),
        when_end="every agent has finished" if team else "you finish")
    steps = f" You have at most {A.STEPS} steps." if A.STEPS > 0 else ""
    text = (_FLAT.replace(_LONE_ENV, _TEAM_ENV if team else _SOLO_ENV).replace(_LONE_SUBMIT, _STEP6))
    text = text.format(problem=problem, timeout=A.TIMEOUT, steps=steps, name=name, n=N)
    marker = "## Important Rules"
    assert marker in text
    return text.replace(marker, tools.strip() + "\n\n" + marker, 1)


def split_cases(n: int) -> tuple[list[int], list[int], bool]:
    """(provisional, final, shared): even-numbered cases are provisional, odd-numbered final; one case serves both."""
    if n <= 1:
        return [0], [0], True
    return list(range(0, n, 2)), list(range(1, n, 2)), False


def subset_mean(cases: list[float], idx: list[int]) -> float:
    vals = [cases[i] for i in idx if i < len(cases)]
    return sum(vals) / len(vals) if vals else 0.0


def trim_chat(chat: list, keep_chars: int, dropped: int) -> tuple[list, int]:
    """Drop the oldest (assistant, user) pairs after [system, instance] until the history fits keep_chars; the
    instance message then says how many steps were removed. Returns (chat, total dropped)."""
    head, hist = chat[:2], chat[2:]
    n = [len(m["content"]) + len(m.get("reasoning_content") or "") for m in hist]   # reasoning kept in history (Flint)
    size = sum(n)
    k = 0
    while size > keep_chars and len(hist) - 2 * k > 2:
        size -= n[2 * k] + n[2 * k + 1]
        k += 1
    if not k:
        return chat, dropped
    dropped += k
    base = head[1]["content"].split("\n\n[Note: ")[0]
    note = (f"\n\n[Note: your {dropped} earliest steps were removed from this conversation to fit the context. Your "
            "files, SCORES.md and BOARD.md still hold everything.]")
    return [head[0], {"role": "user", "content": base + note}] + hist[2 * k:], dropped


class _Agent(T._Agent):
    """fcs_agent_team's agent with a token budget of its own (segments are trimmed in evaluation) and a chat that is
    trimmed to the context."""

    def __init__(self, input, j: int, problem: str):
        self.name = f"agent{j + 1}"
        self.sess = Session(input, self.name, A.THINK, max_len=A.MAX_LEN)
        self.chat = [{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": instance(problem, self.name)}]
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns, self.segs = False, "", [], []
        self.generated, self.dropped, self.seen, self.last_seg = 0, 0, 0, None
        self.n_submit = self.n_post = 0

    def gen(self) -> int:
        return self.generated

    async def think(self):
        """One model turn, as fcs_agent_team's, with this module's token budget and context trimming."""
        if self.generated >= TOKENS:
            self.done, self.exit = True, "TokensExceeded"
            return None
        self.chat, self.dropped = trim_chat(self.chat, CTX_CHARS, self.dropped)
        self.steps += 1
        self.sess.last_messages = None
        n_seg = len(self.sess.segments)
        text, cut = await self.sess.turn(self.chat, min(A.TURN, TOKENS - self.generated))
        for tok in A._SPECIAL:
            text = text.replace(tok, "")
        if len(self.sess.segments) > n_seg:
            seg = self.sess.segments[-1]
            self.generated += sum(seg.loss_mask)
            self.last_seg = seg
            if self.sess.input.evaluation:   # keep one segment: the episode returns only the last one
                self.sess.segments = [seg]
        if not text and cut:
            self.done, self.exit = True, "ContextExceeded"
            self.turns.append({"step": self.steps, "text": "", "cut": True})
            return None
        vis = "" if cut else answer(text, A.THINK).strip()
        actions = [] if cut else [a.strip() for a in A.ACTION_RE.findall(vis)]
        rec = {"step": self.steps, "text": text, "cut": bool(cut)}
        if len(actions) != 1:
            self.fmt_run, self.fmt_total, self.cut_n = self.fmt_run + 1, self.fmt_total + 1, self.cut_n + bool(cut)
            msg = A.LENGTH_ERROR if cut else A.FORMAT_ERROR.format(
                error=f"Expected exactly one action, found {len(actions)}.", n=len(actions))
            self.turns.append({**rec, "format_error": True})
            self.chat = self.chat + [{"role": "assistant", "content": vis or "(cut off)"}, {"role": "user", "content": msg}]
            if self.fmt_run >= A.MAX_FORMAT_ERRORS:
                self.done, self.exit = True, "RepeatedFormatError"
            return None
        self.fmt_run = 0
        self._pending = (vis, rec)
        return actions[0]

    def observe(self, cmd: str, out: dict, rnd: int, order: int) -> None:
        T._Agent.observe(self, cmd, out, rnd, order)
        if self.exit == "Submitted":   # the finish command: the agent's own work ends; it submitted nothing by it
            self.exit = "Finished"


async def episode(input: GenerateFnInput, messages: list, label: str):
    from miles_team.fcs_judge import load_problem
    from pathlib import Path
    problem = A.problem_text(messages)
    agents = [_Agent(input, j, problem) for j in range(N)]
    n_cases = len(load_problem(Path(label))["cases"])
    prov_idx, final_idx, shared = split_cases(n_cases)
    work = os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent_long", uuid.uuid4().hex)
    os.makedirs(work)
    loop = asyncio.get_running_loop()
    rng = random.Random(f"{label}:{input.sample.index}:{getattr(input.sample, 'group_index', 0)}")
    rounds, run_s, subs, board, events = 0, 0.0, [], [], []

    def write_logs():
        with open(os.path.join(work, "SCORES.md"), "w") as f:
            f.write("# Submissions (provisional score = mean over half of the hidden tests)\n\n")
            f.write("| # | round | agent | file | status | provisional |\n|---|---|---|---|---|---|\n")
            for i, s in enumerate(subs, 1):
                f.write(f"| {i} | {s['round']} | {s['agent']} | {s['file']} | {s['status']} | {100 * s['prov']:.2f} |\n")
        with open(os.path.join(work, "BOARD.md"), "w") as f:
            f.write("# Messages\n\n" + "".join(f"- [round {r}] {a}: {m}\n" for r, a, m in board))

    async def do_submit(a, path: str) -> dict:
        p = os.path.realpath(os.path.join(work, path))
        if not p.startswith(os.path.realpath(work) + os.sep) or not os.path.isfile(p):
            return {"output": f"submit: no file {path} in the repository\n", "returncode": 1, "exception_info": ""}
        with open(p, "rb") as f:
            code = f.read(MAX_FILE + 1)
        if len(code) > MAX_FILE:
            return {"output": f"submit: {path} is larger than {MAX_FILE} bytes\n", "returncode": 1, "exception_info": ""}
        code = code.decode("utf-8", errors="replace")
        res = await judge_code(label, code)
        cases = res["cases"]
        prov, final = subset_mean(cases, prov_idx), subset_mean(cases, final_idx)
        rev = (await loop.run_in_executor(A._POOL, lambda: T._git(work, "rev-parse", "HEAD").stdout.strip()))
        s = {"agent": a.name, "round": rounds, "step": a.steps, "file": path, "status": res["status"], "prov": prov,
             "final": final, "infra": bool(res["infra_error"]), "rev": rev,
             "sha": hashlib.sha1(code.encode()).hexdigest()[:12], "code": code}
        subs.append(s)
        a.n_submit += 1
        events.append((rounds, a.name, f"submitted {path}: {res['status']}, provisional {100 * prov:.2f}"))
        best = max(subs, key=lambda x: (x["prov"], -subs.index(x)))
        per = ", ".join(f"{100 * cases[i]:.0f}" for i in prov_idx if i < len(cases))
        msg = (f"Submission #{len(subs)} by {a.name}: {res['status']}. Provisional score {100 * prov:.2f} on "
               f"{len(prov_idx)} tests (per test: {per}). Best provisional so far: {100 * best['prov']:.2f} "
               f"(#{subs.index(best) + 1}, {best['agent']}).\n")
        if res["status"] == "compile error" and res.get("msg"):
            msg += "Compiler output:\n" + res["msg"][:3000] + "\n"
        return {"output": msg, "returncode": 0, "exception_info": ""}

    def do_post(a, text: str) -> dict:
        text = text.strip().strip("'\"")[:MAX_POST]
        board.append((rounds, a.name, text))
        a.n_post += 1
        if N > 1:
            events.append((rounds, a.name, f"posted: {text}"))
        return {"output": "Message posted to the team.\n" if N > 1 else "Written to BOARD.md.\n", "returncode": 0,
                "exception_info": ""}

    def updates_for(a) -> str:
        new = [e for e in events[a.seen:] if e[1] != a.name]
        a.seen = len(events)
        if not new:
            return ""
        return "<team_updates>\n" + "".join(f"[round {r}] {n} {t}\n" for r, n, t in new) + "</team_updates>\n"

    try:
        await loop.run_in_executor(A._POOL, T.repo_init, work)
        await loop.run_in_executor(A._POOL, write_logs)
        while any(not a.done for a in agents) and rounds < MAX_ROUNDS:
            rounds += 1
            active = [a for a in agents if not a.done]
            cmds = await asyncio.gather(*[a.think() for a in active])
            todo = [(a, c) for a, c in zip(active, cmds) if c is not None]
            rng.shuffle(todo)
            for k, (a, cmd) in enumerate(todo):
                m_sub, m_post = SUBMIT_RE.fullmatch(cmd), POST_RE.fullmatch(cmd)
                if m_sub:
                    out, secs = await do_submit(a, m_sub.group(1) or "solution.cpp"), 0.0
                elif m_post:
                    out, secs = do_post(a, m_post.group(1)), 0.0
                else:
                    out = await loop.run_in_executor(A._POOL, A.run_command, cmd, work)
                    secs = out["seconds"]
                run_s += secs
                await loop.run_in_executor(A._POOL, write_logs)
                await loop.run_in_executor(A._POOL, T.commit_as, work, a.name, a.steps)
                upd = updates_for(a)
                if upd:
                    out = {**out, "output": upd + out["output"]}
                a.observe(cmd, out, rounds, k)
        best = max(subs, key=lambda x: (x["prov"], -subs.index(x))) if subs else None
        shares = (await loop.run_in_executor(A._POOL, T.blame_shares, work, best["file"], best["rev"])) if best else {}
        log = (await loop.run_in_executor(A._POOL, lambda: T._git(work, "log", "--format=%an %s").stdout))[-20000:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    S = best["final"] if best else 0.0
    oracle = max((s["final"] for s in subs), default=0.0)
    first = subs[0]["round"] if subs else -1
    info = {"fo_S": S, "fo_R": S, "fo_lg_n": N, "fo_lg_rounds": rounds, "fo_lg_submissions": len(subs),
            "fo_lg_best_prov": best["prov"] if best else 0.0, "fo_lg_oracle_final": oracle,
            "fo_lg_selection_loss": oracle - S, "fo_lg_first_submit_round": first,
            "fo_lg_posts": len(board), "fo_lg_tokens": sum(a.generated for a in agents) / N,
            "fo_tokens": sum(a.generated for a in agents), "fo_latency": max(a.generated for a in agents),
            "fo_ag_steps": sum(a.steps for a in agents) / N, "fo_ag_format_errors": sum(a.fmt_total for a in agents) / N,
            "fo_ag_run_min": run_s / 60, "fo_ag_compile_fail": float(bool(best) and best["status"] == "compile error"),
            "fo_ag_no_code": float(best is None), "fo_ag_top_author": max(shares.values()) if shares else 0.0,
            **{f"fo_ag_exit_{e}": sum(a.exit == e for a in agents) / N for e in
               ("Finished", "TokensExceeded", "ContextExceeded", "RepeatedFormatError")},
            "fo_judge_infra_error": float(any(s["infra"] for s in subs))}
    A._trace({"time": round(time.time(), 1), "eval": input.evaluation, "game": "agent_long", "label": label, "S": S,
              "n": N, "budget": TOKENS, "rounds": rounds, "shared_case": shared, "n_cases": n_cases,
              "submissions": [{k: v for k, v in s.items() if k != "code"} for s in subs],
              "best": {k: v for k, v in best.items()} if best else None, "board": board, "blame": shares,
              "git_log": log,
              "agents": [{"name": a.name, "exit": a.exit, "steps": a.steps, "generated": a.generated,
                          "dropped": a.dropped, "submits": a.n_submit, "posts": a.n_post,
                          "turns": [{**t, "text": t.get("text", "")[-4000:]} for t in a.turns]} for a in agents]})
    return agents, S, info


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Evaluation only for now (user 2026-10-21: evaluate before training). An abort returns an ABORTED sample."""
    try:
        assert input.evaluation, "fcs_agent_long supports evaluation only"
        meta = input.sample.metadata or {}
        assert meta.get("messages"), "fcs_agent_long needs the raw chat in sample.metadata['messages']"
        agents, S, info = await episode(input, meta["messages"], input.sample.label)
        segs = [a.last_seg for a in agents if a.last_seg is not None]
        if not segs:
            raise Aborted("empty episode")
        out = segs[-1]
        out.reward = float(S)
        out.metadata = {**(out.metadata or {}), **info}
        return GenerateFnOutput(samples=out)
    except Aborted:
        s = deepcopy(input.sample)
        s.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=s)


log_rollout = A.log_rollout
log_eval = A.log_eval
