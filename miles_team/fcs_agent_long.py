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
  finish          `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (mini-swe-agent's submit, which Flint uses to submit: smoke
                  w69 had it say "Let me submit" and run it, 0 submits in 4 episodes) submits solution.cpp as `submit`
                  does and the agent goes on; when solution.cpp has not changed since its last submission it ends the
                  agent's work.
  the end         an agent stops when it runs the finish command on an unchanged solution.cpp, uses up MA_LG_TOKENS
                  generated tokens, or makes three format errors in a row (MA_LG_END=budget: not on an unchanged
                  finish command, unless it is the third in a row). When every agent has stopped, the submission
                  with the highest provisional score (the earliest on a tie) is the team's answer, and its FINAL score
                  (the mean over the odd-numbered cases, never shown to the agents) is the episode's score. A problem
                  with a single test case uses it for both (flagged "shared_case" in the trace, fo_lg_shared_case in
                  info). An unsubmitted solution.cpp left by an agent that did not finish is judged apart: fo_lg_S_auto
                  is the score had it been submitted (S itself counts only submissions).
  context         a long episode overflows the 65k-token context, so before each step the oldest steps of an agent's
                  chat are dropped (once over MA_LG_CTX_CHARS, down to MA_LG_CTX_LOW of it) and its instance message
                  says how many; the agents
                  are told to keep what matters in files (notes, SCORES.md, BOARD.md).
  memory          in evaluation only the last turn's segment is kept (every turn re-renders the whole chat, so keeping
                  every segment would hold hundreds of full contexts per agent); generated tokens are counted apart.

Env: MA_LG_N (4), MA_LG_TOKENS (500000 per agent), MA_LG_CTX_CHARS (110000), MA_LG_CTX_LOW (1.0: a trim cuts the
history to this share of MA_LG_CTX_CHARS; 0.75 keeps the prefix cache for many steps), MA_LG_MAX_ROUNDS (2000, a safety cap), MA_LG_END (voluntary | budget: see END), MA_LG_HARNESS (v1: the first prompt and observations; v2: budget line, shell
facts, single-case and combined-finish notes; see HARNESS),
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
from . import fcs_cmd as C
from .fcs_rm import answer, judge_code
from .team_rollout import Aborted, Session

N = int(os.environ.get("MA_LG_N", "4"))
TOKENS = int(os.environ.get("MA_LG_TOKENS", "500000"))
CTX_CHARS = int(os.environ.get("MA_LG_CTX_CHARS", "110000"))
# a trim cuts the history to CTX_LOW x CTX_CHARS (hysteresis): the prompt prefix, and with it the server's prefix cache,
# then holds for many steps between trims. 1.0 = the first version: once full, every step trims and re-prefills it all
# (audit 2026-10-24: ~125k tokens re-prefilled per step past ~120k tokens of history)
CTX_LOW = float(os.environ.get("MA_LG_CTX_LOW", "1.0"))
assert 0 < CTX_LOW <= 1, CTX_LOW
MAX_ROUNDS = int(os.environ.get("MA_LG_MAX_ROUNDS", "2000"))
# the harness version (audit 2026-10-24). v1: the prompt and observations of the smokes and of w100-w105. v2: every
# observation and error message ends with the tokens used and left; the prompt states the shell's limits (/tmp, the
# 4 GB address space and ASan, the auto-commits) and, for a problem with one test case, that its provisional score is
# its final one; a finish command combined with others says that nothing was submitted. Do not pool v1 and v2 rows.
HARNESS = os.environ.get("MA_LG_HARNESS", "v1")
assert HARNESS in ("v1", "v2"), HARNESS
V2 = HARNESS == "v2"
# how an agent's work ends (audit 2026-10-24: Flint ended voluntarily at 16-46% of a 300k budget, so the 500k and 1M
# rows would repeat the 150k one). voluntary: the finish command on an unchanged solution.cpp ends it (the first
# protocol). budget: nothing ends it but the budget (or the context, or format errors); the finish command on an
# unchanged file says so and goes on, and three of those in a row end the agent as IdleFinish. Never pool the two.
END = os.environ.get("MA_LG_END", "voluntary")
assert END in ("voluntary", "budget"), END
BUDGET_END = END == "budget"
IDLE_FINISHES = 3
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
# v2's facts about the shell (no braces: _TEAM_ENV goes through .format). 50 of 866 smoke turns failed on /tmp; one
# episode spent ~110 steps on a heap crash after ASan could not start; Flint restored code by full rewrites.
_ENV_FS = (" Your current directory is the repository and the only place where you can create or change files: /tmp "
           "and every other path outside it cannot be written (/tmp cannot even be listed), so keep all files there, "
           "scratch files included, and use relative paths.")
_ENV_ASAN = (" Each command runs with a 4 GB address-space limit, so -fsanitize=address cannot start; to find memory "
             "errors use -fsanitize=undefined, -D_GLIBCXX_ASSERTIONS or gdb.")
assert C.CMD_MEM == 4 << 30, "_ENV_ASAN names the limit"
_ENV_COMMITS = (" After every command the repository's source files (.cpp, .py, .md, .txt and similar; not binaries or "
                "files over 1 MB) are committed, so `git log` lists every version and `git show <commit>:solution.cpp "
                "> solution.cpp` brings one back.")
_TOOLS = """

## Long work, submissions and messages

You have a budget of about {budget} generated tokens, and you should use it: keep improving your program for as long \
as you can make it better.{budget_shown} Three commands are provided by the harness; each must be the whole action, alone:

- `submit` or `submit <file>`: judges the file (default solution.cpp) right away and shows its PROVISIONAL score \
(0-100): {prov_rule}. You may submit as often as you like.{team_submit}
- `post <message>`: {post_text}
- `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`: submits solution.cpp, as `submit` does, and your work goes on.{fin_end}

When {when_end}, {final_rule} Older steps of this conversation are removed when it gets long, so keep what you need \
to remember in a notes file and re-read it; SCORES.md lists every submission and its provisional score."""
_FIN_END = " Run it on a solution.cpp that has not changed since your last submission to end your work."
_STEP6 = ("6. Submit with `submit` whenever you have something better, read the provisional score, and keep "
          "improving.\n7. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` also submits solution.cpp and shows its "
          "provisional score; your work then goes on, so keep improving and run it again." + _FIN_END)


_FINAL_RULE = ("the submission with the highest provisional score is judged on the OTHER half of the hidden tests; that "
               "final score is the result.")
_PROV_RULE = "the mean over half of the hidden tests, with the score of each of those tests and any compiler errors"
_SHARED_FINAL = ("the submission with the highest provisional score is the result. This problem has a single hidden "
                 "test: the provisional score of your best submission is also its final score.")


def instance(problem: str, name: str, shared: bool = False) -> str:
    """The instance message; shared (v2 only): the problem has one test case, which is provisional and final."""
    team = N > 1
    shared = shared and V2
    tools = _TOOLS.format(
        budget=f"{TOKENS:,}",
        budget_shown=(" The end of each observation shows how many tokens you have used; when they run out, your work "
                      "ends and only what you have submitted counts." if V2 else ""),
        prov_rule="its score on the single hidden test, with any compiler errors" if shared else _PROV_RULE,
        final_rule=_SHARED_FINAL if shared else _FINAL_RULE,
        team_submit=(" Every submission of every agent and its provisional score is shown to all agents." if team
                     else ""),
        post_text=("sends the message to every other agent; their messages appear at the top of your next "
                   "observations, and BOARD.md keeps them all. Use it to split the work, report what you found and "
                   "agree on the next step." if team else "there are no other agents, so this only writes BOARD.md."),
        fin_end="" if BUDGET_END else _FIN_END,
        when_end=(("every agent has used up its token budget" if team else "your token budget is used up") if BUDGET_END
                  else "every agent has finished" if team else "you finish"))
    steps = f" You have at most {A.STEPS} steps." if A.STEPS > 0 else ""
    env = _TEAM_ENV if team else _SOLO_ENV
    if V2:
        env += ("" if team else _ENV_COMMITS) + (_ENV_FS if C.J.LANDLOCK_ABI else "") + _ENV_ASAN
    text = (_FLAT.replace(_LONE_ENV, env).replace(_LONE_SUBMIT, _STEP6.replace(_FIN_END, "") if BUDGET_END else _STEP6))
    text = text.format(problem=problem, timeout=A.TIMEOUT, steps=steps, name=name, n=N)
    marker = "## Important Rules"
    assert marker in text
    return text.replace(marker, tools.strip() + "\n\n" + marker, 1)


# the last lines of the format errors (fcs_agent's text format; chat_eval_long's tool-call format), which say the
# finish command ends the task; v2 says what it does here
_FORMAT_ERROR_ENDS = ("If you have completed your assignment, please consult the first message about how to\nsubmit your "
                      "solution (you will not be able to continue working on this task after that).",
                      "If you want to end the task, please issue the following command: "
                      "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command.")
assert _FORMAT_ERROR_ENDS[0] in A.FORMAT_ERROR, "fcs_agent.FORMAT_ERROR changed"
_FORMAT_ERROR_END_V2 = ("To submit solution.cpp, issue `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (or `submit`) without "
                        "any other command; your work then goes on. Issued on an unchanged solution.cpp, it ends your "
                        "work.")
_FORMAT_ERROR_END_BUDGET = "To submit, run `submit` alone."
_MARK_CMD_RE = re.compile(r"\becho\s+['\"]?" + A.MARK)
_COMBINED_NOTE = ("\nNothing was submitted: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` works only as the whole command, "
                  "alone.\n")


def split_cases(n: int) -> tuple[list[int], list[int], bool]:
    """(provisional, final, shared): even-numbered cases are provisional, odd-numbered final; one case serves both."""
    if n <= 1:
        return [0], [0], True
    return list(range(0, n, 2)), list(range(1, n, 2)), False


def subset_mean(cases: list[float], idx: list[int]) -> float:
    vals = [cases[i] for i in idx if i < len(cases)]
    return sum(vals) / len(vals) if vals else 0.0


def harness_tag() -> str:
    """The settings that change what the agents see or how the episode ends, for results.jsonl and the trace: rows
    with different tags are different protocols."""
    return f"long-{HARNESS}{'-budget' if BUDGET_END else ''}-ctxlow{CTX_LOW:g}"


def sha1(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:12]


def trim_chat(chat: list, keep_chars: int, dropped: int, low_chars: int | None = None) -> tuple[list, int]:
    """Once the history after [system, instance] is over keep_chars, drop the oldest (assistant, user) pairs until it
    fits low_chars (default keep_chars); the instance message then says how many steps were removed. Returns (chat,
    total dropped)."""
    head, hist = chat[:2], chat[2:]
    n = [len(m["content"]) + len(m.get("reasoning_content") or "") for m in hist]   # reasoning kept in history (Flint)
    size = sum(n)
    if size <= keep_chars:
        return chat, dropped
    target = keep_chars if low_chars is None else min(low_chars, keep_chars)
    k = 0
    while size > target and len(hist) - 2 * k > 2:
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

    def __init__(self, input, j: int, problem: str, shared: bool = False):
        self.name = f"agent{j + 1}"
        self.sess = Session(input, self.name, A.THINK, max_len=A.MAX_LEN)
        self.chat = [{"role": "system", "content": A.SYSTEM},
                     {"role": "user", "content": instance(problem, self.name, shared)}]
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns, self.segs = False, "", [], []
        self.generated, self.dropped, self.seen, self.last_seg = 0, 0, 0, None
        self.n_submit = self.n_post = 0
        self.last_sha = None   # sha of solution.cpp at this agent's last submission of it
        self.idle_fin = 0      # unchanged finish commands in a row (MA_LG_END=budget)

    def gen(self) -> int:
        return self.generated

    def _usage(self) -> str:
        """v2: the line that ends every observation and error message (none in v1). It goes after the rendering, so
        the elision, the MARK check and the trace's output are unchanged."""
        if not V2:
            return ""
        return (f"\n<budget>{self.generated:,} of {TOKENS:,} generated tokens used; about "
                f"{max(0, TOKENS - self.generated):,} left.</budget>")

    async def think(self):
        """One model turn, as fcs_agent_team's, with this module's token budget and context trimming."""
        if self.generated >= TOKENS:
            self.done, self.exit = True, "TokensExceeded"
            return None
        self.chat, self.dropped = trim_chat(self.chat, CTX_CHARS, self.dropped, int(CTX_LOW * CTX_CHARS))
        self.steps += 1
        self.sess.last_messages = None
        n_seg, g0, t0 = len(self.sess.segments), self.generated, time.monotonic()
        text, cut = await self.sess.turn(self.chat, min(A.TURN, TOKENS - self.generated))
        # the turn's accounting for the trace: completion tokens, prompt + completion of the call (None without a new
        # segment, so a stale count is not reused), the history dropped so far, the wall time of the model call and,
        # from a session that records it (chat_eval_long.ChatSession.last_call), the server's own numbers
        acct = {"tokens": 0, "used": None, "dropped": self.dropped, "think_s": round(time.monotonic() - t0, 3),
                "t": round(time.time(), 1), **(getattr(self.sess, "last_call", None) or {})}
        for tok in A._SPECIAL:
            text = text.replace(tok, "")
        if len(self.sess.segments) > n_seg:
            seg = self.sess.segments[-1]
            self.generated += sum(seg.loss_mask)
            self.last_seg = seg
            if self.sess.input.evaluation:   # keep one segment: the episode returns only the last one
                self.sess.segments = [seg]
            acct.update(tokens=self.generated - g0, used=getattr(self.sess, "used", None))
        if not text and cut:
            self.done, self.exit = True, "ContextExceeded"
            self.turns.append({"step": self.steps, "text": "", "cut": True, **acct})
            return None
        vis = "" if cut else answer(text, A.THINK).strip()
        actions = [] if cut else [a.strip() for a in A.ACTION_RE.findall(vis)]
        rec = {"step": self.steps, "text": text, "cut": bool(cut), **acct}
        if len(actions) != 1:
            self.fmt_run, self.fmt_total, self.cut_n = self.fmt_run + 1, self.fmt_total + 1, self.cut_n + bool(cut)
            msg = A.LENGTH_ERROR if cut else A.FORMAT_ERROR.format(
                error=f"Expected exactly one action, found {len(actions)}.", n=len(actions))
            if V2 or BUDGET_END:   # the error's last line says the finish command ends the task; here it submits
                for end in _FORMAT_ERROR_ENDS:
                    msg = msg.replace(end, _FORMAT_ERROR_END_BUDGET if BUDGET_END else _FORMAT_ERROR_END_V2)
            self.turns.append({**rec, "format_error": True})
            self.chat = self.chat + [{"role": "assistant", "content": vis or "(cut off)"},
                                     {"role": "user", "content": msg + self._usage()}]
            if self.fmt_run >= A.MAX_FORMAT_ERRORS:
                self.done, self.exit = True, "RepeatedFormatError"
            return None
        self.fmt_run = 0
        self._pending = (vis, rec)
        return actions[0]

    def observe(self, cmd: str, out: dict, rnd: int, order: int) -> None:
        timing = out.get("_timing") or {}   # the episode's waits for this action (judge, command queue, harness)
        out = {k: v for k, v in out.items() if k != "_timing"}
        T._Agent.observe(self, cmd, out, rnd, order)
        if self.turns:
            self.turns[-1].update(seconds=round(out.get("seconds", 0.0), 3), **timing)
        if self.exit == "Submitted":   # the finish command on an unchanged solution.cpp (episode's finish()) ends it
            self.exit = "Finished"
        if not self.done and V2 and self.chat[-1]["role"] == "user":
            self.chat[-1] = {**self.chat[-1], "content": self.chat[-1]["content"] + self._usage()}


async def episode(input: GenerateFnInput, messages: list, label: str):
    from miles_team.fcs_judge import load_problem
    from pathlib import Path
    problem = A.problem_text(messages)
    n_cases = len(load_problem(Path(label))["cases"])
    prov_idx, final_idx, shared = split_cases(n_cases)
    agents = [_Agent(input, j, problem, shared) for j in range(N)]
    work = os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent_long", uuid.uuid4().hex)
    os.makedirs(work)
    loop = asyncio.get_running_loop()
    rng = random.Random(f"{label}:{input.sample.index}:{getattr(input.sample, 'group_index', 0)}")
    rounds, run_s, subs, board, events, t_ep = 0, 0.0, [], [], [], time.monotonic()

    def _put(name: str, text: str) -> None:
        """Write work/name for the agents to read without following what they put there (audit 2026-10-24: an
        `ln -sf ~/x SCORES.md` made the harness overwrite ~/x): a new file under a unique name, then a rename over
        whatever is at the path (a symlink is replaced, not followed). Never raises; on an error the log is stale."""
        path = os.path.join(work, name)
        tmp = os.path.join(work, f".{name}.{uuid.uuid4().hex}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(text)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)   # rmtree does not follow symlinks
            os.replace(tmp, path)
        except OSError:
            try:
                os.unlink(tmp)   # leave nothing for commit_as to pick up
            except OSError:
                pass

    def write_logs():
        _put("SCORES.md", "# Submissions (provisional score = mean over half of the hidden tests)\n\n"
             "| # | round | agent | file | status | provisional |\n|---|---|---|---|---|---|\n"
             + "".join(f"| {i} | {s['round']} | {s['agent']} | {s['file']} | {s['status']} | {100 * s['prov']:.2f} |\n"
                       for i, s in enumerate(subs, 1)))
        _put("BOARD.md", "# Messages\n\n" + "".join(f"- [round {r}] {a}: {m}\n" for r, a, m in board))

    def read_code(path: str):
        """(real path, text or None when larger than MAX_FILE) of a regular file inside the repository (a symlink out
        of it is refused), or None when there is no such file."""
        p = os.path.realpath(os.path.join(work, path))
        if not p.startswith(os.path.realpath(work) + os.sep) or not os.path.isfile(p):
            return None
        with open(p, "rb") as f:
            code = f.read(MAX_FILE + 1)
        return p, (None if len(code) > MAX_FILE else code.decode("utf-8", errors="replace"))

    async def do_submit(a, path: str) -> dict:
        got = read_code(path)
        if got is None:
            return {"output": f"submit: no file {path} in the repository\n", "returncode": 1, "exception_info": ""}
        p, code = got
        if code is None:
            return {"output": f"submit: {path} is larger than {MAX_FILE} bytes\n", "returncode": 1, "exception_info": ""}
        t0 = time.monotonic()
        res = await judge_code(label, code)
        judge_s = round(time.monotonic() - t0, 3)
        cases = res["cases"]
        prov, final = subset_mean(cases, prov_idx), subset_mean(cases, final_idx)
        rev = (await loop.run_in_executor(A._POOL, lambda: T._git(work, "rev-parse", "HEAD").stdout.strip()))
        s = {"agent": a.name, "round": rounds, "step": a.steps, "file": path, "status": res["status"], "prov": prov,
             "final": final, "infra": bool(res["infra_error"]), "rev": rev, "sha": sha1(code), "judge_s": judge_s,
             "code": code}
        subs.append(s)
        a.n_submit += 1
        if p == os.path.realpath(os.path.join(work, "solution.cpp")):
            a.last_sha = s["sha"]
        events.append((rounds, a.name, f"submitted {path}: {res['status']}, provisional {100 * prov:.2f}"))
        best = max(subs, key=lambda x: (x["prov"], -subs.index(x)))
        per = ", ".join(f"{100 * cases[i]:.0f}" for i in prov_idx if i < len(cases))
        msg = (f"Submission #{len(subs)} by {a.name}: {res['status']}. Provisional score {100 * prov:.2f} on "
               f"{len(prov_idx)} tests (per test: {per}). Best provisional so far: {100 * best['prov']:.2f} "
               f"(#{subs.index(best) + 1}, {best['agent']}).\n")
        if res["status"] == "compile error" and res.get("msg"):
            msg += "Compiler output:\n" + res["msg"][:3000] + "\n"
        if BUDGET_END and not V2:   # (v2's budget line says it on every observation)
            msg += f"About {max(0, TOKENS - a.generated):,} of your {TOKENS:,} tokens are left.\n"
        return {"output": msg, "returncode": 0, "exception_info": ""}

    async def finish(a, out: dict) -> dict:
        """The finish command: submit solution.cpp and go on, or (unchanged since a.last_sha) pass `out` on to end."""
        got = read_code("solution.cpp")
        if got is None:
            return {**out, "output": "Finish command received, but there is no solution.cpp to submit. Write "
                                     "solution.cpp and run the command again.\n"}
        if got[1] is not None and sha1(got[1]) == a.last_sha:
            if not BUDGET_END:
                return out
            a.idle_fin = getattr(a, "idle_fin", 0) + 1   # budget mode: the work goes on (episode() ends it after 3)
            k, x = next((i, x) for i, x in reversed(list(enumerate(subs, 1))) if x["agent"] == a.name
                        and x["sha"] == a.last_sha)
            return {"output": f"solution.cpp is unchanged since your submission #{k} (provisional {100 * x['prov']:.2f});"
                              f" nothing was submitted. About {max(0, TOKENS - a.generated):,} of your {TOKENS:,} "
                              "tokens are left. Keep improving solution.cpp.\n", "returncode": 0, "exception_info": ""}
        a.idle_fin = 0
        res = await do_submit(a, "solution.cpp")
        left = max(0, TOKENS - a.generated)
        tokens = "." if V2 else f": about {left:,} of your {TOKENS:,} tokens are left."   # v2: the budget line says it
        again = ("and run the command again to submit again.\n" if BUDGET_END else
                 "and run the command again to submit again; running it while solution.cpp is unchanged since your "
                 "last submission ends your work.\n")
        return {**res, "output": "Finish command received: solution.cpp was submitted.\n" + res["output"]
                + f"Your work goes on{tokens} Keep improving solution.cpp " + again}

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
                ends = False   # the finish command on an unchanged solution.cpp: finish() passes `out` on to end it
                fin = False    # the finish command ran (budget mode counts the unchanged ones in a row)
                timing, t0 = {}, time.monotonic()   # wall times from here, so waits for a worker are seen
                if m_sub:
                    out, secs = await do_submit(a, m_sub.group(1) or "solution.cpp"), 0.0
                    timing["judge_s"] = round(time.monotonic() - t0, 3)
                elif m_post:
                    out, secs = do_post(a, m_post.group(1)), 0.0
                else:
                    out = await loop.run_in_executor(A._POOL, A.run_command, cmd, work, True)
                    secs = out["seconds"]
                    wall = time.monotonic() - t0
                    timing.update(cmd_wall_s=round(wall, 3), cmd_queue_s=round(wall - secs, 3))
                    lines = out["output"].lstrip().splitlines()
                    if lines and lines[0].strip() == A.MARK and out["returncode"] == 0:
                        fin = True
                        t1 = time.monotonic()
                        res = await finish(a, out)
                        ends = res is out
                        out = {**res, "seconds": secs}
                        timing["judge_s"] = round(time.monotonic() - t1, 3)
                    elif V2 and _MARK_CMD_RE.search(cmd):   # combined with other commands: say it did not submit
                        out = {**out, "output": out["output"] + _COMBINED_NOTE}
                run_s += secs
                t1 = time.monotonic()
                await loop.run_in_executor(A._POOL, write_logs)
                await loop.run_in_executor(A._POOL, T.commit_as, work, a.name, a.steps)
                timing["harness_s"] = round(time.monotonic() - t1, 3)
                upd = updates_for(a)
                if upd and not ends:   # an ending observation keeps the MARK as its first line
                    out = {**out, "output": upd + out["output"]}
                a.observe(cmd, {**out, "_timing": timing}, rounds, k)
                if not fin:
                    a.idle_fin = 0
                elif BUDGET_END and getattr(a, "idle_fin", 0) >= IDLE_FINISHES and not a.done:
                    a.done, a.exit = True, "IdleFinish"
        # what the budget cut costs (audit 2026-10-24): an agent that stopped without the finish command (tokens,
        # context, format errors, the round cap) may leave a better solution.cpp unsubmitted. It is judged once, apart
        # from subs: S keeps the documented rule (only submissions count) and S_auto - S measures the cut.
        auto_sub = []
        if any(a.exit != "Finished" for a in agents):
            got = read_code("solution.cpp")
            if got is not None and got[1] and got[1].strip() and sha1(got[1]) not in {x["sha"] for x in subs}:
                res = await judge_code(label, got[1])
                auto_sub.append({"agent": "auto", "round": rounds, "file": "solution.cpp", "status": res["status"],
                                 "prov": subset_mean(res["cases"], prov_idx), "final": subset_mean(res["cases"], final_idx),
                                 "infra": bool(res["infra_error"]), "sha": sha1(got[1]), "auto": True, "code": got[1]})
        best = max(subs, key=lambda x: (x["prov"], -subs.index(x))) if subs else None
        cand = subs + auto_sub
        best_auto = max(cand, key=lambda x: (x["prov"], -cand.index(x))) if cand else None
        shares = (await loop.run_in_executor(A._POOL, T.blame_shares, work, best["file"], best["rev"])) if best else {}
        log = (await loop.run_in_executor(A._POOL, lambda: T._git(work, "log", "--format=%an %s").stdout))[-20000:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    S = best["final"] if best else 0.0
    oracle = max((s["final"] for s in subs), default=0.0)

    def total_min(key: str) -> float:   # summed over every agent's turns
        return sum(t.get(key) or 0.0 for a in agents for t in a.turns) / 60
    first = subs[0]["round"] if subs else -1
    info = {"lg_harness": harness_tag(), "fo_S": S, "fo_R": S, "fo_lg_n": N, "fo_lg_shared_case": float(shared), "fo_lg_rounds": rounds,
            "fo_lg_submissions": len(subs),
            "fo_lg_S_auto": best_auto["final"] if best_auto else 0.0, "fo_lg_auto_judged": float(bool(auto_sub)),
            "fo_lg_best_prov": best["prov"] if best else 0.0, "fo_lg_oracle_final": oracle,
            "fo_lg_selection_loss": oracle - S, "fo_lg_first_submit_round": first,
            "fo_lg_posts": len(board), "fo_lg_tokens": sum(a.generated for a in agents) / N,
            "fo_tokens": sum(a.generated for a in agents), "fo_latency": max(a.generated for a in agents),
            "fo_ag_steps": sum(a.steps for a in agents) / N, "fo_ag_format_errors": sum(a.fmt_total for a in agents) / N,
            "fo_ag_run_min": run_s / 60, "fo_ag_compile_fail": float(bool(best) and best["status"] == "compile error"),
            "fo_ag_no_code": float(best is None), "fo_ag_top_author": max(shares.values()) if shares else 0.0,
            **{f"fo_ag_exit_{e}": sum(a.exit == e for a in agents) / N for e in
               ("Finished", "IdleFinish", "TokensExceeded", "ContextExceeded", "RepeatedFormatError")},
            "fo_judge_infra_error": float(any(s["infra"] for s in subs)),
            # where the episode's wall time went (audit 2026-10-24): model calls as the server saw them (llm) and as the
            # agent waited for them (think), commands' waits for a worker, the judge, the harness's logs and commits
            "fo_lg_wall_min": (time.monotonic() - t_ep) / 60, "fo_lg_llm_min": total_min("llm_s"),
            "fo_lg_think_min": total_min("think_s"), "fo_lg_cmd_queue_min": total_min("cmd_queue_s"),
            "fo_lg_judge_min": total_min("judge_s"), "fo_lg_harness_min": total_min("harness_s")}
    A._trace({"time": round(time.time(), 1), "eval": input.evaluation, "game": "agent_long", "harness": harness_tag(),
              "label": label, "S": S,
              "n": N, "budget": TOKENS, "rounds": rounds, "shared_case": shared, "n_cases": n_cases,
              "submissions": subs, "best": {k: v for k, v in best.items() if k != "code"} if best else None,
              "auto_submission": auto_sub[0] if auto_sub else None, "board": board, "blame": shares, "git_log": log,
              # full turns (text = <think>reasoning</think>answer) and submissions with their code (audit 2026-10-24:
              # the last 4000 characters of a turn hid the reasoning; ~4 MB per 1M-token episode)
              "agents": [{"name": a.name, "exit": a.exit, "steps": a.steps, "generated": a.generated,
                          "dropped": a.dropped, "submits": a.n_submit, "posts": a.n_post, "turns": a.turns}
                         for a in agents]})
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
