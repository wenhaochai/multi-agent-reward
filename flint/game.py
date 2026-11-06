"""The long game (user 2026-10-21: "these agents never work long enough; they need to submit again and again and talk
all the time"). N mini-swe-agents work on one Frontier-CS problem in lockstep rounds: in each round every agent that
is still working takes one model turn (in parallel), then the commands run one after another in a random order (seeded
by the problem and the episode). N = 1 is the lone-agent control.

  submit [file]   (the whole action) judges the file (default solution.cpp) at once. The agent sees its PROVISIONAL score:
                  the mean over the cases at 0-based even indices (files 1.in, 3.in, ...), with per-case scores and
                  compiler errors, like a contest's provisional board. Every submission and its provisional score goes
                  to SCORES.md and to every agent. An agent may submit as often as it likes. A judge infrastructure
                  failure is judged again (twice, after 2 s and 4 s); one that stays is told to the agent and never
                  picked as best unless all are.
  post <message>  (the whole action) sends a message to every other agent. New messages and new submissions of the
                  others are shown at the top of each agent's next observation; BOARD.md keeps the whole log. In a team,
                  after each of its submissions an agent's next command must be a post with a message: any other
                  command is not run, and the reply asks for the post (four team smokes, 32 agents and 3,196 commands,
                  had no post while posting was free).
  finish          `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (mini-swe-agent's submit, which Flint uses to submit)
                  submits solution.cpp as `submit` does, and the agent goes on.
  the end         only the token budget (MA_LG_TOKENS generated tokens per agent), the context or the round cap end an
                  agent's work. A stop attempt (the finish command on an unchanged solution.cpp, or three format errors
                  in a row) starts the agent's conversation over, its files kept, with a note that the work goes on
                  (one Flint at 500k ended its work after 107k tokens on average when it could; told to go on in the
                  same conversation, it only resubmitted). When every agent has stopped, the submission with the
                  highest provisional score (the earliest on a tie) is the team's answer, and its FINAL score (the mean
                  over the cases at 0-based odd indices, files 2.in, 4.in, ..., never shown to the agents) is the
                  episode's score. A problem with a single test case uses it for both, so its final score is not held
                  out (flagged "shared_case" in the trace, fo_lg_shared_case in info). The unsubmitted solution.cpp an
                  agent leaves is judged apart: fo_lg_S_auto is the score had it been submitted (S counts only
                  submissions).
  repositories    each agent of a team works in a repository of its own, shared through team.git (repo.py); a lone
                  agent works in one repository.
  context         a long episode overflows the context, so before each step the oldest steps of an agent's chat are
                  dropped (once over MA_LG_CTX_CHARS, down to MA_LG_CTX_LOW of it, so the server's prefix cache holds
                  between trims) and its instance message says how many; the agents are told to keep what matters in
                  files (notes, SCORES.md, BOARD.md).
  dedup           (MA_LG_DEDUP_SUBMIT=1; off by default) a `submit` of a file whose text equals one of the agent's own
                  earlier submissions that the judge finished (not an infra failure), apart from comments and layout
                  (code_key; owner popup 2026-11-05, after w1090's agents resubmitted one solution up to 1,037 times
                  with a new "// finalN" line or blank line each time), is not judged and not recorded:
                  the reply names that submission and its provisional score and asks for a change first, so S, the
                  best provisional and every score stay as they would be. The finish command keeps its own rule (an
                  unchanged solution.cpp starts the conversation over). The trace counts them per agent
                  (dup_submits), info as fo_lg_dup_submits (per agent) (trace analysis 2026-10-31: in run20 rollout
                  17 one episode submitted the same solution.cpp 124 times in a row, 14k tokens and 6.6 min of
                  judging, with a positive advantage; identical resubmissions rose from 25% to 43% over run20).
  checkpoint      an episode saves its state at the end of a round, every MA_LG_CKPT_S seconds and after every round
                  with a submission, and a later call with the same Checkpoint path resumes from it: a 1M-token episode
                  outlasts one wq segment.

Env: MA_LG_N (4 agents), MA_LG_TOKENS (125000 per agent), MA_LG_CTX_CHARS (400000), MA_LG_CTX_LOW (0.75),
MA_LG_MAX_ROUNDS (20000, a safety cap), MA_LG_CKPT_S (300), MA_LG_DEDUP_SUBMIT (0 = off; 1), MA_AG_TURN (32768 tokens
per turn), MA_AG_MAX_LEN (262144 context tokens), MA_AG_TIMEOUT (60 s per command), MA_AG_TRACE_DIR (traces; none if
unset), MA_AG_CMD_WORKERS (32),
MA_LG_WALL_MIN (0: none; RL rollouts set it: an episode that is still playing after that many minutes of wall time ends
there, like at its token budget, so one slow episode cannot keep the other engines of its group idle).
"""
from __future__ import annotations

import asyncio
import functools
import glob
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts as P
from . import repo as R
from . import sandbox as C
from .chat import ACTION_RE
from .judge import judge_code, load_problem

N = int(os.environ.get("MA_LG_N", "4"))
TEAM = N > 1
TOKENS = int(os.environ.get("MA_LG_TOKENS", "125000"))
CTX_CHARS = int(os.environ.get("MA_LG_CTX_CHARS", "400000"))
# a trim cuts the history to CTX_LOW x CTX_CHARS (hysteresis): the prompt prefix, and with it the server's prefix cache,
# then holds for many steps between trims (audit 2026-10-24: at 1.0 every step past the limit re-prefilled it all)
CTX_LOW = float(os.environ.get("MA_LG_CTX_LOW", "0.75"))
WALL_MIN = float(os.environ.get("MA_LG_WALL_MIN", "0"))
assert 0 < CTX_LOW <= 1, CTX_LOW
MAX_ROUNDS = int(os.environ.get("MA_LG_MAX_ROUNDS", "20000"))
CKPT_S = float(os.environ.get("MA_LG_CKPT_S", "300"))
DEDUP_SUBMIT = os.environ.get("MA_LG_DEDUP_SUBMIT") or "0"   # 0: every submit is judged (every game before 2026-10-31)
if DEDUP_SUBMIT not in ("0", "1"):
    raise ValueError(f"MA_LG_DEDUP_SUBMIT={DEDUP_SUBMIT}: must be 0 (off) or 1")
DEDUP_SUBMIT = DEDUP_SUBMIT == "1"
# MA_LG_DUP_END=K (0 = off, the game as before; needs MA_LG_DEDUP_SUBMIT=1): the K-th unjudged duplicate submit in a row
# since the agent's last judged submission ends the agent's game at that turn (exit DupLoop); the instance message
# states the rule, and in RL that turn's reward is 0 (flint.miles_agent.split_dup_end_turns, reward_post_process) while
# the earlier turns keep S. Owner popups 2026-11-06 (quoted in flint/miles_agent.split_dup_end_turns): penalize this in
# RL too; state the rule in the prompt, score the turn that triggers it 0 and every earlier turn as before. On the 959
# val40 episodes of 2026-11-06, K = 50 ends 38 loops (e_a3's problem 6: 391 duplicates in 5.6 h) and changes no score:
# none raised its best provisional score after that point.
DUP_END = int(os.environ.get("MA_LG_DUP_END") or "0")
if DUP_END < 0 or (DUP_END and not DEDUP_SUBMIT):
    raise ValueError(f"MA_LG_DUP_END={DUP_END}: 0 (off), or a positive count together with MA_LG_DEDUP_SUBMIT=1")
TURN = int(os.environ.get("MA_AG_TURN", "32768"))
MAX_LEN = int(os.environ.get("MA_AG_MAX_LEN", "262144"))
MAX_FORMAT_ERRORS = 3
MAX_POST = 2000               # characters of one message
INFRA_RETRIES, INFRA_BACKOFF = 2, 2.0   # a judge infra failure is judged again, after 2 s, then 4 s
MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMIT_RE = re.compile(r"\s*submit(?:\s+(\S+))?\s*")
POST_RE = re.compile(r"\s*post\s+(.+?)\s*", re.S)
_SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("MA_AG_CMD_WORKERS", "32")))   # commands and the harness's git

# the problem prompt of the data files (one-answer rule + "Generate solution code:"): the statement is what lies between
_PREFIX_RE = re.compile(r"^You are a competitive programmer\. Solve the following problem in C\+\+\..*?\n\n", re.DOTALL)
_SUFFIX = "Generate solution code:"


def problem_text(messages: list) -> str:
    text = messages[-1]["content"]
    text = _PREFIX_RE.sub("", text, count=1).rstrip()
    if text.endswith(_SUFFIX):
        text = text[: -len(_SUFFIX)].rstrip()
    return text


def answer(text: str) -> str:
    """The answer part of a reply: the text after the last </think>; a reply that never closed its reasoning has
    none."""
    if "</think>" not in (text or ""):
        return ""
    return text.rpartition("</think>")[2]


def instance(problem: str, name: str) -> str:
    return P.instance(problem, name, N, TOKENS, C.TIMEOUT, dup_end=DUP_END)


def split_cases(n: int) -> tuple[list[int], list[int], bool]:
    """(provisional, final, shared): cases at 0-based even indices (files 1.in, 3.in, ...) are provisional, odd
    indices (2.in, 4.in, ...) final; a single case serves both (no held-out final)."""
    if n <= 1:
        return [0], [0], True
    return list(range(0, n, 2)), list(range(1, n, 2)), False


def best_of(subs: list[dict]) -> dict:
    """The submission with the highest provisional score, the earliest on a tie; one the judge could not finish
    (infra) only when every submission is one, so an infra zero never lowers the episode's score by being picked."""
    return max(enumerate(subs), key=lambda t: (not t[1]["infra"], t[1]["prov"], -t[0]))[1]


def subset_mean(cases: list[float], idx: list[int]) -> float:
    vals = [cases[i] for i in idx if i < len(cases)]
    return sum(vals) / len(vals) if vals else 0.0


def harness_tag() -> str:
    """The game's name in results.jsonl and the trace: the tag of the same game before the cleanup of 2026-10-28
    (harness v1, budget end, fresh restarts; a team: own repositories, must post)."""
    return (f"long-v1-budget-fresh-ctxlow{CTX_LOW:g}" + ("-own-mustpost" if TEAM else "")
            + ("-dedup" if DEDUP_SUBMIT else "") + (f"-dupend{DUP_END}" if DUP_END else ""))


def sha1(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:12]


_LIT_OR_COMMENT = re.compile("|".join((
    r'R"([^()\\\s"]{0,16})\([\s\S]*?(?:\)\1"|\Z)',   # a raw string (one that is not closed runs to the end, as in g++)
    r'"(?:\\.|[^"\\\n])*"',                           # a string
    r"'(?:\\.|[^'\\\n])*'",                           # a character literal
    r"//[^\n]*",                                      # a line comment
    r"/\*[\s\S]*?(?:\*/|\Z)")))                       # a block comment (one that is not closed runs to the end)
# what code_key does not read, so such a text is compared as it is: NUL (the key's own placeholder), whitespace that
# str.strip and g++ read apart, a lone CR (g++ ends a line there), a line splice (a backslash at a line's end), a digit
# separator (the ' of 1'000 opens no character literal)
_RAW_TEXT = re.compile(r"[\x00\x1c-\x1f\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]|\r(?!\n)"
                       r"|\\[ \t\f\v]*\r?\n|\d'\w")
_OPS = set("+-*/%<>=!&|^~.:?#")


def _word(c: str) -> bool:
    return c.isalnum() or c in "_$\\\0"   # $ and the backslash of a universal character name; \0: a literal's
    # placeholder (u8 "x" is not u8"x")


def _space(m) -> str:
    """A space stays where taking it out could change the tokens: between two word characters or two operator
    characters (a b, x - -y), in an exponent's place (0xe - 1, 1e -5, 1e- 5) and next to a dot (case 1 ... 5:, x .5,
    1. e5)."""
    a, b = m.group(1), m.group(2)
    s, i = m.string, m.start()   # s[i] is the space, s[i - 1] is a
    keep = ((_word(a) and _word(b)) or (a in _OPS and b in _OPS) or (a in "eEpP" and b in "+-")
            or (_word(a) and b == ".") or (a == "." and b.isdigit())
            or (a in "+-" and i >= 3 and s[i - 2] in "eEpP" and (s[i - 3].isdigit() or s[i - 3] == "."))
            or (a == "." and b.isalpha() and i >= 2 and s[i - 2].isdigit()))
    return " " if keep else ""


@functools.lru_cache(maxsize=4096)
def code_key(code: str) -> str:
    """What the dedup compares: the code without its comments (a block comment counts as a space, as in C++), with no
    blank lines and each line's whitespace collapsed to single spaces; outside preprocessor lines a space also goes
    where it separates nothing (_space). String and character literals stay as written, and so does every remaining
    line break, so a preprocessor line keeps its end, and a preprocessor line keeps its spaces (#define F(x) is not
    #define F (x)). A text with what the key does not read (_RAW_TEXT) is its own key. Two texts with one key differ
    only in comments or layout (the audit of 2026-11-05 compiled its pairs with g++). Cached, since a long episode
    compares each submit with all its earlier ones."""
    if _RAW_TEXT.search(code):
        return "\0" + code
    lits = []

    def keep(m):
        t = m.group(0)
        if t.startswith("//"):
            return ""
        if t.startswith("/*"):
            return " "
        lits.append(t)
        return f"\0{len(lits) - 1}\0"
    lines = []
    for line in _LIT_OR_COMMENT.sub(keep, code).split("\n"):
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip(" ")
        if line and not line.startswith(("#", "%:")):
            line = re.sub(r"(?<=(.)) (?=(.))", _space, line)
        if line:
            lines.append(line)
    return re.sub(r"\0(\d+)\0", lambda m: lits[int(m.group(1))], "\n".join(lines))


def trace(rec: dict) -> None:
    d = os.environ.get("MA_AG_TRACE_DIR", "")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"agent_{os.getpid()}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def trim_chat(chat: list, keep_chars: int, dropped: int, low_chars: int | None = None) -> tuple[list, int]:
    """Once the history after [system, instance] is over keep_chars, drop the oldest (assistant, user) pairs until it
    fits low_chars (default keep_chars); the instance message then says how many steps were removed. Returns (chat,
    total dropped)."""
    head, hist = chat[:2], chat[2:]
    n = [len(m["content"]) + len(m.get("reasoning_content") or "") for m in hist]   # reasoning stays in the history
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
    base = head[1]["content"].split(P.TRIM_SEP)[0]
    return [head[0], {"role": "user", "content": base + P.TRIM_NOTE.format(dropped=dropped)}] + hist[2 * k:], dropped


class Agent:
    """One agent: its chat, its model session (chat.ChatSession) and its counters. A turn's reasoning goes back into the
    history as `reasoning_content` (Flint's card asks for it), and trim_chat counts it against MA_LG_CTX_CHARS."""
    restarts = 0       # conversations started over
    owe_post = False   # a team agent has submitted and not posted since
    post_refused = 0   # commands not run because a post was owed
    dup_submits = 0    # submits not judged: the file equals one of this agent's judged submissions (MA_LG_DEDUP_SUBMIT)
    dup_run = 0        # of them, those in a row since this agent's last judged submission (MA_LG_DUP_END)

    def __init__(self, sess, j: int, problem: str):
        self.name = f"agent{j + 1}"
        self.sess = sess
        self.chat = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": instance(problem, self.name)}]
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns = False, "", []
        self.generated, self.dropped, self.seen = 0, 0, 0
        self.n_submit = self.n_post = 0
        self.last_sha = None   # sha of solution.cpp at this agent's last submission of it
        self.idle_fin = 0      # unchanged finish commands in a row

    def restart(self, note: str) -> None:
        """The conversation starts over from the system message and the instance message with `note` (an earlier
        restart's note and the trim note go, and `dropped` counts from 0 again); the session forgets the old
        conversation's calls."""
        base = self.chat[1]["content"].split(P.TRIM_SEP)[0].split(P.RESTART_SEP)[0]
        self.chat = [self.chat[0], {"role": "user", "content": base + note}]
        self.restarts, self.dropped, self.fmt_run, self.idle_fin = self.restarts + 1, 0, 0, 0
        self.sess.forget()

    def _keep_reasoning(self) -> None:
        """The assistant message the last model turn added keeps that turn's reasoning."""
        m = self.chat[-2] if len(self.chat) >= 4 else None
        if self.sess.fresh and m is not None and m.get("role") == "assistant" and "reasoning_content" not in m:
            self.chat[-2] = {**m, "reasoning_content": self.sess.last_reasoning}
            self.sess.fresh = False

    async def think(self):
        """One model turn: the command to run, or None (a format error, the budget, or a full context)."""
        self.sess.last_tc = None
        cmd = await self._think()
        if cmd is not None and self.sess.last_tc is not None:
            cmd = self.sess.last_tc[0].strip()   # the whole command: ACTION_RE stops at a line of ```
        if cmd is None:   # a format or length error added (assistant, error) now
            self._keep_reasoning()
        return cmd

    async def _think(self):
        if self.generated >= TOKENS:
            self.done, self.exit = True, "TokensExceeded"
            return None
        self.chat, self.dropped = trim_chat(self.chat, CTX_CHARS, self.dropped, int(CTX_LOW * CTX_CHARS))
        self.steps += 1
        g0, t0 = self.generated, time.monotonic()
        text, cut = await self.sess.turn(self.chat, min(TURN, TOKENS - self.generated))
        # the turn's accounting for the trace: completion tokens, prompt + completion of the call (None without a model
        # call, so a stale count is not reused), the history dropped so far, the wall time of the model call and the
        # server's own numbers (ChatSession.last_call)
        acct = {"tokens": 0, "used": None, "dropped": self.dropped, "think_s": round(time.monotonic() - t0, 3),
                "t": round(time.time(), 1), **(self.sess.last_call or {})}
        for tok in _SPECIAL:
            text = text.replace(tok, "")
        if self.sess.completion is not None:
            self.generated += self.sess.completion
            acct.update(tokens=self.generated - g0, used=self.sess.used)
        if not text and cut:
            self.done, self.exit = True, "ContextExceeded"
            self.turns.append({"step": self.steps, "text": "", "cut": True, **acct})
            return None
        vis = "" if cut else answer(text).strip()
        actions = [] if cut else [a.strip() for a in ACTION_RE.findall(vis)]
        rec = {"step": self.steps, "text": text, "cut": bool(cut), **acct}
        if len(actions) != 1:
            self.fmt_run, self.fmt_total, self.cut_n = self.fmt_run + 1, self.fmt_total + 1, self.cut_n + bool(cut)
            msg = P.LENGTH_ERROR if cut else P.FORMAT_ERROR.format(
                error=f"Expected exactly one action, found {len(actions)}.")
            self.turns.append({**rec, "format_error": True})
            self.chat = self.chat + [{"role": "assistant", "content": vis or "(cut off)"}, {"role": "user", "content": msg}]
            return None   # three in a row: the episode starts the conversation over after this round
        self.fmt_run = 0
        self._pending = (vis, rec)
        return actions[0]

    def observe(self, cmd: str, out: dict, rnd: int, order: int) -> None:
        timing = out.get("_timing") or {}   # the episode's waits for this action (judge, command queue, harness)
        out = {k: v for k, v in out.items() if k != "_timing"}
        n = len(self.chat)
        vis, rec = self._pending
        self.timeouts += "timed out" in out["exception_info"]
        self.turns.append({**rec, "round": rnd, "order": order, "command": cmd, "returncode": out["returncode"],
                           "output": out["output"][:20000], "exception": out["exception_info"],
                           "seconds": round(out.get("seconds", 0.0), 3), **timing})
        self.chat = self.chat + [{"role": "assistant", "content": vis}, {"role": "user", "content": P.observation(out)}]
        self._keep_reasoning()
        if self.sess.cont:   # a call an earlier response made: joined to that response's tool calls (chat.to_api)
            self.chat[-2] = {**self.chat[-2], "_cont": True}
        if self.sess.last_tc is not None and len(self.chat) == n + 2:   # chat.to_api sends the call as Flint made it
            self.chat[-2] = {**self.chat[-2], "_cmds": [cmd], "_fn": self.sess.last_tc[1]}


class Paused(Exception):
    """The caller set ckpt.stop: the episode saved its state at the end of a round and stopped; a rerun resumes."""


_AGENT_STATE = ("chat", "steps", "fmt_run", "fmt_total", "timeouts", "cut_n", "done", "exit", "turns", "generated",
                "dropped", "seen", "n_submit", "n_post", "last_sha", "idle_fin", "restarts", "owe_post", "post_refused",
                "dup_submits", "dup_run")
_SESS_STATE = ("used", "queue", "cont", "n_cont", "last_reasoning", "fresh")   # ChatSession's state
_TAR_MAX = 32 << 20   # a larger file (a generated test input) stays out of the checkpoint; the agent can rebuild it


def _tar_member(ti: tarfile.TarInfo):
    if TEAM and ti.name.split("/")[1:2] and ti.name.split("/")[1].endswith(".tmp"):   # an agent's /tmp starts empty
        return None
    return ti if (ti.isdir() or ti.issym() or (ti.isfile() and ti.size <= _TAR_MAX)) else None


def _untar_member(ti: tarfile.TarInfo, dest: str):
    try:   # tarfile's "data" rules (no absolute paths, nothing outside dest); a member that breaks them is skipped
        return tarfile.data_filter(ti, dest)
    except tarfile.FilterError:
        return None


class Checkpoint:
    """An episode's saved state: <path>.r<round>.tgz holds the work dir (.git included) and <path>.pkl the rest. The
    tar is written first under a new name, then the pickle (atomically) names it, then older tars go: a kill at any
    point leaves a pickle and the tar of the same round. stop: set by the caller; the episode saves and raises Paused
    at the end of its round. prior_s: wall seconds of the earlier runs; saves: how many this run made; failed: how many
    saves failed (a full quota, a file the agent made unreadable or that shrank while it was read): the episode goes on
    and the last good save stays."""

    def __init__(self, path, every: float = CKPT_S):
        self.path, self.every, self.stop = str(path), every, False
        self.t0 = self.saved_at = time.time()
        self.prior_s, self.saves, self.failed = 0.0, 0, 0

    def exists(self) -> bool:
        return os.path.exists(self.path + ".pkl")

    def load(self) -> dict | None:
        try:
            with open(self.path + ".pkl", "rb") as f:
                st = pickle.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:   # unreadable: start fresh
            print(f"checkpoint {self.path}: unreadable ({e!r}), starting fresh", flush=True)
            return None
        return st   # the episode sets prior_s once the state matches and its tar is restored

    def due(self, submitted: bool) -> bool:
        return self.stop or submitted or time.time() - self.saved_at >= self.every

    def save(self, state: dict, work: str) -> bool:
        """Never raises: a failed save logs, removes its own partial files and keeps the last good save (the next try
        comes after `every` seconds, or at the next submission or stop)."""
        tgz = f"{self.path}.r{state['rounds']}.tgz"
        named = False   # the pickle names the new tar: from here on the save is good
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with tarfile.open(tgz + ".tmp", "w:gz", compresslevel=1) as t:
                t.add(work, arcname=".", filter=_tar_member)
            os.replace(tgz + ".tmp", tgz)
            state = {**state, "tar": os.path.basename(tgz), "wall_s": self.prior_s + time.time() - self.t0}
            with open(self.path + ".pkl.tmp", "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(self.path + ".pkl.tmp", self.path + ".pkl")
            named = True
            for old in glob.glob(glob.escape(self.path) + ".r*.tgz"):
                if os.path.basename(old) != state["tar"]:
                    os.remove(old)
        except Exception as e:
            print(f"checkpoint {self.path}: save after round {state['rounds']} "
                  f"{'kept, old tars left' if named else 'failed, the last good save stays'} ({e!r})", flush=True)
            if not named:
                for f in (tgz + ".tmp", tgz, self.path + ".pkl.tmp"):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                self.saved_at, self.failed = time.time(), self.failed + 1
                return False
        self.saved_at, self.saves = time.time(), self.saves + 1
        return True

    def restore(self, st: dict, work: str) -> None:
        with tarfile.open(os.path.join(os.path.dirname(self.path), st["tar"]), "r:gz") as t:
            t.extractall(work, filter=_untar_member)

    def clear(self) -> None:
        for f in [self.path + ".pkl", self.path + ".pkl.tmp"] + glob.glob(glob.escape(self.path) + ".r*.tgz*"):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


def rm_work(work: str) -> None:
    """Remove an episode's repository and its commands' /tmp."""
    for d in (work, C.private_tmp(work)):
        shutil.rmtree(d, ignore_errors=True)


async def episode(input, messages: list, label: str, new_session, ckpt: Checkpoint | None = None):
    """One episode on problem `label`. input carries sample.index and sample.group_index (the episode number), which
    seed the command order; new_session(input, name) makes an agent's model session (chat.ChatSession). Returns
    (agents, S, info)."""
    problem = problem_text(messages)
    n_cases = len(load_problem(Path(label))["cases"])
    prov_idx, final_idx, shared = split_cases(n_cases)
    agents = [Agent(new_session(input, f"agent{j + 1}"), j, problem) for j in range(N)]   # a resume restores them
    if ckpt is None:
        work = os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent_long", uuid.uuid4().hex)
    else:   # one name per checkpoint, so paths the agent wrote still hold after a resume in the same job
        work = os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent_long", os.path.basename(ckpt.path))
        rm_work(work)
    os.makedirs(work)
    for a in agents:   # not in the checkpoint: a resume in another job has another TMPDIR
        a.dir = os.path.join(work, a.name) if TEAM else work
    dir_of = {a.name: a.dir for a in agents}
    loop = asyncio.get_running_loop()
    rng = random.Random(f"{label}:{input.sample.index}:{getattr(input.sample, 'group_index', 0)}")
    rounds, run_s, subs, board, events, t_ep = 0, 0.0, [], [], [], time.monotonic()
    # a team: harness git trouble, solution.cpp contents by first owner (sha1 -> agent), each agent's last one, the
    # whole-file adoptions (round, agent, from), and this action's submissions whose commit is named after own_commit
    sync, origin, last_sol, adoptions, pending = {"fetch_fail": 0, "commit_skipped": 0, "git_fail": 0}, {}, {}, [], []
    # what a checkpoint must match: the episode and every setting that shapes its prompts, budget or ending; the keys
    # and values of the same game before the cleanup of 2026-10-28, so this code resumes its checkpoints
    config = {"v": 2, "label": label, "index": input.sample.index, "group": getattr(input.sample, "group_index", 0),
              "N": N, "TOKENS": TOKENS, "CTX_CHARS": CTX_CHARS, "CTX_LOW": CTX_LOW, "HARNESS": "v1", "END": "budget",
              "VERDICTS": False, "TURN": TURN, "MAX_LEN": MAX_LEN, "ON_STOP": "fresh",
              "prompt": sha1(P.SYSTEM + instance(problem, "agent1")),
              **({"REPOS": "own", "POST": "submit"} if TEAM else {}), **({"WALL_MIN": WALL_MIN} if WALL_MIN else {}),
              **({"DEDUP_SUBMIT": True} if DEDUP_SUBMIT else {})}

    def state() -> dict:   # whole turns: the trace keeps them whole too
        return {"config": config, "rounds": rounds, "run_s": run_s, "subs": subs, "board": board, "events": events,
                "rng": rng.getstate(), "wall_ep": time.monotonic() - t_ep,
                **({"sync": sync, "origin": origin, "last_sol": last_sol, "adoptions": adoptions} if TEAM else {}),
                "agents": [{"agent": {k: getattr(a, k) for k in _AGENT_STATE},
                            "sess": {k: getattr(a.sess, k) for k in _SESS_STATE}} for a in agents]}

    def _put(name: str, text: str, d: str) -> None:
        """Write d/name (d: an agent's repository) for the agents to read without following what they put there
        (audit 2026-10-24: an `ln -sf ~/x SCORES.md` made the harness overwrite ~/x): a new file under a unique name,
        then a rename over whatever is at the path (a symlink is replaced, not followed). Never raises; on an error the
        log is stale."""
        path = os.path.join(d, name)
        tmp = os.path.join(d, f".{name}.{uuid.uuid4().hex}")
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
        for d in dict.fromkeys(a.dir for a in agents):
            _put("SCORES.md", "# Submissions (provisional score = mean over half of the hidden tests)\n\n"
                 "| # | round | agent | file | status | provisional |\n|---|---|---|---|---|---|\n"
                 + "".join(f"| {i} | {s['round']} | {s['agent']} | {s['file']} | {s['status']} | {100 * s['prov']:.2f} |\n"
                           for i, s in enumerate(subs, 1)), d)
            _put("BOARD.md", "# Messages\n\n" + "".join(f"- [round {r}] {a}: {m}\n" for r, a, m in board), d)

    def read_code(path: str, d: str):
        """(real path, text or None when larger than MAX_FILE) of a regular file inside the repository (a symlink out
        of it is refused), or None when there is no such file. A file the agent made unreadable (chmod 000; the harness
        runs as the same uid) is None too, so submit, the finish command and the auto-judge never raise on it."""
        try:
            p = os.path.realpath(os.path.join(d, path))
            if not p.startswith(os.path.realpath(d) + os.sep) or not os.path.isfile(p):
                return None
            with open(p, "rb") as f:
                code = f.read(R.MAX_FILE + 1)
        except OSError:
            return None
        return p, (None if len(code) > R.MAX_FILE else code.decode("utf-8", errors="replace"))

    async def judge(code: str) -> tuple[dict, int, float]:
        """(result, infra retries, wall seconds) of judging code: the wait for an FCS_RM_CONCURRENCY slot and the
        retries included. The one judge time of the trace (a submission's judge_s, its turn's, fo_lg_judge_min)."""
        t0 = time.monotonic()
        res = await judge_code(label, code)
        tries = 0   # a transient fork / disk / memory failure must not cost the agent a submission
        while res["infra_error"] and tries < INFRA_RETRIES:
            tries += 1
            await asyncio.sleep(INFRA_BACKOFF * tries)
            res = await judge_code(label, code)
        return res, tries, round(time.monotonic() - t0, 3)

    async def do_submit(a, path: str) -> dict:
        got = read_code(path, a.dir)
        if got is None:
            return {"output": f"submit: no file {path} in the repository\n", "returncode": 1, "exception_info": ""}
        p, code = got
        if code is None:
            return {"output": f"submit: {path} is larger than {R.MAX_FILE} bytes\n", "returncode": 1,
                    "exception_info": ""}
        res, tries, judge_s = await judge(code)
        cases = res["cases"]
        prov, final = subset_mean(cases, prov_idx), subset_mean(cases, final_idx)
        rev = "" if TEAM else (await loop.run_in_executor(POOL, lambda: R.git(a.dir, "rev-parse", "HEAD").stdout.strip()))
        s = {"agent": a.name, "round": rounds, "step": a.steps, "file": path, "status": res["status"], "prov": prov,
             "final": final, "infra": bool(res["infra_error"]), "infra_retries": tries, "rev": rev, "sha": sha1(code),
             "code": code, "judge_s": judge_s,
             # trace only: every case (final ones too) and the judge's per-case lines; never in the agent's view
             "cases": [round(c, 4) for c in cases], "case_info": res.get("case_info", [])}
        subs.append(s)
        a.n_submit += 1
        a.dup_run = 0   # a judged submission ends a run of duplicates (MA_LG_DUP_END)
        if p == os.path.realpath(os.path.join(a.dir, "solution.cpp")):
            a.last_sha = s["sha"]
        tail = f": {res['status']}, provisional {100 * prov:.2f}"
        if TEAM:   # the episode names the commit once own_commit has run
            pending.append((len(events), s, p, tail))
        events.append((rounds, a.name, f"submitted {path}{tail}"))
        best = best_of(subs)   # an infra one only when every submission is: then there is no best yet
        best_txt = "none yet" if best["infra"] else f"{100 * best['prov']:.2f} (#{subs.index(best) + 1}, {best['agent']})"
        per = ", ".join(f"{100 * cases[i]:.0f}" for i in prov_idx if i < len(cases))
        msg = (f"Submission #{len(subs)} by {a.name}: {res['status']}. Provisional score {100 * prov:.2f} on "
               f"{len(prov_idx)} tests (per test: {per}). Best provisional so far: {best_txt}.\n")
        if res["infra_error"]:
            msg += (f"The judge could not finish this submission after {tries + 1} tries (status: {res['status']}). "
                    "It is not counted as your best. If the status is a compile resource or time failure, your code "
                    "may be too heavy to compile; otherwise submit again.\n")
        if res["status"] == "compile error" and res.get("msg"):
            msg += "Compiler output:\n" + res["msg"][:3000] + "\n"
        msg += f"About {max(0, TOKENS - a.generated):,} of your {TOKENS:,} tokens are left.\n"
        return {"output": msg, "returncode": 0, "exception_info": ""}

    def dup_submit(a, path: str):
        """MA_LG_DEDUP_SUBMIT: the reply to a submit of a file whose text equals one of a's own submissions the judge
        finished (the latest such), or else equals one apart from comments and layout (code_key), or None (judge it).
        Such a text is that submission's program, so its result would be that submission's again; the state is subs,
        which the checkpoint keeps. a.last_sha moves to this text, as a judged submit of solution.cpp would move it, so
        the finish command then acts as in the game that judges the duplicate (on this very text: a restart)."""
        got = read_code(path, a.dir)
        if got is None or got[1] is None:
            return None
        mine = [(i, x) for i, x in reversed(list(enumerate(subs, 1))) if x["agent"] == a.name and not x["infra"]]
        how = "unchanged"
        k, x = next(((i, x) for i, x in mine if x["code"] == got[1]), (0, None))
        if x is None:
            how, key = "unchanged apart from comments and whitespace", code_key(got[1])
            k, x = next(((i, x) for i, x in mine if code_key(x["code"]) == key), (0, None))
        if x is None:
            return None
        a.dup_submits += 1
        a.dup_run += 1
        if got[0] == os.path.realpath(os.path.join(a.dir, "solution.cpp")):
            a.last_sha = sha1(got[1])
        out = (f"submit: {path} is {how} since your submission #{k} (provisional {100 * x['prov']:.2f}), so it was not "
               f"judged again and no new submission was recorded. Change {path} before you submit it again.\n")
        if DUP_END and a.dup_run >= DUP_END:   # the main loop ends this agent's game after this turn
            out += (f"This is your {a.dup_run}th unchanged submission in a row, so your game ends now and this "
                    "submission scores 0.\n")
        return {"output": out, "returncode": 1, "exception_info": ""}

    async def finish(a, out: dict) -> dict:
        """The finish command (its command output `out`): submit solution.cpp; unchanged since a.last_sha, a reply
        that the work goes on."""
        got = read_code("solution.cpp", a.dir)
        if got is None:
            return {**out, "output": "Finish command received, but there is no solution.cpp to submit. Write "
                                     "solution.cpp and run the command again.\n"}
        if got[1] is not None and sha1(got[1]) == a.last_sha:
            a.idle_fin += 1
            k, x = next(((i, x) for i, x in reversed(list(enumerate(subs, 1))) if x["agent"] == a.name
                         and x["sha"] == a.last_sha), (None, None))
            if x is None and DEDUP_SUBMIT:   # a.last_sha is a layout variant's, from dup_submit
                key = code_key(got[1])
                k, x = next(((i, x) for i, x in reversed(list(enumerate(subs, 1))) if x["agent"] == a.name
                             and not x["infra"] and code_key(x["code"]) == key), (None, None))
            if x is None:
                k, x = len(subs), {"prov": 0.0}
            return {"output": f"solution.cpp is unchanged since your submission #{k} (provisional {100 * x['prov']:.2f});"
                              " nothing was submitted. " + P.KEEP_GOING.format(left=max(0, TOKENS - a.generated),
                                                                               total=TOKENS) + "\n",
                    "returncode": 0, "exception_info": "", "idle_finish": True}
        a.idle_fin = 0
        res = await do_submit(a, "solution.cpp")
        left = max(0, TOKENS - a.generated)
        return {**res, "output": "Finish command received: solution.cpp was submitted.\n" + res["output"]
                + f"Your work goes on: about {left:,} of your {TOKENS:,} tokens are left. Keep improving solution.cpp "
                "and run the command again to submit again.\n"}

    def do_post(a, text: str) -> dict:
        text = text.strip().strip("'\"")[:MAX_POST]
        board.append((rounds, a.name, text))
        a.n_post += 1
        if TEAM:
            events.append((rounds, a.name, f"posted: {text}"))
        return {"output": "Message posted to the team.\n" if TEAM else "Written to BOARD.md.\n", "returncode": 0,
                "exception_info": ""}

    def updates_for(a) -> str:
        new = [e for e in events[a.seen:] if e[1] != a.name]
        a.seen = len(events)
        if not new:
            return ""
        return "<team_updates>\n" + "".join(f"[round {r}] {n} {t}\n" for r, n, t in new) + "</team_updates>\n"

    def start_over(a) -> None:
        """A stop attempt (the turn just recorded) starts a's conversation over."""
        best = best_of(subs) if subs else None
        what = (f"The best provisional score so far is {100 * best['prov']:.2f} (submission "
                f"#{next(i for i, x in enumerate(subs, 1) if x is best)}, by {best['agent']})." if best else
                "Nothing has been submitted yet.")
        a.turns[-1]["restart"] = True
        a.restart(P.RESTART.format(best=what, left=max(0, TOKENS - a.generated), total=TOKENS))

    async def own_commit(a, notes: list, timing: dict) -> None:
        """A team agent's harness commit and publish after its command: the commit of its pending submissions named,
        whole-file adoptions counted, and what git could not do told to the agent (notes)."""
        oc = await loop.run_in_executor(POOL, R.own_commit, work, a.name, a.steps)
        sync["commit_skipped"] += bool(oc["skipped"])
        sync["git_fail"] += bool(oc["commit_fail"] or oc["publish_rc"])
        if oc["skipped"] == "no .git directory":
            notes.append("Your repository has no .git directory, so nothing is committed or published "
                         "(`git init -q` starts a new one).")
        elif oc["skipped"]:
            op = R.BUSY_OP[oc["skipped"]]
            notes.append(f"A git {op} is in progress in your repository, so this step was not committed "
                         f"or published: finish it (resolve the conflicts, `git add`, then `git {op} "
                         f"--continue`) or undo it (`git {op} --abort`).")
        if oc["commit_fail"] or oc["publish_rc"]:
            what = "; ".join(filter(None, [oc["commit_fail"], oc["publish_rc"] and f"publish exit {oc['publish_rc']}"]))
            notes.append(f"git failed after this command ({what}), so the team may not see your latest work.")
        for i, sub, p, tail in pending:   # name the commit only when it holds the submitted code
            rel = oc["head"] and await loop.run_in_executor(POOL, R.holds, a.dir, oc["head"], p, sub["code"])
            sub["rev"], sub["path"] = (oc["head"], rel) if rel else ("", "")
            where = ((f" (commit {oc['head'][:7]} on team/{a.name})" if not oc["publish_rc"] else
                      f" (commit {oc['head'][:7]}, not published)") if rel else " (not committed)")
            events[i] = (events[i][0], events[i][1], f"submitted {sub['file']}{where}{tail}")
        pending.clear()
        cur = R.sol_sha(a.dir)   # a whole-file adoption: solution.cpp now as another agent's first was
        if cur and cur != last_sol.get(a.name):
            src = origin.setdefault(cur, a.name)
            if src != a.name:
                adoptions.append((rounds, a.name, src))
        last_sol[a.name] = cur
        if notes:
            timing["git_notes"] = notes

    try:
        st = (await loop.run_in_executor(POOL, ckpt.load)) if ckpt is not None else None
        if st is not None and st.get("config") != config:
            print(f"checkpoint {ckpt.path}: other settings ({st.get('config')} vs {config}), starting fresh", flush=True)
            st = None
        if st is not None:
            try:
                await loop.run_in_executor(POOL, ckpt.restore, st, work)
            except Exception as e:
                print(f"checkpoint {ckpt.path}: tar unreadable ({e!r}), starting fresh", flush=True)
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work)
                st = None
        if st is not None:   # the end of the saved round; the work dir replaces whatever a later command half did
            rounds, run_s, subs, board, events = st["rounds"], st["run_s"], st["subs"], st["board"], st["events"]
            t_ep = time.monotonic() - st["wall_ep"]   # fo_lg_wall_min counts the earlier runs up to the saved round
            ckpt.prior_s = float(st.get("wall_s", 0.0))   # a state not resumed leaves the row's minutes this run's
            rng.setstate(st["rng"])
            if TEAM:
                sync.update(st.get("sync", {}))
                origin.update(st.get("origin", {}))
                last_sol.update(st.get("last_sol", {}))
                adoptions.extend(st.get("adoptions", []))
            for a, d in zip(agents, st["agents"]):
                for k, v in d["agent"].items():
                    setattr(a, k, v)
                for k, v in d["sess"].items():
                    setattr(a.sess, k, v)
        else:
            if TEAM:
                await loop.run_in_executor(POOL, R.own_init, work, [a.name for a in agents])
            else:
                await loop.run_in_executor(POOL, R.repo_init, work)
            await loop.run_in_executor(POOL, write_logs)
        while any(not a.done for a in agents) and rounds < MAX_ROUNDS:
            if WALL_MIN and time.monotonic() - t_ep > 60 * WALL_MIN:   # the wall cap ends the episode as its budget would
                for a in agents:
                    if not a.done:
                        a.done, a.exit = True, "WallExceeded"
                break
            rounds += 1
            active = [a for a in agents if not a.done]
            cmds = await asyncio.gather(*[a.think() for a in active])
            for a, c in zip(active, cmds):   # a run of format errors is a stop attempt
                if c is None and not a.done and a.fmt_run >= MAX_FORMAT_ERRORS:
                    start_over(a)
            todo = [(a, c) for a, c in zip(active, cmds) if c is not None]
            rng.shuffle(todo)
            n_subs = len(subs)
            for k, (a, cmd) in enumerate(todo):
                m_sub, m_post = SUBMIT_RE.fullmatch(cmd), POST_RE.fullmatch(cmd)
                # a team agent that owes a post: any command but a post with a message is not run
                refused = TEAM and a.owe_post and not (m_post and m_post.group(1).strip().strip("'\"").strip())
                fin = False    # the finish command ran (the unchanged ones in a row are counted)
                timing, t0 = {}, time.monotonic()   # wall times from here, so waits for a worker are seen
                notes = []   # a team: what the harness's git could not do, told to the agent after the output
                if TEAM and not (m_sub or m_post or refused):   # the teammates' latest commits, as of this command
                    r = await loop.run_in_executor(POOL, R.fetch_team, work, a.dir)
                    if r.returncode:
                        sync["fetch_fail"] += 1
                        notes.append(f"Fetching the team's branches into your repository failed (git exit "
                                     f"{r.returncode}), so team/* may be out of date.")
                    timing["fetch_s"], t0 = round(time.monotonic() - t0, 3), time.monotonic()
                n_sub = len(subs)
                if refused:
                    a.post_refused += 1
                    timing["post_refused"] = True
                    out, secs = {"output": P.POST_REFUSED, "returncode": 1, "exception_info": ""}, 0.0
                elif m_sub:
                    dup = dup_submit(a, m_sub.group(1) or "solution.cpp") if DEDUP_SUBMIT else None
                    if dup is not None:
                        timing["dup_submit"] = True
                        if DUP_END and a.dup_run >= DUP_END:
                            timing["dup_end"] = True
                    out, secs = dup or await do_submit(a, m_sub.group(1) or "solution.cpp"), 0.0
                elif m_post:
                    out, secs = do_post(a, m_post.group(1)), 0.0
                    a.owe_post = False
                else:
                    out = await loop.run_in_executor(POOL, C.run_command, cmd, a.dir, True)
                    secs = out["seconds"]
                    wall = time.monotonic() - t0
                    timing.update(cmd_wall_s=round(wall, 3), cmd_queue_s=round(wall - secs, 3))
                    lines = out["output"].lstrip().splitlines()
                    if lines and lines[0].strip() == MARK and out["returncode"] == 0:
                        fin = True
                        res = await finish(a, out)
                        if res.pop("idle_finish", False):   # an unchanged finish, counted in the trace
                            timing["idle_finish"] = True
                        out = {**res, "seconds": secs}
                run_s += secs
                if len(subs) > n_sub:   # submit, or a finish command that submitted: the judge time of do_submit
                    timing["judge_s"] = subs[-1]["judge_s"]
                    if TEAM:
                        a.owe_post = True
                        out = {**out, "output": out["output"] + P.POST_NEXT}
                t1 = time.monotonic()
                await loop.run_in_executor(POOL, write_logs)
                if TEAM:
                    await own_commit(a, notes, timing)
                else:
                    await loop.run_in_executor(POOL, R.commit_as, a.dir, a.name, a.steps)
                timing["harness_s"] = round(time.monotonic() - t1, 3)
                upd = updates_for(a)
                if upd:
                    out = {**out, "output": upd + out["output"]}
                if notes:
                    sep = "\n" if out["output"] and not out["output"].endswith("\n") else ""
                    out = {**out, "output": out["output"] + sep + "".join(f"[harness] {n}\n" for n in notes)}
                a.observe(cmd, {**out, "_timing": timing}, rounds, k)
                if timing.get("dup_end"):   # MA_LG_DUP_END: this turn ends the agent's game (its turn is the trace's last)
                    a.done, a.exit = True, "DupLoop"
                if timing.get("idle_finish"):   # an unchanged finish command: the agent tries to stop
                    start_over(a)
                if not fin:
                    a.idle_fin = 0
            if ckpt is not None and any(not a.done for a in agents) and rounds < MAX_ROUNDS \
                    and ckpt.due(len(subs) > n_subs):
                fut = loop.run_in_executor(POOL, ckpt.save, state(), work)
                try:
                    await asyncio.shield(fut)
                except asyncio.CancelledError:   # finish the save before the finally below removes the work dir
                    await fut
                    raise
                if ckpt.stop:
                    raise Paused(f"paused after round {rounds}")
        # what the budget cut costs (audit 2026-10-24): an agent's last solution.cpp may be better than anything it
        # submitted. It is judged once, apart from subs: S keeps the rule (only submissions count) and S_auto - S
        # measures the cut.
        auto_sub = []
        for a in agents if TEAM else agents[:1]:   # a team: each agent's solution.cpp; alone: the one repository
            got = read_code("solution.cpp", a.dir)
            # with MA_LG_DEDUP_SUBMIT a layout variant of a judged submission is that program: not judged again either
            if got is not None and got[1] and got[1].strip() and sha1(got[1]) not in {x["sha"] for x in subs + auto_sub} \
                    and not (DEDUP_SUBMIT and code_key(got[1]) in {code_key(x["code"]) for x in subs + auto_sub
                                                                    if not x["infra"]}):
                res, tries, judge_s = await judge(got[1])
                auto_sub.append({**({"owner": a.name} if TEAM else {}), "agent": "auto", "round": rounds,
                                 "file": "solution.cpp", "status": res["status"],
                                 "prov": subset_mean(res["cases"], prov_idx), "final": subset_mean(res["cases"], final_idx),
                                 "infra": bool(res["infra_error"]), "infra_retries": tries, "sha": sha1(got[1]),
                                 "judge_s": judge_s, "auto": True, "code": got[1]})
        best = best_of(subs) if subs else None
        best_auto = best_of(subs + auto_sub) if subs or auto_sub else None   # an infra auto-judge never beats a real one
        team_git = os.path.join(work, "team.git")
        if not TEAM:
            shares = (await loop.run_in_executor(POOL, R.blame_shares, work, best["file"], best["rev"])) if best else {}
            log = (await loop.run_in_executor(POOL, lambda: R.git(work, "log", "--format=%an %s").stdout))[-20000:]
        else:   # blame in team.git, which keeps every published commit, then in the agent's own repository
            shares = {}
            for d in ((team_git, dir_of[best["agent"]]) if best and best["rev"] else ()):
                shares = await loop.run_in_executor(POOL, R.blame_shares, d, best["path"], best["rev"])
                if shares:
                    break
            log = (await loop.run_in_executor(POOL, lambda: R.git(team_git, "log", "--all", "--format=%an %s")
                                              .stdout))[:20000]   # the newest commits
    finally:
        rm_work(work)
    S = best["final"] if best else 0.0
    oracle = max((s["final"] for s in subs), default=0.0)

    def total_min(key: str) -> float:   # summed over every agent's turns
        return sum(t.get(key) or 0.0 for a in agents for t in a.turns) / 60
    first = subs[0]["round"] if subs else -1
    # the info keys of the same game before the cleanup of 2026-10-28, so its rows pool with these
    info = {"lg_harness": harness_tag(), "fo_S": S, "fo_R": S, "fo_lg_n": N, "fo_lg_shared_case": float(shared),
            "fo_lg_rounds": rounds, "fo_lg_submissions": len(subs),
            "fo_lg_S_auto": best_auto["final"] if best_auto else 0.0, "fo_lg_auto_judged": float(bool(auto_sub)),
            "fo_lg_best_prov": best["prov"] if best else 0.0, "fo_lg_oracle_final": oracle,
            "fo_lg_selection_loss": oracle - S, "fo_lg_first_submit_round": first,
            "fo_lg_posts": len(board), "fo_lg_tokens": sum(a.generated for a in agents) / N,
            "fo_tokens": sum(a.generated for a in agents), "fo_latency": max(a.generated for a in agents),
            "fo_ag_steps": sum(a.steps for a in agents) / N, "fo_ag_format_errors": sum(a.fmt_total for a in agents) / N,
            "fo_ag_run_min": run_s / 60, "fo_ag_compile_fail": float(bool(best) and best["status"] == "compile error"),
            "fo_ag_no_code": float(best is None), "fo_ag_top_author": max(shares.values()) if shares else 0.0,
            **{f"fo_ag_exit_{e}": sum(a.exit == e for a in agents) / N for e in
               ("Finished", "IdleFinish", "TokensExceeded", "ContextExceeded", "RepeatedFormatError", "WallExceeded")},
            # MA_LG_DUP_END only (off: the info as before, byte for byte): 1 when an agent's game ended at a run of
            # duplicate submits (alone: at the episode's last turn)
            **({"fo_ag_exit_DupLoop": sum(a.exit == "DupLoop" for a in agents) / N,
                "fo_lg_dup_end": float(any(a.exit == "DupLoop" for a in agents))} if DUP_END else {}),
            # a submission still infra after the retries (run_eval reruns such a row)
            "fo_judge_infra_error": float(any(s["infra"] for s in subs)),
            "fo_judge_infra_unresolved": float(any(s["infra"] for s in subs)),
            "fo_judge_infra_any": float(any(s["infra"] or s["infra_retries"] for s in subs)),   # any infra judge call
            "fo_judge_infra_retries": sum(s["infra_retries"] for s in subs), "fo_lg_verdicts": 0.0,
            # where the episode's wall time went (audit 2026-10-24): model calls as the server saw them (llm) and as the
            # agent waited for them (think), commands' waits for a worker, the judge (every submission and the
            # auto-judge, retries included), the harness's logs and commits
            "fo_lg_wall_min": (time.monotonic() - t_ep) / 60, "fo_lg_llm_min": total_min("llm_s"),
            "fo_lg_think_min": total_min("think_s"), "fo_lg_cmd_queue_min": total_min("cmd_queue_s"),
            "fo_lg_judge_min": sum(s["judge_s"] for s in subs + auto_sub) / 60,
            "fo_lg_harness_min": total_min("harness_s"),
            "fo_lg_idle_finishes": sum(bool(t.get("idle_finish")) for a in agents for t in a.turns) / N,
            "fo_lg_restarts": sum(a.restarts for a in agents) / N,
            **({"fo_lg_dup_submits": sum(a.dup_submits for a in agents) / N} if DEDUP_SUBMIT else {}),
            **({"fo_lg_must_post": 1.0, "fo_lg_post_refused": sum(a.post_refused for a in agents) / N,
                "fo_lg_own_repos": 1.0, "fo_lg_adoptions": len(adoptions), "fo_lg_fetch_min": total_min("fetch_s"),
                "fo_lg_fetch_fail": sync["fetch_fail"], "fo_lg_commit_skipped": sync["commit_skipped"],
                "fo_lg_git_fail": sync["git_fail"]} if TEAM else {})}
    trace({"time": round(time.time(), 1), "eval": True, "game": "agent_long", "harness": harness_tag(),
           "label": label, "S": S, "n": N, "budget": TOKENS, "verdicts": False, "rounds": rounds,
           "shared_case": shared, "n_cases": n_cases,
           "submissions": subs, "best": {k: v for k, v in best.items() if k != "code"} if best else None,
           "auto_submission": auto_sub[0] if auto_sub else None,
           **({"repos": "own", "auto_submissions": auto_sub, "adoptions": adoptions, "sync": sync, "post": "submit"}
              if TEAM else {}),
           "board": board, "blame": shares, "git_log": log,
           # full turns (text = <think>reasoning</think>answer) and submissions with their code (audit 2026-10-24)
           "agents": [{"name": a.name, "exit": a.exit, "steps": a.steps, "generated": a.generated,
                       "dropped": a.dropped, "submits": a.n_submit, "posts": a.n_post, "restarts": a.restarts,
                       **({"post_refused": a.post_refused} if TEAM else {}),
                       **({"dup_submits": a.dup_submits} if DEDUP_SUBMIT else {}), "turns": a.turns}
                      for a in agents]})
    return agents, S, info
