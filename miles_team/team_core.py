"""Game core of labs-molt's team_math agent, copied VERBATIM from labs-molt-team
examples/python/agents/team_math.py @ d33547e (lines 59-157 and 178-216: knobs, prompt templates, delegate parsing,
reports, critical-path latency, team reward, history suffix, grading, traces), so a miles run plays the same game
with the same prompts and rewards as the molt arms. Only the grader path changed: it points at the molt worktree.
tests/test_core_parity.py checks every constant and function against the molt original.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import socket
import threading
from pathlib import Path

_MOLT_TEAM = os.environ.get("MOLT_TEAM_PATH", "/scratch/gpfs/GROUP/USER/project/labs-molt-team")
_GRADER_PATH = Path(_MOLT_TEAM) / "examples" / "python" / "utils" / "math_grader.py"
_GRADER_SPEC = importlib.util.spec_from_file_location("math_grader", _GRADER_PATH)
_GRADER = importlib.util.module_from_spec(_GRADER_SPEC)
_GRADER_SPEC.loader.exec_module(_GRADER)

MATES = int(os.environ.get("MA_TM_MATES", "3"))
LEAD_TURNS = int(os.environ.get("MA_TM_LEAD_TURNS", "4"))
MATE_TURNS = int(os.environ.get("MA_TM_MATE_TURNS", "3"))
LEAD_BUDGET = int(os.environ.get("MA_TM_LEAD_BUDGET", "5120"))
MATE_BUDGET = int(os.environ.get("MA_TM_MATE_BUDGET", "6144"))
REPORT_CHARS = int(os.environ.get("MA_TM_REPORT_CHARS", "1200"))
FORK = os.environ.get("MA_TM_FORK", "0") == "1"
COLLAB_BONUS = float(os.environ.get("MA_TM_COLLAB_BONUS", "0.1"))
LAT_COEF = float(os.environ.get("MA_TM_LAT_COEF", "0.1"))
LAT_NORM = float(os.environ.get("MA_TM_LAT_NORM", "16384"))
LAT_CAP = float(os.environ.get("MA_TM_LAT_CAP", "0.3"))
PREFILL_RATE = float(os.environ.get("MA_TM_PREFILL_RATE", "0.1"))
TRACE_DIR = os.environ.get("MA_TM_TRACE_DIR", "")
TRACE_EVERY = int(os.environ.get("MA_TM_TRACE_EVERY", "16"))
LEAD_THINK = os.environ.get("MA_TM_LEAD_THINK", "1") == "1"
MATE_THINK = os.environ.get("MA_TM_MATE_THINK", "1") == "1"
assert MATES >= 0 and LEAD_TURNS >= 1 and MATE_TURNS >= 1

LEAD_SUFFIX = (
    "\n\nYou lead a team of up to {k} teammates. Each teammate is an expert solver who sees this problem but none of "
    "your reasoning. You may delegate work or send a teammate a follow-up by writing, after your reasoning, one block "
    "per message:\n<delegate to=1>what teammate 1 should do</delegate>\n(use to=1..{k}; a new number starts a new "
    "teammate). You will get their reports back and can check them, ask follow-ups, or delegate again. Each teammate "
    "takes at most {m} messages. You have at most {t} turns of {b} tokens each. When you are confident, give the final "
    "answer in \\boxed{{}} and write no delegate block."
)
SOLO_SUFFIX = (
    "\n\nSolve the problem and put the final answer in \\boxed{{}}. You have at most {t} turns of {b} tokens each; if a "
    "turn runs out of tokens you can continue in the next one."
)
MATE_FIRST = (
    "\n\nYou are teammate {j} of a team. Your lead asks:\n{msg}\n\nDo it carefully. End with a short report for the "
    "lead, and put any final numerical answer in \\boxed{{}}."
)
MATE_FORK = "\n\nThe lead's notes so far:\n{notes}"
MATE_NEXT = "Message from your lead:\n{msg}\n\nReply with a short report; put any final answer in \\boxed{{}}."
MAILBOX = "Reports from your teammates:\n{reports}\n\n{tail}"
TAIL_MORE = "You may delegate, or give the final answer in \\boxed{{}} with no delegate block. Turns left: {left}."
TAIL_SOLO = "Continue, and give the final answer in \\boxed{{}}. Turns left: {left}."
TAIL_LAST = "This is your last turn: give the final answer in \\boxed{}."
CUT_NOTE = "(Your previous turn ran out of tokens before you finished; nothing was sent.)"
LIMIT_NOTE = "(Teammate(s) {who} already used all {m} messages; nothing was sent to them.)"
INVALID_NOTE = "(Your delegate blocks were not understood - use <delegate to=N>message</delegate> with N in 1..{k}; nothing was sent.)"
CHEAP_REPORT = 1000  # tokens: a finished teammate reply shorter than this in a bonus-paid episode counts as cheap
LEAD_RESERVE = 256  # tokens kept free below max_length when sizing a lead turn

_DELEGATE_RE = re.compile(r"<delegate\s+to\s*=\s*\"?(\d+)\"?\s*>(.*?)</delegate>", re.DOTALL | re.IGNORECASE)
_SPECIAL_RE = re.compile(r"<｜[^｜]*｜>")


def visible(text: str, thinking: bool = True) -> str:
    """The part of a reply others may see, special tokens stripped: after the closing </think>; for a thinking session
    that never closed its reasoning, nothing; a non-thinking session's reply (no think block of its own) whole."""
    if "</think>" in text:
        return _SPECIAL_RE.sub(" ", text.split("</think>")[-1]).strip()
    return "" if thinking else _SPECIAL_RE.sub(" ", text).strip()


def parse_delegations(text: str, k: int = MATES, thinking: bool = True) -> list[tuple[int, str]]:
    """``<delegate to=N>msg</delegate>`` blocks in the visible part of a finished reply, in order, one per teammate
    (the last message to a teammate wins), N in 1..k. A cut-off reply delegates nothing (the caller does not parse it)."""
    out: dict[int, str] = {}
    for n, msg in _DELEGATE_RE.findall(visible(text, thinking)):
        n = int(n)
        if 1 <= n <= k and msg.strip():
            out[n] = msg.strip()[:2000]
    return sorted(out.items())


def report_line(j: int, text: str, answer: str, cut: bool, limit: int = REPORT_CHARS, failed: bool = False,
                thinking: bool = True) -> str:
    if failed:
        return f"- Teammate {j}: (did not respond; no report)"
    if cut:
        return f"- Teammate {j}: (ran out of tokens before finishing; no report)"
    body = visible(text, thinking)[:limit] or "(no report)"
    return f"- Teammate {j} (answer: {answer or 'none'}):\n{body}"


def critical_path(rounds: list[dict], prefill_rate: float = PREFILL_RATE) -> float:
    """Derived latency of the episode in token units. ``rounds``: per lead turn, {"lead": (gen, new_prompt),
    "mates": [(gen, new_prompt), ...]} for the teammates that answered the delegations of THAT lead turn. Rounds are
    sequential; teammates within a round run in parallel, so only the slowest counts."""
    total = 0.0
    for r in rounds:
        g, p = r["lead"]
        total += g + prefill_rate * p
        if r.get("mates"):
            total += max(g2 + prefill_rate * p2 for g2, p2 in r["mates"])
    return total


def team_reward(task: float, delegated: bool, read_report: bool, latency_tokens: float) -> tuple[float, float, float]:
    """(R_lead, collab, latency_penalty) for one episode; teammates get R_lead - collab."""
    collab = COLLAB_BONUS if (delegated and read_report) else 0.0
    lat = min(LAT_CAP, LAT_COEF * latency_tokens / LAT_NORM) if LAT_NORM > 0 else 0.0
    return task + collab - lat, collab, lat



def _grade(text: str, prompt: str, label) -> tuple[float, str]:
    try:
        r = _GRADER.score_response(text, prompt, label)
        return float(r.get("reward", 0.0)), str(r.get("prediction", ""))
    except Exception:
        return 0.0, ""


def _with_suffix(messages: list, suffix: str) -> list:
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = m["content"] + suffix
            return out
    out.append({"role": "user", "content": suffix.strip()})
    return out


_TRACE_LOCK = threading.Lock()
_TRACE_SEEN = 0


def _trace_due() -> bool:
    global _TRACE_SEEN
    if not TRACE_DIR or TRACE_EVERY <= 0:
        return False
    with _TRACE_LOCK:
        _TRACE_SEEN += 1
        return (_TRACE_SEEN - 1) % TRACE_EVERY == 0


def _write_trace(record: dict) -> None:
    path = Path(TRACE_DIR) / f"traces-{socket.gethostname()}-{os.getpid()}.jsonl"
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _TRACE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

