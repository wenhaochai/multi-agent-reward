"""A team without a lead (owner 2026-10-13): N mini-swe-agents (miles_team.fcs_agent) work at the same time in one
shared git repo and send no messages; they see only each other's files, commits and command output.

Rounds: in each round every active agent takes one step (its model turns run in parallel), then the commands run one
after another in a random order (seeded by the problem and the episode), so a command sees everything the earlier
commands of the round wrote. After each command the harness commits the repo's source files under the acting agent's
git name, so git blame of the final program says who wrote each line. An agent's submit command
(`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) ends only its own work; when every agent has submitted or hit a limit
(steps, tokens, context, three format errors in a row), the hidden tests judge the repo's solution.cpp once, if at
least one agent submitted (otherwise the team scores 0, as a lone mini-swe-agent without a submission does).

Rewards (MA_AG_REWARD, training): shared (every agent's steps get the team reward R = S - MA_AG_MU x c, where c is the
minutes all commands ran; owner 2026-10-13) or blame (agent j gets R x its share of the final program's lines; the owner
withdrew blame credit as a design on 2026-10-13, so it stays off). Env: MA_AG_N (4) and fcs_agent's MA_AG_* limits, per agent.
--custom-generate-function-path miles_team.fcs_agent_team.generate
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

from . import fcs_agent as A
from .fcs_rm import answer, judge_code
from .team_rollout import Aborted, Session

N = int(os.environ.get("MA_AG_N", "4"))
REWARD = os.environ.get("MA_AG_REWARD", "shared")
assert REWARD in ("shared", "blame"), REWARD
# what the hidden tests judge (user 2026-10-20 popup, after probe 15881283: in 172 of 179 episodes an agent rewrote
# solution.cpp after another had submitted, and agents that never submitted left unfinished programs that were judged):
#   repo         solution.cpp at the end of the episode (the first probe)
#   last_submit  solution.cpp as it was when the last agent to submit issued its submit command
SCORE = os.environ.get("MA_AG_SCORE", "repo")
assert SCORE in ("repo", "last_submit"), SCORE
SOURCES = (".cpp", ".cc", ".h", ".hpp", ".py", ".txt", ".md", ".sh", ".in", ".out", ".ans")  # binaries stay out

# fcs_agent's instance template with the team's situation and submit rule in place of the lone agent's
_LONE_ENV = ("You work in an empty directory with a Linux shell (g++, python3 and the usual command-line tools; no "
             "internet).")
_TEAM_ENV = ("You are {name}, one of {n} agents working at the same time in this shared git repository, with a Linux "
             "shell (g++, python3, git and the usual command-line tools; no internet). There is no leader and you "
             "cannot send messages: you see the other agents only through the files and commits in the repository. "
             "The agents act in rounds: in each round every agent runs one command, in a random order. After every "
             "command, the repository's source files are committed under the name of the agent that ran it. The "
             "team's submission is the file solution.cpp at the repository root.")
_LONE_SUBMIT = ("6. Submit your solution and finish your work by issuing the following command: "
                "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat solution.cpp`.\n   The output of that command after "
                "its first line is your submission.")
_TEAM_SUBMIT = ("6. When you are done, finish your work by issuing the following command: "
                "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n   This ends only your own work. When every agent has "
                "finished, solution.cpp at the repository root is judged as the team's submission.")
assert _LONE_ENV in A.INSTANCE and _LONE_SUBMIT in A.INSTANCE.replace("\\\n", "")
TEAM_INSTANCE = A.INSTANCE.replace(_LONE_ENV, _TEAM_ENV).replace(_LONE_SUBMIT, _TEAM_SUBMIT)
assert _TEAM_ENV in TEAM_INSTANCE and _TEAM_SUBMIT in TEAM_INSTANCE


def team_prompt(problem: str, name: str) -> str:
    """The instance message of agent `name`: {name} and {n} are format fields of TEAM_INSTANCE like {problem} (a
    .replace after .format never ran: .format raised KeyError 'name', job 15877957)."""
    steps = f" You have at most {A.STEPS} steps." if A.STEPS > 0 else ""   # as fcs_agent: 0 = no step limit, none named
    return TEAM_INSTANCE.format(problem=problem, timeout=A.TIMEOUT, steps=steps, name=name, n=N)


# the harness's git runs outside the sandbox (review 2026-10-20): no global config, and no hooks or fsmonitor command that
# an agent could have written into the repo's .git
_GIT_SAFE = ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
MAX_COMMIT_BYTES = 1 << 20   # files the harness commits; a large .in/.txt stays uncommitted (git timeout, review 2026-10-20)


def _git(work: str, *args, env: dict | None = None) -> subprocess.CompletedProcess:
    e = {"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": work, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
         **(env or {})}
    try:
        return subprocess.run(["git", *_GIT_SAFE, *args], cwd=work, env=e, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError) as ex:   # an aborted episode is resubmitted; a crash would end the run
        raise Aborted(f"harness git {args[:1]}: {type(ex).__name__}") from ex


def repo_init(work: str) -> None:
    _git(work, "init", "-q", "-b", "main")
    who = {"GIT_AUTHOR_NAME": "harness", "GIT_AUTHOR_EMAIL": "harness@team", "GIT_COMMITTER_NAME": "harness",
           "GIT_COMMITTER_EMAIL": "harness@team"}
    _git(work, "commit", "-q", "--allow-empty", "-m", "start", env=who)


def commit_as(work: str, name: str, step: int) -> None:
    """The harness's commit after a command: the repo's source files, under the acting agent's name (trusted code,
    outside the sandbox; the agents' own git use stays possible)."""
    who = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": f"{name}@team", "GIT_COMMITTER_NAME": name,
           "GIT_COMMITTER_EMAIL": f"{name}@team"}
    changed = _git(work, "ls-files", "-z", "--others", "--modified", "--deleted", "--exclude-standard").stdout
    files = sorted({f for f in changed.split("\0") if f.endswith(SOURCES)
                    and not (os.path.isfile(os.path.join(work, f)) and os.path.getsize(os.path.join(work, f)) > MAX_COMMIT_BYTES)})
    if files:  # (a pathspec that matches nothing would fail the whole add)
        _git(work, "add", "-A", "--", *files, env=who)
        _git(work, "commit", "-q", "-m", f"{name} step {step}", env=who)


def snapshot(work: str, path: str = "solution.cpp") -> tuple:
    """(the file's text or "", the commit it is in) right after the harness commit that follows a command."""
    p = os.path.join(work, path)
    text = open(p, errors="replace").read() if os.path.isfile(p) else ""
    return text, _git(work, "rev-parse", "HEAD").stdout.strip()


def blame_shares(work: str, path: str = "solution.cpp", rev: str = "HEAD") -> dict:
    """{author: share of the file's non-blank lines}, from git blame of the committed file."""
    r = _git(work, "blame", "--line-porcelain", rev, "--", path)
    if r.returncode:
        return {}
    counts, author, nonblank = {}, None, True
    for line in r.stdout.splitlines():
        if line.startswith("author "):
            author = line[7:]
        elif line.startswith("\t"):
            if line[1:].strip():
                counts[author] = counts.get(author, 0) + 1
    total = sum(counts.values())
    return {a: c / total for a, c in counts.items()} if total else {}


class _Agent:
    def __init__(self, input, j: int, problem: str):
        self.name = f"agent{j + 1}"
        self.sess = Session(input, self.name, A.THINK, max_len=A.MAX_LEN)
        self.chat = [{"role": "system", "content": A.SYSTEM},
                     {"role": "user", "content": team_prompt(problem, self.name)}]
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns, self.segs = False, "", [], []

    def gen(self) -> int:
        return A._gen(self.sess)

    async def think(self):
        """One model turn: the command to run, or None (a format error, a limit, or a full context)."""
        if A.STEPS > 0 and self.steps >= A.STEPS:   # 0: no step limit (fcs_agent, user 2026-10-18)
            self.done, self.exit = True, "LimitsExceeded"; return None
        if self.gen() >= A.TOKENS:
            self.done, self.exit = True, "TokensExceeded"; return None
        self.steps += 1
        self.sess.last_messages = None
        n_seg = len(self.sess.segments)
        text, cut = await self.sess.turn(self.chat, min(A.TURN, A.TOKENS - self.gen()))
        for tok in A._SPECIAL:
            text = text.replace(tok, "")
        if len(self.sess.segments) > n_seg:
            self.segs.append(len(self.sess.segments) - 1)
        if not text and cut:
            self.done, self.exit = True, "ContextExceeded"
            self.turns.append({"step": self.steps, "text": "", "cut": True}); return None
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
        vis, rec = self._pending
        self.timeouts += "timed out" in out["exception_info"]
        self.turns.append({**rec, "round": rnd, "order": order, "command": cmd, "returncode": out["returncode"],
                           "output": out["output"][:20000], "exception": out["exception_info"]})
        lines = out["output"].lstrip().splitlines()
        if lines and lines[0].strip() == A.MARK and out["returncode"] == 0:
            self.done, self.exit = True, "Submitted"
            return
        self.chat = self.chat + [{"role": "assistant", "content": vis}, {"role": "user", "content": A.observation(out)}]


async def episode(input: GenerateFnInput, messages: list, label: str):
    problem = A.problem_text(messages)
    agents = [_Agent(input, j, problem) for j in range(N)]
    work = os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent_team", uuid.uuid4().hex)
    os.makedirs(work)
    loop = asyncio.get_running_loop()
    rng = random.Random(f"{label}:{input.sample.index}:{getattr(input.sample, 'group_index', 0)}")
    rounds, order_log, run_s, last_snap = 0, [], 0.0, None
    try:
        await loop.run_in_executor(A._POOL, repo_init, work)
        while any(not a.done for a in agents):
            rounds += 1
            active = [a for a in agents if not a.done]
            cmds = await asyncio.gather(*[a.think() for a in active])
            todo = [(a, c) for a, c in zip(active, cmds) if c is not None]
            rng.shuffle(todo)
            order_log.append([a.name for a, _ in todo])
            for k, (a, cmd) in enumerate(todo):
                out = await loop.run_in_executor(A._POOL, A.run_command, cmd, work)
                run_s += out["seconds"]
                await loop.run_in_executor(A._POOL, commit_as, work, a.name, a.steps)
                a.observe(cmd, out, rounds, k)
                if a.exit == "Submitted":   # the submitter's version, after its own command's commit
                    last_snap = (a.name, *await loop.run_in_executor(A._POOL, snapshot, work))
        final, rev = None, "HEAD"
        if any(a.exit == "Submitted" for a in agents):
            if SCORE == "last_submit":
                _, final, rev = last_snap
            else:
                final = (await loop.run_in_executor(A._POOL, snapshot, work))[0]
        shares = await loop.run_in_executor(A._POOL, blame_shares, work, "solution.cpp", rev)
        log = (await loop.run_in_executor(A._POOL, lambda: _git(work, "log", "--format=%an %s").stdout))[-20000:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res = await judge_code(label, final) if final and final.strip() else None
    S = res["score"] if res else 0.0
    R = S - A.MU * run_s / 60   # the team reward: the score minus the cost of every agent's command run time
    gens = [a.gen() for a in agents]
    top = max(shares.values()) if shares else 0.0
    info = {"fo_S": S, "fo_R": R, "fo_ag_run_min": run_s / 60, "fo_ag_n": N, "fo_ag_rounds": rounds,
            "fo_ag_submitted": sum(a.exit == "Submitted" for a in agents) / N,
            "fo_ag_steps": sum(a.steps for a in agents) / N, "fo_ag_format_errors": sum(a.fmt_total for a in agents) / N,
            "fo_ag_cut": sum(a.cut_n for a in agents) / N, "fo_ag_timeouts": sum(a.timeouts for a in agents) / N,
            "fo_ag_compile_fail": float(res is not None and res["status"] == "compile error"),
            "fo_ag_no_code": float(res is None), "fo_ag_top_author": top, "fo_ag_authors": len(shares),
            "fo_tokens": sum(gens), "fo_latency": max(gens),
            **{f"fo_ag_exit_{e}": sum(a.exit == e for a in agents) / N for e in   # shares of agents, as fcs_agent's keys
               ("Submitted", "LimitsExceeded", "TokensExceeded", "ContextExceeded", "RepeatedFormatError")},
            "fo_judge_infra_error": float(bool(res and res["infra_error"]))}
    A._trace({"time": round(time.time(), 1), "eval": input.evaluation, "game": "agent_team", "label": label, "S": S, "R": R, "run_min": run_s / 60,
              "status": res["status"] if res else "no submission", "rounds": rounds, "order": order_log,
              "blame": shares, "git_log": log, "final": final, "score_rule": SCORE,
              "final_by": last_snap[0] if (SCORE == "last_submit" and last_snap) else None,
              "agents": [{"name": a.name, "exit": a.exit, "steps": a.steps, "turns": a.turns} for a in agents]})
    return agents, R, shares, info


def _reward(S: float, shares: dict, name: str) -> float:
    return S if REWARD == "shared" else S * shares.get(name, 0.0)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """As fcs_agent.generate (review 2026-10-16 there; ported for the swarm RL run, 2026-10-20): an engine abort, an empty
    episode or (in training) a judge infrastructure error returns an ABORTED sample, which miles drops and resubmits,
    instead of ending the run."""
    try:
        return await _generate(input)
    except Aborted:
        s = deepcopy(input.sample)
        s.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=s)


async def _generate(input: GenerateFnInput) -> GenerateFnOutput:
    meta = input.sample.metadata or {}
    messages = meta.get("messages")
    assert messages, "fcs_agent_team needs the raw chat in sample.metadata['messages']"
    agents, R, shares, info = await episode(input, messages, input.sample.label)   # R = S - MU x run minutes
    segs = [(a, i) for a in agents for i in a.segs]
    if not segs:
        raise Aborted("empty episode")
    if info.get("fo_judge_infra_error") and not input.evaluation:
        raise Aborted("judge infrastructure error")   # its 0 is not the policy's (fcs_agent and fcs_oracle do the same)
    if input.evaluation:
        a, i = segs[-1]
        out = a.sess.segments[i]
        out.reward = float(R)
        out.metadata = {**(out.metadata or {}), **info}
        return GenerateFnOutput(samples=out)
    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    g = input.sample.group_index if input.sample.group_index is not None else input.sample.index
    samples = []
    for a, i in segs:
        s = a.sess.segments[i]
        s.reward, s.rollout_id, s.group_index = float(_reward(R, shares, a.name)), rollout_id, g
        samples.append(s)
    samples[0].metadata = {**samples[0].metadata, **info}
    return GenerateFnOutput(samples=samples)


log_rollout = A.log_rollout
log_eval = A.log_eval
