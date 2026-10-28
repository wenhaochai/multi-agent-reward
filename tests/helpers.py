"""Shared pieces of the game tests: the environment (call env() before importing flint.game, whose settings are read at
import), a scripted model session with ChatSession's interface, a scripted agent that skips the model, and the
checks. Tests run in miles.sif (python with aiohttp and yaml), one process per test: tests/run_all.sh."""
import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"   # 10 cases
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MSG = [{"role": "user", "content": "problem"}]
# wall-clock numbers differ between any two runs
CLOCK = {"t", "think_s", "seconds", "cmd_wall_s", "cmd_queue_s", "harness_s", "judge_s", "llm_s", "t_req_start",
         "fetch_s"}
_ok = [True]


def env(**kw) -> str:
    """Commands run in the caller (no case workers), TMPDIR and the trace dir are fresh; returns the trace dir. TMPDIR
    is never under /tmp: a long-game command gets a private /tmp mounted over it, which would hide a work dir there."""
    os.environ.setdefault("FCS_CASE_PROC", "0")
    base = os.environ.get("TMPDIR", "")
    base = base if base and not os.path.realpath(base).startswith("/tmp") else "/scratch/gpfs/GROUP/USER/tmp"
    os.environ["TMPDIR"] = tempfile.mkdtemp(dir=base)
    os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
    os.environ.update({k: str(v) for k, v in kw.items()})
    sys.path.insert(0, ROOT)
    return os.environ["MA_AG_TRACE_DIR"]


def check(name, cond, detail=""):
    _ok[0] &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


def done():
    print("ALL OK" if _ok[0] else "SOME FAILED")
    sys.exit(0 if _ok[0] else 1)


def inp(index=0, episode=0):
    return SimpleNamespace(sample=SimpleNamespace(index=index, group_index=episode), evaluation=True)


def traces() -> list:
    return [json.loads(x) for f in sorted(glob.glob(os.path.join(os.environ["MA_AG_TRACE_DIR"], "*.jsonl")))
            for x in open(f)]


class FakeSession:
    """ChatSession's interface without a server: reply(session, messages) -> (text, cut); each model call costs
    `tokens` completion tokens."""

    def __init__(self, input, name, reply, tokens=10):
        self.input, self.name, self.reply, self.tokens = input, name, reply, tokens
        self.used, self.queue, self.cont, self.n_cont, self.last_reasoning, self.fresh = 0, [], False, 0, "", False
        self.last_call, self.last_tc, self.completion = {}, None, None

    def forget(self):
        self.queue, self.cont, self.n_cont, self.used, self.fresh = [], False, 0, 0, False

    async def turn(self, messages, max_tokens):
        self.last_call, self.last_tc = {}, None
        text, cut = await self.reply(self, messages)
        self.completion = self.tokens
        self.used += 7
        return text, cut


def sessions(reply, tokens=10):
    return lambda input, name: FakeSession(input, name, reply, tokens)


def block(cmd: str, k=0) -> str:
    return f"<think>r{k}</think>step {k}\n\n```mswea_bash_command\n{cmd}\n```"


def scripted(G):
    """An agent that runs SCRIPT[name] (or Scripted.script, shared) and skips the model; outs keeps each output."""

    class Scripted(G.Agent):
        script = []
        scripts = {}

        def __init__(self, sess, j, problem):
            super().__init__(sess, j, problem)
            self.chat = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "INSTANCE"}]
            self.outs, self.chats = [], []

        async def _think(self):
            sc = Scripted.scripts.get(self.name, Scripted.script)
            if not sc:
                self.done, self.exit = True, "TokensExceeded"
                return None
            self.chats.append(list(self.chat))
            self.steps += 1
            self.generated += 1000
            cmd = sc.pop(0)
            self._pending = (cmd, {"step": self.steps, "text": cmd, "cut": False})
            return cmd

        def observe(self, cmd, out, rnd, order):
            self.outs.append(out["output"])
            super().observe(cmd, out, rnd, order)
    return Scripted


def no_session(input, name):
    return FakeSession(input, name, None)
