"""fcs_agent_long with MA_LG_END=budget and MA_LG_ON_STOP=fresh (smoke w251, 2026-10-26: after the reply alone, no
Flint changed solution.cpp again). A stop attempt (the finish command on an unchanged solution.cpp, or a run of format
errors) starts the agent's conversation over: the system message and the instance message with a note that the work
goes on (the best provisional score, the tokens left). Its files stay, an earlier note and the trim note go, the session
forgets the old conversation's calls, and every restart is counted. Scripted agent (and the real think with a stub
session for the format errors), stubbed judge."""
import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_END"] = "budget"
os.environ["MA_LG_ON_STOP"] = "fresh"
os.environ["MA_LG_N"] = "1"
os.environ["MA_LG_TOKENS"] = "100000"
os.environ["TMPDIR"] = tempfile.mkdtemp()
os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
INP = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
judged = []
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


Real = L._Agent


class Scripted(Real):
    script = []

    def __init__(self, input, j, problem, shared=False):
        self.name = f"agent{j + 1}"
        self.steps = self.fmt_run = self.fmt_total = self.timeouts = self.cut_n = 0
        self.done, self.exit, self.turns, self.segs = False, "", [], []
        self.generated, self.dropped, self.seen, self.last_seg = 0, 0, 0, None
        self.n_submit = self.n_post = self.idle_fin = 0
        self.last_sha = None
        self.chat = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "INSTANCE"}]
        # a chat session in the middle of a response with more calls (chat_eval_long.ChatSession's fields)
        self.sess = SimpleNamespace(queue=[("ls", {})], cont=True, n_cont=2, used=50000, fresh=True)
        self.chats = []   # the chat each turn was given

    async def think(self):
        if not Scripted.script:
            self.done, self.exit = True, "TokensExceeded"
            return None
        self.chats.append(list(self.chat))
        self.steps += 1
        self.generated += 1000
        cmd = Scripted.script.pop(0)
        self._pending = (cmd, {"step": self.steps, "text": cmd, "cut": False})
        return cmd


def run(script):
    Scripted.script = list(script)
    judged.clear()
    agents, S, info = asyncio.run(L.episode(INP, [{"role": "user", "content": "problem"}], LABEL))
    return agents[0], info


L.judge_code, L._Agent = fake_judge, Scripted
a, info = run(["printf 'int main(){}\\n' > solution.cpp", FIN, FIN, "true", FIN, "true"])
check("only the changed file is judged; the script runs out (TokensExceeded)", len(judged) == 1
      and a.exit == "TokensExceeded" and a.steps == 6, (judged, a.exit, a.steps))
check("each unchanged finish starts the conversation over, flagged on its turn and counted", a.restarts == 2
      and [bool(t.get("restart")) for t in a.turns] == [False, False, True, False, True, False]
      and info["fo_lg_restarts"] == 2.0 and info["fo_lg_idle_finishes"] == 2.0,
      (a.restarts, [t.get("restart") for t in a.turns], info.get("fo_lg_restarts")))
c4, c6 = a.chats[3], a.chats[5]
check("the next turn sees only the system message and the instance message with the note", len(c4) == 2
      and c4[0]["content"] == "SYSTEM" and c4[1]["content"].startswith("INSTANCE\n\n[Continued: your earlier "
                                                                       "conversation on this problem has ended")
      and "The best provisional score so far is 50.00 (submission #1, by agent1)." in c4[1]["content"]
      and "about 97,000 of your 100,000 tokens are left" in c4[1]["content"], c4)
check("a second restart replaces the note, with the tokens left then", len(c6) == 2
      and c6[1]["content"].count("[Continued: ") == 1 and "about 95,000 of your 100,000 tokens" in c6[1]["content"], c6)
check("the session forgets the old conversation's calls", a.sess.queue == [] and not a.sess.cont
      and a.sess.n_cont == 0 and a.sess.used == 0 and not a.sess.fresh, vars(a.sess))
check("the harness tag names the protocol", info["lg_harness"] == "long-v1-budget-fresh-ctxlow1", info["lg_harness"])

ag = Scripted(INP, 0, "problem")
note = L._RESTART.format(best="B.", left=1, total=2)
ag.chat = ([{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "INSTANCE" + note}]
           + [{"role": "assistant", "content": "x" * 100}, {"role": "user", "content": "y" * 100}] * 4)
chat, d = L.trim_chat(ag.chat, 500, 0, 300)
check("a trim after a restart keeps the note and adds its own", chat[1]["content"].startswith("INSTANCE" + note)
      and "[Note: your 3 earliest steps were removed" in chat[1]["content"] and d == 3, (chat[1], d))
ag.chat, ag.dropped = chat, d
ag.restart(L._RESTART.format(best="C.", left=1, total=2))
check("a restart drops the trim note and the old note, and dropped counts from 0", ag.chat == [
    {"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "INSTANCE" + L._RESTART.format(
        best="C.", left=1, total=2)}] and ag.dropped == 0 and ag.restarts == 1, (ag.chat, ag.dropped))


class Stub:
    texts = []

    def __init__(self, input, name, think, max_len=None):
        self.input, self.segments, self.last_messages = input, [], None
        self.queue, self.cont, self.n_cont, self.used, self.fresh = [], False, 0, 0, False
        self.sizes = []   # the number of messages each call was given

    async def turn(self, messages, max_tokens):
        self.sizes.append(len(messages))
        self.segments.append(SimpleNamespace(loss_mask=[1] * 25000))
        return (Stub.texts.pop(0) if Stub.texts else "<think>r</think>no command here"), False


L.Session, L._Agent = Stub, Real
Stub.texts = ["<think>r</think>no command here"] * 3 + ["<think>r</think>```mswea_bash_command\ntrue\n```"]
agents, S, info = asyncio.run(L.episode(INP, [{"role": "user", "content": "problem"}], LABEL))
a = agents[0]
check("three format errors in a row start the conversation over before the next call", a.restarts == 1
      and a.sess.sizes == [2, 4, 6, 2] and a.turns[2].get("restart") and a.turns[2].get("format_error")
      and not a.turns[1].get("restart") and a.fmt_run == 0 and a.exit == "TokensExceeded",
      (a.restarts, a.sess.sizes, [(t.get("restart"), t.get("format_error")) for t in a.turns], a.exit))
check("the note says nothing was submitted, and the tokens left", "Nothing has been submitted yet." in a.chat[1]["content"]
      and "about 25,000 of your 100,000 tokens are left" in a.chat[1]["content"]
      and a.chat[1]["content"].count("[Continued: ") == 1, a.chat[1]["content"][-900:])
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
