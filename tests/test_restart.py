"""Only the budget ends an agent's work (owner 2026-10-25), and a stop attempt starts its conversation over (smoke w251,
2026-10-26: after a reply alone, no Flint changed solution.cpp again). The finish command on an unchanged solution.cpp
is not judged: its reply says the work goes on, and the next turn sees only the system message and the instance message
with a note (the best provisional score, the tokens left); its files stay, an earlier note and the trim note go, the
session forgets the old conversation's calls, and every restart is counted. Three format errors in a row are a stop
attempt too. Scripted agent (and the real think with a fake session for the format errors), stubbed judge."""
import asyncio

from helpers import FIN, LABEL, MSG, check, done, env, inp, scripted, sessions

env(MA_LG_N=1, MA_LG_TOKENS=100000)
from flint import game as G  # noqa: E402
from flint import prompts as P  # noqa: E402

judged = []


async def fake_judge(label, code):
    judged.append(code)
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


Scripted = scripted(G)
Real = G.Agent


def stuck_session(input, name):   # a session in the middle of a response with more calls
    s = sessions(None)(input, name)
    s.queue, s.cont, s.n_cont, s.used, s.fresh = [("ls", {})], True, 2, 50000, True
    return s


def run(script, new_session=stuck_session):
    Scripted.script = list(script)
    judged.clear()
    agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, new_session))
    return agents[0], info


G.judge_code, G.Agent = fake_judge, Scripted
a, info = run(["printf 'int main(){}\\n' > solution.cpp", FIN, FIN, "true", FIN, "true"])
check("only the changed file is judged; the script runs out (TokensExceeded)", len(judged) == 1
      and a.exit == "TokensExceeded" and a.steps == 6, (judged, a.exit, a.steps))
check("an unchanged finish: nothing submitted, the work goes on, with next steps",
      "nothing was submitted" in a.outs[2] and "Your work goes on until your token budget is used up" in a.outs[2]
      and "brute-force" in a.outs[2] and "of your 100,000 tokens are left" in a.outs[2], a.outs[2])
check("the finish reply on a changed file names no way to end", "ends your work" not in a.outs[1]
      and "run the command again to submit again" in a.outs[1], a.outs[1])
check("each unchanged finish starts the conversation over, flagged on its turn and counted", a.restarts == 2
      and [bool(t.get("restart")) for t in a.turns] == [False, False, True, False, True, False]
      and info["fo_lg_restarts"] == 2.0 and info["fo_lg_idle_finishes"] == 2.0
      and info["fo_ag_exit_Finished"] == 0.0, (a.restarts, [t.get("restart") for t in a.turns], info))
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
a, info = run(["printf 'int main(){}\\n' > solution.cpp", "submit", "printf 'int main(){return 0;}\\n' > solution.cpp",
               FIN, FIN])
check("the submit reply says how many tokens are left", "of your 100,000 tokens are left" in a.outs[1], a.outs[1])
check("a changed file is submitted, the next unchanged finish restarts", len(judged) == 2 and a.restarts == 1
      and a.exit == "TokensExceeded", (judged, a.restarts, a.exit))

ag = Scripted(None, 0, "problem")
note = P.RESTART.format(best="B.", left=1, total=2)
ag.chat = ([{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "INSTANCE" + note}]
           + [{"role": "assistant", "content": "x" * 100}, {"role": "user", "content": "y" * 100}] * 4)
ag.sess = sessions(None)(None, "agent1")
chat, d = G.trim_chat(ag.chat, 500, 0, 300)
check("a trim after a restart keeps the note and adds its own", chat[1]["content"].startswith("INSTANCE" + note)
      and "[Note: your 3 earliest steps were removed" in chat[1]["content"] and d == 3, (chat[1], d))
ag.chat, ag.dropped = chat, d
ag.restart(P.RESTART.format(best="C.", left=1, total=2))
check("a restart drops the trim note and the old note, and dropped counts from 0", ag.chat == [
    {"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "INSTANCE" + P.RESTART.format(
        best="C.", left=1, total=2)}] and ag.dropped == 0 and ag.restarts == 1, (ag.chat, ag.dropped))

# the real think: three format errors in a row start the conversation over before the next model call
texts = ["<think>r</think>no command here"] * 3 + ["<think>r</think>```mswea_bash_command\ntrue\n```"]
sizes = []


async def reply(sess, messages):
    sizes.append(len(messages))
    return (texts.pop(0) if texts else "<think>r</think>no command here"), False

G.Agent = Real
agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, sessions(reply, tokens=25000)))
a = agents[0]
check("three format errors in a row start the conversation over before the next call", a.restarts == 1
      and sizes == [2, 4, 6, 2] and a.turns[2].get("restart") and a.turns[2].get("format_error")
      and not a.turns[1].get("restart") and a.fmt_run == 0 and a.exit == "TokensExceeded",
      (a.restarts, sizes, [(t.get("restart"), t.get("format_error")) for t in a.turns], a.exit))
check("the note says nothing was submitted, and the tokens left", "Nothing has been submitted yet." in a.chat[1]["content"]
      and "about 25,000 of your 100,000 tokens are left" in a.chat[1]["content"]
      and a.chat[1]["content"].count("[Continued: ") == 1, a.chat[1]["content"][-900:])


async def no_cmd(sess, messages):
    return "<think>r</think>no command here", False

ag = Real(sessions(no_cmd)(None, "agent1"), 0, "problem")
asyncio.run(ag.think())
check("the format error ends with: submit alone", ag.chat[-1]["content"].endswith("To submit, run `submit` alone.")
      and "not be able to continue" not in ag.chat[-1]["content"], ag.chat[-1]["content"][-300:])
done()
