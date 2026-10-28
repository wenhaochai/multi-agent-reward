"""A team's post rule (owner 2026-10-26 popup; four team smokes, 32 agents, had no post while posting was free): after
each of its submissions (submit, or a finish command that submitted) an agent's next command must be `post <message>`
with a message; any other command, an empty post and a finish command included, is not run, and the reply asks for the
post. The rule is in the prompt and after each submission's output; refusals are counted, flagged on their turns and
kept across a checkpoint; the harness tag and the trace name the rule. Fake session (replies keyed on the agent's turn
count; 1000 tokens a turn and a 12,000-token budget, so each agent makes 12 calls), stubbed judge, real problem 0."""
import asyncio
import os
import tempfile

from helpers import FIN, LABEL, MSG, block, check, done, env, inp, sessions, traces

env(MA_LG_N=2, MA_LG_TOKENS=12000)
from flint import game as G  # noqa: E402
from flint import prompts as P  # noqa: E402

SCRIPT = {
    "agent1": ["printf 'int main(){return 0;}\\n' > solution.cpp", "submit", "touch marker1", "post ''", 'post " "',
               "post a stub, provisional 14; next a real algorithm", "ls marker1 2>&1; echo rc=$?",
               "printf 'int main(){int y;return 0;}\\n' > solution.cpp", FIN, FIN, "post second version", FIN],
    "agent2": ["ls", "echo hi > b.txt", "cat BOARD.md", "submit b.txt", "cat SCORES.md", "post agent2 tried b.txt"],
}
hook = {"stop_at": None, "ckpt": None}


async def reply(sess, messages):
    k = sum(m["role"] == "assistant" for m in messages)   # this agent's turns so far: the same after a resume
    if hook["stop_at"] == (sess.name, k):
        hook["ckpt"].stop = True
    s = SCRIPT[sess.name]
    return block(s[k] if k < len(s) else "true", k), False


async def fake_judge(label, code):
    return {"cases": [(len(code) % 7) / 7] * 10, "status": "ok", "infra_error": "", "msg": ""}


G.judge_code = fake_judge
NEW = sessions(reply, tokens=1000)


def turn_of(a, cmd):
    return next(t for t in a.turns if t.get("command") == cmd)


def broken(rec) -> int:
    """The smoke gate's check (jobs/smoke.sh): a command that ran while a post was owed, by the harness's rule."""
    bad = 0
    for a in rec["agents"]:
        owe = False
        for t in a["turns"]:
            c = t.get("command")
            if c is None or t.get("post_refused"):
                continue
            m = G.POST_RE.fullmatch(c)
            bad += owe and not (m and m.group(1).strip().strip("'\"").strip())
            owe = "judge_s" in t
    return bad


agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, NEW))
a1, a2 = agents
rec = traces()[-1]
check("the prompt states the rule", P.TEAM_POST in a1.chat[1]["content"] and P.TEAM_POST in a2.chat[1]["content"])
t = turn_of(a1, "submit")
check("a submission's output ends by asking for the post", t["output"].endswith(P.POST_NEXT), t["output"][-300:])
t = turn_of(a1, "touch marker1")
check("a command while a post is owed is not run: flagged, exit 1, the reply asks for the post",
      t.get("post_refused") and t["returncode"] == 1 and t["output"].endswith(P.POST_REFUSED), t)
t, t2 = turn_of(a1, "post ''"), turn_of(a1, 'post " "')
check("an empty post, or one of quoted spaces, is refused too, and not posted", t.get("post_refused")
      and t2.get("post_refused") and all(m.strip() for _, n, m in rec["board"] if n == "agent1"), (t, t2, rec["board"]))
t = turn_of(a1, "ls marker1 2>&1; echo rc=$?")
check("after the post, commands run again; the refused one never ran", not t.get("post_refused")
      and "No such file" in t["output"] and "rc=2" in t["output"], t["output"])
fins = [t for t in a1.turns if t.get("command") == FIN]
check("a finish command that submitted makes the next one wait for a post; an unchanged one restarts",
      len(fins) == 3 and "judge_s" in fins[0] and fins[0]["output"].endswith(P.POST_NEXT)
      and fins[1].get("post_refused") and not fins[2].get("post_refused") and fins[2].get("idle_finish")
      and fins[2].get("restart") and a1.exit == "TokensExceeded",
      [(t.get("judge_s"), t.get("post_refused"), t["output"][:80]) for t in fins] + [a1.exit])
check("refusals are counted per agent", a1.post_refused == 4 and a2.post_refused == 1, (a1.post_refused, a2.post_refused))
check("the board holds the posts in order", [(n, m) for _, n, m in rec["board"]] ==
      [("agent1", "a stub, provisional 14; next a real algorithm"), ("agent2", "agent2 tried b.txt"),
       ("agent1", "second version")], rec["board"])
check("teammates see the posts at the top of their next observation", any(
      "<team_updates>" in t.get("output", "") and "posted: agent2 tried b.txt" in t["output"] for t in a1.turns))
check("every agent used its budget; no command ran while a post was owed (the smoke gate's check)",
      a2.exit == "TokensExceeded" and a1.generated == a2.generated == 12000 and broken(rec) == 0
      and a1.owe_post is False, (a2.exit, broken(rec), a1.owe_post))
check("info, the harness tag and the trace name the rule", info["fo_lg_must_post"] == 1.0
      and info["fo_lg_post_refused"] == 2.5 and info["lg_harness"] == "long-v1-budget-fresh-ctxlow0.75-own-mustpost"
      and rec["post"] == "submit" and [x["post_refused"] for x in rec["agents"]] == [4, 1],
      (info.get("fo_lg_post_refused"), info["lg_harness"], rec.get("post")))

# a checkpoint taken right after the submission keeps the owed post: the resumed run refuses the next command and ends
# exactly as the uninterrupted run
ref = (len(rec["submissions"]), [(n, m) for _, n, m in rec["board"]], [x["post_refused"] for x in rec["agents"]],
       rec["rounds"], rec["S"])
ck = G.Checkpoint(os.path.join(tempfile.mkdtemp(), "ep"), every=1e9)
hook.update(stop_at=("agent1", 1), ckpt=ck)
try:
    asyncio.run(G.episode(inp(), MSG, LABEL, NEW, ckpt=ck))
    paused = False
except G.Paused:
    paused = True
st = ck.load()
check("paused right after the submission, with the post owed in the saved state", paused and st and
      st["agents"][0]["agent"]["owe_post"] is True and st["config"].get("POST") == "submit",
      (paused, st and st["agents"][0]["agent"].get("owe_post"), st and st["config"]))
hook.update(stop_at=None)
ck.stop = False
agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, NEW, ckpt=ck))
rec2 = traces()[-1]
got = (len(rec2["submissions"]), [(n, m) for _, n, m in rec2["board"]], [x["post_refused"] for x in rec2["agents"]],
       rec2["rounds"], rec2["S"])
check("the resumed run refuses the next command and ends as the uninterrupted one",
      got == ref and turn_of(agents[0], "touch marker1").get("post_refused"), (got, ref))
done()
