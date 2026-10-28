"""The trace keeps whole turns and submissions with their code, and every turn's accounting (tokens, used, dropped,
think_s, the session's last_call; seconds and the episode's waits on a turn that ran an action), plus the per-episode
time sums in info (audit 2026-10-24). The real Agent.think with a fake session. Stubbed judge."""
import asyncio
import json

from helpers import LABEL, MSG, block, check, done, env, inp, sessions, traces

env(MA_LG_N=1, MA_LG_TOKENS=4000)
from flint import game as G  # noqa: E402

LONG = "r" * 10000


async def fake_judge(label, code):
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


SCRIPT = [block("printf 'int main(){}\\n' > solution.cpp").replace("<think>r0", "<think>" + LONG), block("submit"),
          "<think>think</think>THOUGHT: go.", block("true")]


async def reply(sess, messages):
    sess.last_call = {"llm_s": 0.01, "prompt_tokens": 1000, "completion_tokens": 1000, "cached_tokens": None}
    return SCRIPT[min(sess.calls, 3)], False


def new_session(input, name):
    s = sessions(reply, tokens=1000)(input, name)
    s.calls = 0
    real = s.turn

    async def turn(messages, max_tokens):
        out = await real(messages, max_tokens)
        s.calls += 1
        return out
    s.turn = turn
    return s


G.judge_code = fake_judge
agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, new_session))
rec = traces()[-1]
turns = rec["agents"][0]["turns"]
check("the episode ran 4 turns and used its budget", len(turns) == 4 and rec["agents"][0]["exit"] == "TokensExceeded",
      [t.get("command") for t in turns])
check("a 10k-character turn survives whole", LONG in turns[0]["text"], len(turns[0]["text"]))
check("every turn has tokens, used, dropped, think_s and t",
      all(t["tokens"] > 0 and t["used"] and "dropped" in t and "think_s" in t and "t" in t for t in turns), turns)
acted = [t for t in turns if "command" in t]
check("every turn that acted has seconds and harness_s",
      len(acted) == 3 and all("seconds" in t and "harness_s" in t for t in acted), acted)
check("a shell command has its queue wait, a submit its judge time",
      "cmd_queue_s" in acted[0] and "judge_s" in acted[1] and "_timing" not in json.dumps(turns), acted)
check("last_call merged into every turn", all("llm_s" in t for t in turns), turns)
check("the submission keeps its code and judge time", rec["submissions"][0]["code"].startswith("int main")
      and "judge_s" in rec["submissions"][0] and rec["best"] and "code" not in rec["best"], rec["submissions"])
check("per-episode time sums in info", all(k in info for k in (
    "fo_lg_wall_min", "fo_lg_llm_min", "fo_lg_think_min", "fo_lg_cmd_queue_min", "fo_lg_judge_min",
    "fo_lg_harness_min")) and info["fo_lg_llm_min"] > 0, info)
check("the observation the model saw has no timing", "_timing" not in json.dumps(agents[0].chat))
done()
