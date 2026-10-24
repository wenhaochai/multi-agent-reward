"""fcs_agent_long's trace keeps whole turns and submissions with their code, and every turn's accounting (tokens,
used, dropped, think_s, the session's last_call if it has one; seconds and the episode's waits on a turn that ran an
action), plus the per-episode time sums in info (audit 2026-10-24). The real _Agent.think with a stub session: one
with last_call (chat_eval_long.ChatSession's shape) and one without (team_rollout.Session's). Stubbed judge."""
import asyncio
import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_N"] = "1"
os.environ["MA_LG_TOKENS"] = "100000"
os.environ["TMPDIR"] = tempfile.mkdtemp()
TRACE = os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_agent as A  # noqa: E402
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
LONG = "r" * 10000


async def fake_judge(label, code):
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


def reply(cmd: str | None, reasoning: str = "think") -> str:
    body = "THOUGHT: go." + (f"\n\n```mswea_bash_command\n{cmd}\n```" if cmd else "")
    return f"<think>{reasoning}</think>{body}"


class Stub:
    with_last_call = True

    def __init__(self, input, name, think, max_len=None):
        self.input, self.segments, self.last_messages, self.used = input, [], None, 0
        self.script = [reply("printf 'int main(){}\\n' > solution.cpp", LONG), reply("submit"), reply(None),
                       reply(FIN)]

    async def turn(self, messages, max_tokens):
        text = self.script.pop(0)
        n = len(text) // 4
        self.segments.append(SimpleNamespace(loss_mask=[1] * n))
        self.used = 1000 + n
        if Stub.with_last_call:
            self.last_call = {"llm_s": 0.01, "prompt_tokens": 1000, "completion_tokens": n, "cached_tokens": None}
        return text, False


L.judge_code = fake_judge
L.Session = Stub
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


for with_lc in (True, False):
    Stub.with_last_call = with_lc
    for f in glob.glob(os.path.join(TRACE, "*.jsonl")):
        os.remove(f)
    inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
    agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
    rec = [json.loads(x) for f in glob.glob(os.path.join(TRACE, "*.jsonl")) for x in open(f)][-1]
    turns = rec["agents"][0]["turns"]
    tag = "with last_call" if with_lc else "without last_call"
    check(f"{tag}: the episode ran 4 turns and finished", len(turns) == 4 and rec["agents"][0]["exit"] == "Finished",
          [t.get("command") for t in turns])
    check(f"{tag}: a 10k-character turn survives whole", LONG in turns[0]["text"], len(turns[0]["text"]))
    check(f"{tag}: every turn has tokens, used, dropped, think_s and t",
          all(t["tokens"] > 0 and t["used"] and "dropped" in t and "think_s" in t and "t" in t for t in turns), turns)
    acted = [t for t in turns if "command" in t]
    check(f"{tag}: every turn that acted has seconds and harness_s",
          len(acted) == 3 and all("seconds" in t and "harness_s" in t for t in acted), acted)
    check(f"{tag}: a shell command has its queue wait, a submit its judge time",
          "cmd_queue_s" in acted[0] and "judge_s" in acted[1] and "_timing" not in json.dumps(turns), acted)
    check(f"{tag}: last_call merged only when the session has it", all(("llm_s" in t) == with_lc for t in turns), turns)
    check(f"{tag}: the submission keeps its code and judge time", rec["submissions"][0]["code"].startswith("int main")
          and "judge_s" in rec["submissions"][0] and rec["best"] and "code" not in rec["best"], rec["submissions"])
    check(f"{tag}: per-episode time sums in info", all(k in info for k in (
        "fo_lg_wall_min", "fo_lg_llm_min", "fo_lg_think_min", "fo_lg_cmd_queue_min", "fo_lg_judge_min",
        "fo_lg_harness_min")) and (info["fo_lg_llm_min"] > 0) == with_lc, info)
    check(f"{tag}: the observation the model saw has no timing", "_timing" not in json.dumps(agents[0].chat))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
