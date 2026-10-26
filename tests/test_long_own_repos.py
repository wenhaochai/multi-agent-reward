"""fcs_agent_long with MA_LG_REPOS=own (owner 2026-10-25): two agents, each in a repository of its own, share code only
through the harness's team.git. Scripted turns (stub session), stubbed judge, the real command sandbox: a teammate's
commit is visible as team/agent<k> one command later, `git show` and `git cherry-pick` bring it over with its author,
no agent can write another's repository or team.git or read another's working tree or /tmp, SCORES.md and BOARD.md
are never committed, and the trace counts the foreign commits. Run in miles.sif."""
import asyncio
import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_N"] = "2"
os.environ["MA_LG_REPOS"] = "own"
os.environ["MA_LG_TOKENS"] = "100000"
os.environ["TMPDIR"] = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
TRACE = os.environ["MA_AG_TRACE_DIR"] = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_agent_long as L  # noqa: E402

LABEL = "/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems/0"
FIN = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
HACK = ("echo x > ../agent2/hack.cpp && echo W1; echo x > ../team.git/hack && echo W2; "
        "git -C ../team.git update-ref refs/heads/agent2 HEAD && echo W3; cat ../agent2/solution.cpp && echo R1; "
        "echo secret > /tmp/s && echo TMP")
SCRIPTS = {   # one command per round; agent2 starts reading in round 3, when agent1's round-1 commit is published
    "agent1": ["printf 'int main(){return 1;}\\n' > solution.cpp && printf '// util\\n' > util.h", "submit", HACK,
               "post use my util.h", FIN],
    "agent2": ["true", "true", "git show team/agent1:solution.cpp > solution.cpp && cat solution.cpp",
               "git cherry-pick team/agent1 >/dev/null && git log -1 --format=%an -- util.h && cat util.h",
               "cat /tmp/s || echo NOTMP; test -f SCORES.md && test -f BOARD.md && echo LOGS; echo ===; "
               "git status --porcelain; git log --all --name-only --format= | sort -u", "submit", FIN],
}


async def fake_judge(label, code):
    return {"cases": [0.5] * 10, "status": "ok", "infra_error": "", "msg": ""}


class Stub:
    def __init__(self, input, name, think, max_len=None):
        self.input, self.segments, self.last_messages, self.used = input, [], None, 0
        self.script = list(SCRIPTS[name])

    async def turn(self, messages, max_tokens):
        cmd = self.script.pop(0)
        self.segments.append(SimpleNamespace(loss_mask=[1] * 50))
        self.used = 1000
        return f"<think>t</think>THOUGHT: go.\n\n```mswea_bash_command\n{cmd}\n```", False


L.judge_code = fake_judge
L.Session = Stub
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


pr = L.instance("Read n.", "agent2")
check("prompt: own repository and team branches", "a git repository of its own" in pr and "team/agent1 to team/agent2"
      in pr and "git show team/agent2:solution.cpp" in pr and "shared git repository" not in pr, pr[:1500])
check("harness tag says own", L.harness_tag().endswith("-own"), L.harness_tag())

inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
rec = [json.loads(x) for f in glob.glob(os.path.join(TRACE, "*.jsonl")) for x in open(f)][-1]
out = {a["name"]: [t.get("output", "") for t in a["turns"]] for a in rec["agents"]}
a1, a2 = out["agent1"], out["agent2"]
for a in rec["agents"]:
    print(a["name"], [(t.get("command", "")[:20], t.get("seconds"), t.get("fetch_s"), t.get("harness_s"), t.get("think_s"))
                      for t in a["turns"]])
check("both agents ran their scripts and finished", [a["exit"] for a in rec["agents"]] == ["Finished", "Finished"]
      and len(a1) == 5 and len(a2) == 7, {k: len(v) for k, v in out.items()})
check("a teammate's file comes over with git show", "int main(){return 1;}" in a2[2], a2[2])
check("cherry-pick keeps the author",
      "agent1\n// util" in a2[3], a2[3])
check("no write to another's repository or team.git, no read of another's tree",
      not any(w in a1[2] for w in ("W1", "W2", "W3", "R1")) and "TMP" in a1[2], a1[2])
check("/tmp is the agent's own", "NOTMP" in a2[4] and "secret" not in a2[4], a2[4])
tail = a2[4].split("===")[-1]
check("SCORES.md and BOARD.md in each repository, never committed or listed as changes",
      "LOGS" in a2[4] and "SCORES.md" not in tail and "BOARD.md" not in tail and "util.h" in tail, a2[4])
ev = "".join(a2)
check("the submission event names the commit on team/agent1", "submitted solution.cpp (commit " in ev
      and "on team/agent1)" in ev and "use my util.h" in ev, ev[-3000:])
check("two submissions, best by agent1", len(rec["submissions"]) == 2 and rec["best"]["agent"] == "agent1",
      rec["submissions"])
check("blame of the best is agent1's", rec["blame"] == {"agent1": 1.0}, rec["blame"])
check("foreign commits: agent2 holds agent1's", rec["foreign_commits"]["agent2"] >= 1
      and rec["foreign_commits"]["agent1"] == 0 and info["fo_lg_foreign_commits"] > 0, rec["foreign_commits"])
check("git log comes from team.git", "agent1" in rec["git_log"] and "agent2" in rec["git_log"], rec["git_log"])
check("info and trace say own repos", info["fo_lg_own_repos"] == 1.0 and rec["repos"] == "own"
      and rec["harness"].endswith("-own") and "fo_lg_fetch_min" in info, info)
check("fetch time recorded on shell turns", all("fetch_s" in t for a in rec["agents"] for t in a["turns"]
                                                if t.get("command") and not t["command"].startswith(("submit", "post"))),
      [t.keys() for t in rec["agents"][0]["turns"]])
left = os.listdir(os.path.join(os.environ["TMPDIR"], "agent_long"))
check("work dir, team.git and every /tmp removed", left == [], left)

# the checkpoint keeps every repository and team.git, and no agent's /tmp
work = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
L.own_init(work, ["agent1", "agent2"])
os.makedirs(os.path.join(work, "agent1.tmp"))
open(os.path.join(work, "agent1.tmp", "x"), "w").write("x")
ck = L.Checkpoint(os.path.join(tempfile.mkdtemp(dir=os.environ["TMPDIR"]), "e0"))
ck.save({"config": {}, "rounds": 0}, work)
back = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
ck.restore(ck.load(), back)
check("checkpoint: repositories and team.git kept, /tmp dropped",
      sorted(os.listdir(back)) == ["agent1", "agent2", "team.git"]
      and L.T._git(os.path.join(back, "agent2"), "rev-parse", "team/agent1").returncode == 0, sorted(os.listdir(back)))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
