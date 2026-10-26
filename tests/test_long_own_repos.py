"""fcs_agent_long with MA_LG_REPOS=own (owner 2026-10-25): two agents, each in a repository of its own, share code only
through the harness's team.git. Scripted turns (stub session), stubbed judge, the real command sandbox: a teammate's
commit is visible as team/agent<k> one command later; `git checkout team/agent1 -- solution.cpp` brings its file over
(counted as an adoption) and `git cherry-pick` keeps the author; no agent can write another's repository or team.git or
read another's working tree or /tmp; SCORES.md and BOARD.md are never committed. Audit 2026-10-25: a conflicted
cherry-pick is not committed by the harness (the agent is told, and --abort still works); a deleted .git is told and
counted; a submission's event names the commit only when that commit holds the submitted code (the finish path too).
Run in miles.sif, with FCS_CASE_PROC=0 and 1."""
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
SCRIPTS = {   # one command per round; what agent2 reads in round r was published by agent1 in round r-1 or earlier
    "agent1": ["printf 'int main(){return 1;}\\n' > solution.cpp && printf '// util\\n' > util.h", "submit", HACK,
               "post use my util.h", "printf 'int main(){return 5;}\\n' > solution.cpp && " + FIN, FIN],
    "agent2": ["printf 'int main(){return 2;}\\n' > solution.cpp", "true",
               "git checkout team/agent1 -- solution.cpp && cat solution.cpp",
               "git cherry-pick team/agent1 >/dev/null && git log -1 --format=%an -- util.h && cat util.h",
               "cat /tmp/s || echo NOTMP; test -f SCORES.md && test -f BOARD.md && echo LOGS; echo ===; "
               "git status --porcelain; git log --all --name-only --format= | sort -u; echo ===; "
               "printf 'int main(){return 7;}\\n' > solution.cpp",
               "git cherry-pick team/agent1; git status --porcelain",
               "git cherry-pick --abort && git log -1 --format='%an %s' && cat solution.cpp",
               "rm -rf .git && echo gone", "submit", FIN],
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
check("prompt: own repository, team branches, the checkout recipe", "a git repository of its own" in pr
      and "team/agent1 to team/agent2" in pr and "git checkout team/agent2 -- solution.cpp" in pr
      and "cherry-pick" not in pr and "shared git repository" not in pr, pr[:1500])
check("harness tag says own", L.harness_tag().endswith("-own"), L.harness_tag())

inp = SimpleNamespace(sample=SimpleNamespace(index=0), evaluation=True)
agents, S, info = asyncio.run(L.episode(inp, [{"role": "user", "content": "problem"}], LABEL))
rec = [json.loads(x) for f in glob.glob(os.path.join(TRACE, "*.jsonl")) for x in open(f)][-1]
turns = {a["name"]: a["turns"] for a in rec["agents"]}
out = {k: [t.get("output", "") for t in v] for k, v in turns.items()}
a1, a2 = out["agent1"], out["agent2"]
check("both agents ran their scripts and finished", [a["exit"] for a in rec["agents"]] == ["Finished", "Finished"]
      and len(a1) == 6 and len(a2) == 10, {k: len(v) for k, v in out.items()})
check("a teammate's file comes over with git checkout", "int main(){return 1;}" in a2[2], a2[2])
check("the copy is one adoption, from agent1 in round 3", rec["adoptions"] == [[3, "agent2", "agent1"]]
      and info["fo_lg_adoptions"] == 1, rec["adoptions"])
check("cherry-pick keeps the author", "agent1\n// util" in a2[3], a2[3])
check("no write to another's repository or team.git, no read of another's tree",
      not any(w in a1[2] for w in ("W1", "W2", "W3", "R1")) and "TMP" in a1[2], a1[2])
check("/tmp is the agent's own", "NOTMP" in a2[4] and "secret" not in a2[4], a2[4])
mid = a2[4].split("===")[1] if a2[4].count("===") == 2 else ""
check("SCORES.md and BOARD.md in each repository, never committed or listed as changes",
      "LOGS" in a2[4] and "SCORES.md" not in mid and "BOARD.md" not in mid and "util.h" in mid, a2[4])
check("a conflicted cherry-pick is reported and not committed", "CONFLICT" in a2[5] and "UU solution.cpp" in a2[5]
      and "[harness] A git cherry-pick is in progress" in a2[5], a2[5])
check("--abort still works and HEAD is agent2's own step 5", "agent2 agent2 step 5" in a2[6]
      and "return 7" in a2[6] and "[harness]" not in a2[6], a2[6])
check("team.git never got the conflict", "step 6" not in rec["git_log"] and "agent2 step 5" in rec["git_log"],
      rec["git_log"])
check("a deleted .git is told", "[harness] Your repository has no .git directory" in a2[7], a2[7])
check("the fetch into a repository without .git is a recorded failure", rec["sync"]["fetch_fail"] >= 1
      and any("Fetching the team's branches" in n for n in turns["agent2"][9].get("git_notes", [])), rec["sync"])
check("skipped commits counted", rec["sync"]["commit_skipped"] >= 4 and info["fo_lg_commit_skipped"] >= 4, rec["sync"])
subs = rec["submissions"]
check("three submissions: agent1, agent1 (finish path), agent2", [s["agent"] for s in subs] == ["agent1", "agent1", "agent2"],
      [(s["agent"], s["round"]) for s in subs])
ev = "".join(a2)
r0, r1 = subs[0]["rev"], subs[1]["rev"]
check("each event names the commit that holds its code, the finish path included", r0 and r1 and r0 != r1
      and f"submitted solution.cpp (commit {r0[:7]} on team/agent1)" in ev
      and f"submitted solution.cpp (commit {r1[:7]} on team/agent1)" in ev and subs[1]["path"] == "solution.cpp", ev[-3000:])
check("a submission from a repository without .git names no commit", subs[2]["rev"] == "" and subs[2]["path"] == "",
      subs[2])
check("best by agent1, blamed in team.git", rec["best"]["agent"] == "agent1" and rec["blame"] == {"agent1": 1.0},
      (rec["best"], rec["blame"]))
check("git log comes from team.git, newest first", rec["git_log"].startswith("agent") and "harness start" in rec["git_log"],
      rec["git_log"])
check("info and trace say own repos", info["fo_lg_own_repos"] == 1.0 and rec["repos"] == "own"
      and rec["harness"].endswith("-own") and "fo_lg_fetch_min" in info and "fo_lg_git_fail" in info, info)
check("fetch time recorded on shell turns", all("fetch_s" in t for v in turns.values() for t in v
                                                if t.get("command") and not t["command"].startswith(("submit", "post"))),
      [t.keys() for t in turns["agent1"]])
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
