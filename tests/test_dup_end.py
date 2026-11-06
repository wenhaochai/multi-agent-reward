"""The duplicate-submit end (MA_LG_DUP_END=K with MA_LG_DEDUP_SUBMIT=1; owner popups 2026-11-06 「RL里面应该也惩罚这个
行为」, 「首先prompt里写，然后触发的那个turn就记0分，前面都正常」). (1) The setting is checked at import. (2) A lone scripted
agent (tests/helpers.scripted, stubbed judge): the K-th unjudged duplicate submit in a row ends its game at that turn
(exit DupLoop, the turn marked dup_end, its reply says so), later commands never run, a judged submission in between
starts the count again, S and the best provisional are those of the judged submissions, the harness tag ends in
-dupend<K>, info has fo_lg_dup_end; off (K = 0) the same script runs to its end. (3) The instance message states the
rule only with K > 0, and with K = 0 it is the committed prompts.py's message byte for byte. (4) In RL
(flint.miles_agent): split_dup_end_turns moves the last trained run of the episode's newest leaf into a sample of its
own marked dup_end_turn (the episode's trained tokens unchanged; other episodes and groups out of the loss untouched),
and reward_post_process gives that sample 0 minus the group mean, every other sample S minus the group mean (miles'
normalization over episodes). python3 tests/test_dup_end.py (no GPU; (4) imports miles: run in miles-v0.1.3.sif with
miles-rollout2 on PYTHONPATH)."""
import asyncio
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from helpers import LABEL, MSG, ROOT, check, done, env, inp, scripted, sessions, traces

env(MA_LG_N=1, MA_LG_TOKENS=10000, MA_LG_DEDUP_SUBMIT=1, MA_LG_DUP_END=3)
from flint import game as G  # noqa: E402
from flint import prompts as P  # noqa: E402


def clean_env(**kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith("MA_")}
    e.update({k: str(v) for k, v in kw.items()})
    return e


# (1) the setting at import
for kw, ok in ((dict(MA_LG_DUP_END="0"), True), (dict(MA_LG_DUP_END=""), True),
               (dict(MA_LG_DUP_END="50", MA_LG_DEDUP_SUBMIT="1"), True), (dict(MA_LG_DUP_END="50"), False),
               (dict(MA_LG_DUP_END="-1", MA_LG_DEDUP_SUBMIT="1"), False), (dict(MA_LG_DUP_END="x", MA_LG_DEDUP_SUBMIT="1"), False)):
    p = subprocess.run([sys.executable, "-c", "import flint.game as G; print(G.DUP_END)"], cwd=ROOT,
                       env=clean_env(**kw), capture_output=True, text=True)
    check(f"import with {kw}: {'accepted' if ok else 'an error'}", (p.returncode == 0) == ok, (p.stdout, p.stderr[-300:]))

# (2) a lone agent, scripted
judged = []


async def fake_judge(label, code):
    judged.append(code)
    m = re.search(r"P=(\d+)", code)
    return {"cases": [int(m.group(1)) / 100 if m else 0.1] * 10, "status": "ok", "infra_error": "", "msg": ""}


Scripted = scripted(G)
G.judge_code, G.INFRA_BACKOFF = fake_judge, 0.01
real_agent = G.Agent


def run(script, k):
    G.Agent, G.DUP_END = Scripted, k
    Scripted.script = list(script)
    judged.clear()
    agents, S, info = asyncio.run(G.episode(inp(), MSG, LABEL, sessions(None)))
    G.Agent, G.DUP_END = real_agent, 3
    return agents[0], S, info, traces()[-1]


A = "printf 'int main(){return 0;}// P=40\\n' > solution.cpp"
B = "printf 'int main(){return 1;}// P=55\\n' > solution.cpp"
sc = [A, "submit", "submit", "submit", "submit", B, "submit"]
a, S, info, tr = run(sc, 3)
turns = tr["agents"][0]["turns"]
check("K=3: one judged submit, then the 3rd duplicate in a row ends the game", len(judged) == 1 and a.exit == "DupLoop"
      and a.done, (len(judged), a.exit))
check("K=3: B and its submit never ran (the trace ends at the 5th command)", len(turns) == 5 and turns[-1].get("dup_end")
      and not any(t.get("dup_end") for t in turns[:-1]), [(t.get("command"), t.get("dup_end")) for t in turns])
check("K=3: the last reply says the game ends and the submission scores 0",
      "your game ends now and this submission scores 0" in a.outs[-1], a.outs[-1][-300:])
check("K=3: S and the best provisional are the judged submission's", abs(S - 0.4) < 1e-9
      and abs(info["fo_lg_best_prov"] - 0.4) < 1e-9, (S, info["fo_lg_best_prov"]))
check("K=3: info fo_lg_dup_end 1, exit DupLoop counted, harness tag -dupend3", info["fo_lg_dup_end"] == 1.0
      and info["fo_ag_exit_DupLoop"] == 1.0 and tr["harness"].endswith("-dedup-dupend3"),
      (info.get("fo_lg_dup_end"), info.get("fo_ag_exit_DupLoop"), tr["harness"]))
sc2 = [A, "submit", "submit", "submit", B, "submit", "submit", "submit", "submit"]
a, S, info, tr = run(sc2, 3)
check("K=3: a judged submit in between starts the count again (2 dups, B judged, then 3 dups end it)",
      len(judged) == 2 and a.exit == "DupLoop" and len(tr["agents"][0]["turns"]) == 9 and abs(S - 0.55) < 1e-9,
      (len(judged), a.exit, len(tr["agents"][0]["turns"]), S))
a, S, info, tr = run(sc, 0)
check("K=0 (off): the same script runs to its end, B judged, no DupLoop, no dup-end keys in info",
      a.exit != "DupLoop" and len(judged) == 2 and "fo_lg_dup_end" not in info and "fo_ag_exit_DupLoop" not in info
      and not tr["harness"].endswith("dupend3") and len(tr["agents"][0]["turns"]) == 7,
      (a.exit, len(judged), sorted(k for k in info if "dup" in k.lower()), tr["harness"]))

# (3) the instance message
spec = importlib.util.spec_from_loader("old_prompts", loader=None)
old = importlib.util.module_from_spec(spec)
src = subprocess.run(["git", "-C", ROOT, "show", "HEAD:flint/prompts.py"], capture_output=True, text=True).stdout
exec(compile(src, "old_prompts.py", "exec"), old.__dict__)
for n in (1, 4):
    check(f"n={n}, K=0: the instance message is HEAD's byte for byte",
          P.instance("PROBLEM", "agent1", n, 128000, 60) == old.instance("PROBLEM", "agent1", n, 128000, 60))
    m = P.instance("PROBLEM", "agent1", n, 128000, 60, dup_end=50)
    check(f"n={n}, K=50: the message states the rule once, after the submit command",
          m.count("after 50 such submissions in a row your game ends at once and that submission scores 0") == 1
          and m.index("Do not submit an unchanged program again") < m.index("`post <message>`"), m[:200])

# (4) RL: the sample split and the rewards
from flint import miles_agent as MA  # noqa: E402


def smp(rid, group, node, mask, S, dup=0.0):
    return SimpleNamespace(rollout_id=rid, index=rid, group_index=group, reward=S, loss_mask=list(mask),
                           remove_sample=False, metadata={"fo_lg_dup_end": dup, "leaf": {"node_id": node}})


e1 = [smp(1, 7, 3, [0, 1, 1, 0, 1, 1, 1], 0.6, 1.0), smp(1, 7, 9, [0, 1, 0, 1, 1], 0.6, 1.0)]   # newest leaf: node 9
e2 = [smp(2, 7, 4, [1, 1, 0, 1], 0.2)]
e3 = [smp(3, 8, 5, [1, 1, 1], 0.0, 1.0)]
e3[0].remove_sample = True   # a group out of the loss stays as it is
data = [[e1, e2], [e3]]
before = sum(sum(s.loss_mask) for s in e1)
n = MA.split_dup_end_turns(data)
check("split: one episode split (the removed group untouched)", n == 1 and len(e1) == 3 and len(e3) == 1, (n, len(e1), len(e3)))
check("split: the newest leaf's last trained run moved to a sample marked dup_end_turn", e1[1].loss_mask == [0, 1, 0, 0, 0]
      and e1[2].loss_mask == [0, 0, 0, 1, 1] and e1[2].metadata.get("dup_end_turn") and not e1[1].metadata.get("dup_end_turn")
      and e1[0].loss_mask == [0, 1, 1, 0, 1, 1, 1], [s.loss_mask for s in e1])
check("split: the episode's trained tokens unchanged", sum(sum(s.loss_mask) for s in e1) == before)
try:
    from miles.utils.types import Sample
except ImportError:
    Sample = None
if Sample is None:
    check("reward_post_process: miles importable (run in miles-v0.1.3.sif with miles-rollout2 on PYTHONPATH)", False)
else:
    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False,
                           n_samples_per_prompt=2, rollout_batch_size=1, reward_key=None, eval_reward_key=None)

    def S_(rid, group, r, turn=False):
        s = Sample(index=rid, group_index=group, reward=r)
        s.rollout_id = rid
        s.metadata = {"dup_end_turn": True} if turn else {}
        return s

    ss = [S_(1, 0, 0.6), S_(1, 0, 0.6), S_(1, 0, 0.6, turn=True), S_(2, 0, 0.2), S_(3, 1, 0.5), S_(4, 1, 0.1)]
    raw, norm = MA.reward_post_process(args, ss)
    want = [0.2, 0.2, -0.4, -0.2, 0.2, -0.2]   # group 0 mean over episodes 0.4: the turn gets 0 - 0.4
    check("reward_post_process: the turn 0 - mean, the rest S - mean (the mean over episodes, not samples)",
          all(abs(x - y) < 1e-6 for x, y in zip(norm, want)) and raw == [0.6, 0.6, 0.6, 0.2, 0.5, 0.1], (raw, norm))
done()
