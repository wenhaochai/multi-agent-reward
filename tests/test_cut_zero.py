"""The cut-reply rule (MA_LG_CUT_ZERO=1; owner 2026-11-06, plan B, after production drifted into replies cut at the
turn limit). (1) The setting is checked at import. (2) The instance message states the rule only when on, next to the
budget, at the game's turn limit; off, it is 6f9d54d's message byte for byte, with dupend on or off. (3) The harness
tag ends in -cutzero. (4) split_cut_turns: a sample whose status is truncated at max_tokens = the turn limit gives its
last run of trained tokens to a copy marked cut_turn (the episode's trained tokens unchanged, one rollout_id for the
episode); a cut at a smaller limit (a budget that runs out), a removed group and a completed sample stay as they
are; with dupend, both splits happen and do not take each other's runs. (5) reward_post_process: a cut_turn sample
gets 0 minus the group mean, like a dup_end_turn one; the rest S minus the mean. (6) The settings check: on, an
episode reports MA_LG_CUT_ZERO as rl_inner.sh records it. (7) On w1319's dumped rollout 0 (real samples): one
cut_turn sample per reply cut at 16,384 tokens in the traces (8 of the 23 truncated samples; the other 15 are budget
ends; a copy, or the sample itself when the reply is all it trains), each training 16,384 tokens, and miles'
compact-rollout check passes. python3 tests/test_cut_zero.py in
miles-v0.1.3.sif with miles-numa4 on PYTHONPATH (no GPU)."""
import glob
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from helpers import ROOT, check, done, env

env(MA_LG_N=1, MA_LG_TOKENS=128000, MA_LG_DEDUP_SUBMIT=1, MA_LG_DUP_END=50, MA_LG_CUT_ZERO=1, MA_AG_TURN=16384)
from flint import game as G  # noqa: E402
from flint import miles_agent as MA  # noqa: E402
from flint import prompts as P  # noqa: E402


def clean_env(**kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith("MA_")}
    e.update({k: str(v) for k, v in kw.items()})
    return e


# (1) the setting at import
for kw, ok in ((dict(MA_LG_CUT_ZERO="0"), True), (dict(MA_LG_CUT_ZERO=""), True), (dict(MA_LG_CUT_ZERO="1"), True),
               (dict(MA_LG_CUT_ZERO="2"), False), (dict(MA_LG_CUT_ZERO="yes"), False)):
    p = subprocess.run([sys.executable, "-c", "import flint.game as G; print(G.CUT_ZERO)"], cwd=ROOT,
                       env=clean_env(**kw), capture_output=True, text=True)
    check(f"import with {kw}: {'accepted' if ok else 'an error'}", (p.returncode == 0) == ok, (p.stdout, p.stderr[-300:]))

# (2) the instance message
spec = importlib.util.spec_from_loader("old_prompts", loader=None)
old = importlib.util.module_from_spec(spec)
src = subprocess.run(["git", "-C", ROOT, "show", "6f9d54d:flint/prompts.py"], capture_output=True, text=True).stdout
exec(compile(src, "old_prompts.py", "exec"), old.__dict__)
RULE = "A reply that reaches 16,384 tokens is cut off there: it runs no command and scores 0."
for n in (1, 4):
    for dup in (0, 50):
        check(f"n={n}, dupend={dup}, cut off: the message is 6f9d54d's byte for byte",
              P.instance("PROBLEM", "agent1", n, 128000, 60, dup_end=dup)
              == old.instance("PROBLEM", "agent1", n, 128000, 60, dup_end=dup))
    m = P.instance("PROBLEM", "agent1", n, 128000, 60, dup_end=50, cut_turn=16384)
    check(f"n={n}, cut on: the rule once, after the budget sentence and before the commands",
          m.count(RULE) == 1 and m.index("you can make it better.") < m.index(RULE) < m.index("Three commands are provided"),
          m[:400])
    check(f"n={n}, cut on: the message is the old one with the rule added and nothing else",
          m.replace(" " + RULE, "") == old.instance("PROBLEM", "agent1", n, 128000, 60, dup_end=50))
check("the game states the rule at its turn limit", RULE in G.instance("PROBLEM", "agent1"))

# (3) the harness tag
check("harness tag ends in -dedup-dupend50-cutzero", G.harness_tag().endswith("-dedup-dupend50-cutzero"), G.harness_tag())

# (4) the split, on stand-ins, at a turn limit of 3 tokens (the run must end the response and be the limit long)
TRUNC, DONE_ = SimpleNamespace(value="truncated"), SimpleNamespace(value="completed")


def smp(rid, group, node, mask, S, status=DONE_, cap=3, dup=0.0):
    return SimpleNamespace(rollout_id=rid, index=rid if rid is not None else 90 + node, group_index=group, reward=S,
                           loss_mask=list(mask), remove_sample=False, status=status,
                           metadata={"fo_lg_dup_end": dup, "leaf": {"node_id": node}, "turn_args": {"max_tokens": cap}})


os.environ["MA_AG_TURN"] = "3"
a = [smp(1, 7, 2, [0, 1, 1, 0, 1], 0.6), smp(1, 7, 4, [1, 0, 1, 1, 1], 0.6, TRUNC)]    # cut reply + other turns: a copy
b = [smp(2, 7, 3, [1, 1, 0, 1, 1], 0.2, TRUNC, cap=2)]                                 # the budget's last reply: kept
c = [smp(3, 8, 5, [1, 1, 1], 0.0, TRUNC)]
c[0].remove_sample = True                                                              # a group out of the loss
d = [smp(None, 8, 0, [0, 1, 1, 1], 0.5, TRUNC)]                                        # the cut reply is all it trains
e = [smp(5, 8, 1, [1, 1, 0, 1, 1, 1], 0.3, TRUNC), smp(5, 8, 6, [1, 0, 1, 1], 0.3, DONE_, dup=1.0)]   # a cut and a dup end
f_ = [smp(6, 9, 2, [1, 1, 0, 1, 1], 0.4, TRUNC)]      # a length finish short of the limit (the context, say): skipped
g = [smp(7, 9, 3, [1, 1, 1, 0], 0.4, TRUNC)]          # a last run that does not end the response: skipped
data = [[a, b], [c], [d, e], [f_, g]]
before = {id(x): sum(sum(s.loss_mask) for s in x) for x in (a, b, c, d, e, f_, g)}
os.environ["MA_LG_DUP_END"] = "50"
nd = MA.split_dup_end_turns(data)
nc = MA.split_cut_turns(data)
check("split: one dup end, three cut replies (a, d, e), not the budget end b, the removed c, the odd f and g",
      nd == 1 and nc == 3 and len(a) == 3 and len(b) == 1 and len(c) == 1 and len(d) == 1 and len(e) == 4
      and len(f_) == 1 and len(g) == 1, (nd, nc, [len(x) for x in (a, b, c, d, e, f_, g)]))
check("split: a's cut reply moved to a copy marked cut_turn, the source marked split",
      a[1].loss_mask == [1, 0, 0, 0, 0] and a[2].loss_mask == [0, 0, 1, 1, 1] and a[2].metadata.get("cut_turn")
      and not a[1].metadata.get("cut_turn") and a[1].metadata.get("cut_split") and a[0].loss_mask == [0, 1, 1, 0, 1],
      [(s.loss_mask, s.metadata) for s in a])
check("split: d trains only its cut reply: marked in place, no copy, mask unchanged",
      d[0].metadata.get("cut_turn") and d[0].loss_mask == [0, 1, 1, 1], d[0].metadata)
check("split: f and g left as they are", not f_[0].metadata.get("cut_turn") and not g[0].metadata.get("cut_turn")
      and f_[0].loss_mask == [1, 1, 0, 1, 1] and g[0].loss_mask == [1, 1, 1, 0])
check("split: trained tokens per episode unchanged",
      all(sum(sum(s.loss_mask) for s in x) == before[id(x)] for x in (a, b, c, d, e, f_, g)))
check("split: one rollout_id per episode with copies", {s.rollout_id for s in a} == {1} and {s.rollout_id for s in e} == {5},
      [[s.rollout_id for s in x] for x in (a, e)])
check("split: e's dup-end copy and cut copy are different runs, one each",
      sum(1 for s in e if s.metadata.get("dup_end_turn")) == 1 and sum(1 for s in e if s.metadata.get("cut_turn")) == 1
      and [s.loss_mask for s in e if s.metadata.get("dup_end_turn")][0] == [0, 0, 1, 1]
      and [s.loss_mask for s in e if s.metadata.get("cut_turn")][0] == [0, 0, 0, 1, 1, 1], [(s.loss_mask, s.metadata) for s in e])
masks = [[list(s.loss_mask) for s in x] for x in (a, b, c, d, e, f_, g)]
check("split: a second call of either splits nothing more and changes no mask",
      MA.split_cut_turns(data) == 0 and MA.split_dup_end_turns(data) == 0
      and masks == [[list(s.loss_mask) for s in x] for x in (a, b, c, d, e, f_, g)])
os.environ["MA_AG_TURN"] = "16384"

# (4b) the job option
for args_, want in ((["cutzero=1"], "_cutzero_"), (["cutzero=2"], "cutzero=1 is the only value"),
                    (["cutzero=0"], "cutzero=1 is the only value")):
    r = subprocess.run(["bash", os.path.join(ROOT, "rl", "rl_job.sh"), "train67_test", "4", "8", "set=signal33", *args_],
                       env={**os.environ, "FETCHED_JOB_ID": "dry", "RL_JOB_PRINT_ONLY": "1"}, capture_output=True, text=True)
    check(f"rl_job.sh {' '.join(args_)}: {want!r}", want in (r.stdout + r.stderr), (r.stdout + r.stderr)[-300:])
r = subprocess.run(["bash", os.path.join(ROOT, "rl", "rl_job.sh"), "run20", "cutzero=1"],
                   env={**os.environ, "FETCHED_JOB_ID": "dry", "RL_JOB_PRINT_ONLY": "1"}, capture_output=True, text=True)
check("rl_job.sh run20 cutzero=1: refused (a train67 setting)", r.returncode != 0 and "cutzero=1" in (r.stdout + r.stderr),
      (r.returncode, (r.stdout + r.stderr)[-200:]))

# (5) the rewards, with real miles Samples
from miles.utils.types import Sample  # noqa: E402

args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False,
                       n_samples_per_prompt=2, rollout_batch_size=1, reward_key=None, eval_reward_key=None)


def S_(rid, group, r, mark=None):
    s = Sample(index=rid, group_index=group, reward=r)
    s.rollout_id = rid
    s.metadata = {mark: True} if mark else {}
    return s


ss = [S_(1, 0, 0.6), S_(1, 0, 0.6, "cut_turn"), S_(1, 0, 0.6, "dup_end_turn"), S_(2, 0, 0.2), S_(3, 1, 0.5),
      S_(4, 1, 0.1, "cut_turn"), S_(4, 1, 0.1)]
raw, norm = MA.reward_post_process(args, ss)
want = [0.2, -0.4, -0.4, -0.2, 0.2, -0.3, -0.2]   # group means over episodes 0.4 and 0.3
check("reward_post_process: cut_turn and dup_end_turn 0 - mean, the rest S - mean",
      all(abs(x - y) < 1e-6 for x, y in zip(norm, want)), (raw, norm))

# (6) the settings check
G.CUT_ZERO = True
got = MA._game_settings(G)
check("settings: on, the episode reports MA_LG_CUT_ZERO = 1", got.get("MA_LG_CUT_ZERO") == 1, got)
G.CUT_ZERO = False
check("settings: off, the key is absent, as before", "MA_LG_CUT_ZERO" not in MA._game_settings(G))
G.CUT_ZERO = True
check("rl_inner.sh records MA_LG_CUT_ZERO in game_settings.json",
      "'MA_LG_CUT_ZERO') if k in os.environ" in open(os.path.join(ROOT, "rl", "rl_inner.sh")).read())

# (7) real samples: w1319's rollout 0
import torch  # noqa: E402

from miles.ray.rollout.rollout_data_conversion import validate_compact_rollout_ids  # noqa: E402

R = "/scratch/gpfs/GROUP/USER/tmp/flint_s5/runs/train67_test_signal33_dupend1_p2g8"
raw_samples = []
for f in sorted(glob.glob(f"{R}/r0/samples/*.pt")):
    o = torch.load(f, weights_only=False)
    raw_samples.extend(o["samples"] if isinstance(o, dict) else o)


def as_sample(dct):
    s = Sample.from_dict(dct) if hasattr(Sample, "from_dict") else None
    if s is None:
        s = Sample()
        for k, v in dct.items():
            setattr(s, k, v)
    return s


reals = [as_sample(x) for x in raw_samples]
by_ep = {}
for s in reals:
    by_ep.setdefault((s.group_index, s.rollout_id if s.rollout_id is not None else ("i", s.index)), []).append(s)
groups = {}
for (g, _), eps in by_ep.items():
    groups.setdefault(g, []).append(eps)
real_data = [groups[g] for g in sorted(groups)]
cap_cuts = 0
for f in glob.glob(f"{R}/r0/traces/*.jsonl"):
    t = json.load(open(f))
    cap_cuts += sum(1 for x in t["agents"][0]["turns"]
                    if str(x.get("finish_reason")) == "length" and int(x.get("completion_tokens") or 0) == 16384)
before = sum(sum(s.loss_mask) for s in reals)
os.environ["MA_LG_DUP_END"] = "0"
n_real = MA.split_cut_turns(real_data)
after = [s for grp in real_data for ep in grp for s in MA._samples(ep)]
marked = [s for s in after if (s.metadata or {}).get("cut_turn")]
truncated = sum(1 for s in reals if str(getattr(s.status, "value", s.status)).lower() == "truncated")
check(f"real rollout 0: one cut_turn sample per reply cut at 16,384 tokens in the traces ({cap_cuts}; {truncated} truncated "
      f"samples; {len(after) - len(reals)} copies, the rest marked in place)",
      n_real == cap_cuts == len(marked) and cap_cuts > 0 and truncated > cap_cuts, (n_real, cap_cuts, len(marked), truncated))
check("real rollout 0: each cut_turn sample trains 16,384 tokens; all trained tokens unchanged",
      all(sum(s.loss_mask) == 16384 for s in marked) and sum(sum(s.loss_mask) for s in after) == before,
      ([sum(s.loss_mask) for s in marked], before))
err = None
try:
    validate_compact_rollout_ids(real_data)
except AssertionError as ex:
    err = str(ex)
check("real rollout 0: miles' compact-rollout check passes after the split", err is None, err)
done()
