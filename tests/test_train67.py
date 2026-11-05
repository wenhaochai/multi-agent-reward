"""train67 (rl/rl_job.sh train67*): (a) rl/train67_data.py: the 67 problems (112 only with its fixed checker), each
epoch a permutation fixed by the seed, every step P different problems, the same file on every build, byte-identical to
864a904's without a set; a set (data/train67_sets/signal34.txt) the same way over its 34 problems, an unknown pid an
error; (b) the GRPO
groups miles forms from it: P groups of G per step, one problem each, centered within the group; (c) the fractional
stop rules (wall cap over 25% of a rollout's episodes in 2 rollouts in a row; over 25% of the episodes out of the
loss), with run20's absolute rule kept when the fractions are unset; (d) log_rollout's train67 record: per-problem S
means, the fraction of episodes that submitted, the planned problems, the in-process sample check (failed episodes
out of the loss, other problems end the run) and the thin dumps. Runs in miles-v0.1.3.sif: tests/run_all.sh."""
import json
import os
import random
import subprocess
import tempfile
from types import SimpleNamespace

from helpers import ROOT, check, done, env

env(MA_LG_N=1, MA_LG_TOKENS=90, MA_AG_TOOL_STRICT=0)
import sys  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "rl"))
import train67_data as T  # noqa: E402
from flint import miles_agent as MA  # noqa: E402

SRC = os.path.join(ROOT, "data", "fcs11_train67_agent.jsonl")
rows = [json.loads(x) for x in open(SRC)]
pids = [r["metadata"]["pid"] for r in rows]

# (a) selection and order
check("data: 67 problems in the source, 112 among them", len(pids) == 67 and "112" in pids)
check("data: the fixed checker of 112 is in this tree", T.fixed_checker("112"))
sel = T.select(rows)
check("data: all 67 kept (112 has its fixed checker), label in the metadata",
      len(sel) == 67 and all(r["metadata"]["label"] == r["label"] for r in sel))
real = T.fixed_checker
T.fixed_checker = lambda pid: False
check("data: 112 left out without its fixed checker",
      [r["metadata"]["pid"] for r in T.select(rows)] == [p for p in pids if p != "112"])
T.fixed_checker = real
for P in (4, 3, 8, 5):
    seq = T.order(67, P, 67, 30)
    epochs = [seq[e * 67:(e + 1) * 67] for e in range(30)]
    steps = [seq[i:i + P] for i in range(0, len(seq) - P + 1, P)]
    check(f"order P={P}: every epoch a permutation of the 67", all(sorted(e) == list(range(67)) for e in epochs))
    check(f"order P={P}: every step {P} different problems", all(len(set(s)) == P for s in steps))
check("order: fixed by the seed, another seed differs", T.order(67, 4, 67, 30) == T.order(67, 4, 67, 30)
      and T.order(67, 4, 67, 30) != T.order(67, 4, 68, 30))
perm0 = list(range(67))
random.Random(67).shuffle(perm0)
check("order: epoch 0 is random.Random(seed)'s permutation", T.order(67, 4, 67, 30)[:64] == perm0[:64])
tmp = tempfile.mkdtemp()
a, b = os.path.join(tmp, "a.jsonl"), os.path.join(tmp, "b.jsonl")
T.build(SRC, a, 4)
T.build(SRC, b, 4)
check("build: byte-identical on every build (a resumed run reads the same file)", open(a, "rb").read() == open(b, "rb").read())
built = [json.loads(x) for x in open(a)]
check("build: 30 epochs of 67 rows", len(built) == 30 * 67)
first16 = {r["metadata"]["pid"] for r in built[:64]}
check("build: the first 16 steps cover 64 different problems", len(first16) == 64)
out = subprocess.run([sys.executable, os.path.join(ROOT, "rl", "train67_data.py"), SRC, os.path.join(tmp, "c.jsonl"), "4"],
                     capture_output=True, text=True)
check("cli: runs and reports", out.returncode == 0 and "67 problems (112 kept" in out.stdout, out.stdout + out.stderr)
# without a set: byte-identical to rl-train67's 864a904: its train67_data.py and flint/checkers, exported from git (so it
# sees 112's fixed checker as this tree does), run on the same source
btree = os.path.join(tmp, "btree")
r = subprocess.run(f"mkdir -p {btree} && git -C {ROOT} archive 864a904 rl/train67_data.py flint/checkers | tar -x -C {btree}",
                   shell=True, capture_output=True, text=True)
for P in (4, 3, 8, 5):
    x, y = os.path.join(tmp, f"b{P}.jsonl"), os.path.join(tmp, f"t{P}.jsonl")
    o1 = subprocess.run([sys.executable, os.path.join(btree, "rl", "train67_data.py"), SRC, x, str(P)], capture_output=True, text=True)
    o2 = subprocess.run([sys.executable, os.path.join(ROOT, "rl", "train67_data.py"), SRC, y, str(P)], capture_output=True, text=True)
    check(f"no set, P={P}: file and printed line byte-identical to 864a904's", r.returncode == 0 and o1.returncode == 0
          and o2.returncode == 0 and open(x, "rb").read() == open(y, "rb").read() and o1.stdout == o2.stdout,
          (r.stderr, o1.stderr, o2.stderr, o1.stdout, o2.stdout))

# a set: data/train67_sets/signal34.txt (owner popup 2026-10-31)
SIG = ["1", "9", "23", "24", "33", "41", "44", "47", "48", "50", "61", "75", "83", "109", "111", "112", "174", "177", "186",
       "188", "301", "302", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315"]
check("set signal34: the 34 pids of the owner's list", T.load_set("signal34") == SIG and len(SIG) == 34)
sel34 = T.select(rows, T.load_set("signal34"))
check("set signal34: 34 rows, in the source's order, label in the metadata",
      [r["metadata"]["pid"] for r in sel34] == [p for p in pids if p in SIG]
      and all(r["metadata"]["label"] == r["label"] for r in sel34))
for P in (4, 3, 8):
    s1, s2 = os.path.join(tmp, f"s{P}a.jsonl"), os.path.join(tmp, f"s{P}b.jsonl")
    got = T.build(SRC, s1, P, set_name="signal34")
    T.build(SRC, s2, P, set_name="signal34")
    gp = [r["metadata"]["pid"] for r in got]
    ep = [gp[e * 34:(e + 1) * 34] for e in range(30)]
    check(f"set signal34 P={P}: 30 epochs of 34 rows, each a permutation of the 34", len(gp) == 30 * 34
          and all(sorted(e, key=int) == sorted(SIG, key=int) for e in ep))
    check(f"set signal34 P={P}: every step {P} different problems",
          all(len(set(gp[i:i + P])) == P for i in range(0, len(gp) - P + 1, P)))
    check(f"set signal34 P={P}: deterministic (byte-identical on every build), the seeded order of the 34 rows",
          open(s1, "rb").read() == open(s2, "rb").read() and gp == [sel34[i]["metadata"]["pid"] for i in T.order(34, P, 67, 30)])
out = subprocess.run([sys.executable, os.path.join(ROOT, "rl", "train67_data.py"), SRC, os.path.join(tmp, "d.jsonl"), "4",
                      "--set", "signal34"], capture_output=True, text=True)
check("cli --set: runs and reports the set and its size", out.returncode == 0
      and "set signal34: 34 problems (112 kept" in out.stdout and "1020 rows" in out.stdout, out.stdout + out.stderr)
os.makedirs(os.path.join(ROOT, "data", "train67_sets"), exist_ok=True)
bad = os.path.join(ROOT, "data", "train67_sets", f"_test_bad_{os.getpid()}.txt")
open(bad, "w").write("# a pid that is not a training problem\n1\n0\n9999\n")
try:
    out = subprocess.run([sys.executable, os.path.join(ROOT, "rl", "train67_data.py"), SRC, os.path.join(tmp, "e.jsonl"),
                          "4", "--set", os.path.basename(bad)[:-4]], capture_output=True, text=True)
    check("set with an unknown pid: an error naming it, no file", out.returncode != 0 and "['0', '9999']" in out.stderr
          and not os.path.exists(os.path.join(tmp, "e.jsonl")), out.stdout + out.stderr)
finally:
    os.remove(bad)
for name in ("no_such_set", "../signal34", ""):
    try:
        T.load_set(name)
        raised = False
    except ValueError:
        raised = True
    check(f"set {name!r}: an error", raised)

# (b) the groups miles forms: its data source reads P rows per rollout in file order (no --rollout-shuffle) and gives
# each row's G copies one group_index; GRPO centers rewards within a group_index
try:
    from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
    from miles.rollout.data_source import RolloutDataSource
    from miles.utils.types import Sample

    P, G = 4, 8
    ds = RolloutDataSource.__new__(RolloutDataSource)
    ds.args = SimpleNamespace(n_samples_per_prompt=G, rollout_shuffle=False)
    ds.epoch_id = ds.sample_group_index = ds.sample_index = ds.sample_offset = 0
    ds.metadata = {}

    class Rows(list):   # what get_samples reads of miles' Dataset: len() and .samples
        @property
        def samples(self):
            return self

    ds.dataset = Rows(Sample(prompt=r["prompt"], label=r["label"], metadata=r["metadata"]) for r in built)
    groups0 = ds.get_samples(P)
    groups1 = ds.get_samples(P)
    check("groups: P groups of G per rollout", len(groups0) == P and all(len(g) == G for g in groups0))
    check("groups: one problem per group, the file's rows in order",
          [g[0].metadata["pid"] for g in groups0] == [r["metadata"]["pid"] for r in built[:4]]
          and [g[0].metadata["pid"] for g in groups1] == [r["metadata"]["pid"] for r in built[4:8]]
          and all(len({s.metadata["pid"] for s in g}) == 1 for g in groups0))
    check("groups: one group_index per group, distinct episode indices",
          [{s.group_index for s in g} for g in groups0] == [{0}, {1}, {2}, {3}]
          and len({s.index for g in groups0 for s in g}) == P * G)
    # GRPO centering: within each problem's group (std normalization off, as in the run)
    flat = [s for g in groups0 for s in g]
    for s in flat:
        s.rollout_id = s.index
    rew = [float(s.group_index * 10 + (s.index % G)) for s in flat]
    args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=False, n_samples_per_prompt=G,
                           rollout_batch_size=P)
    adv = _normalize_rewards_by_rollout(args, flat, rew, None)
    check("groups: advantages centered within each problem's group",
          all(abs(sum(adv[k * G:(k + 1) * G])) < 1e-5 for k in range(P))
          and abs(adv[0] - (0 - 3.5)) < 1e-5 and abs(adv[G] - (0 - 3.5)) < 1e-5, adv[:10])
except ImportError as e:
    check("groups: miles importable (run in miles-v0.1.3.sif)", False, e)

# (c) stop rules
rd = tempfile.mkdtemp()
open(os.path.join(rd, "train.log"), "w").write("")


def rolls(*recs):
    open(os.path.join(rd, "steps.jsonl"), "w").write("".join(json.dumps({"rollout_id": i, **r}) + "\n" for i, r in enumerate(recs)))


S32 = [0.3] * 32
os.environ["MA_RL_WALL_STOP_FRAC"] = "0.25"
os.environ["MA_RL_MAX_DROPPED_FRAC"] = "0.25"
rolls({"S": S32, "wall_exceeded": 9}, {"S": S32, "wall_exceeded": 9})
check("rules B=32: over 25% at the wall cap (9 of 32) in 2 rollouts stops", "wall cap" in (MA._stop_rule(rd, 1) or ""))
rolls({"S": S32, "wall_exceeded": 8}, {"S": S32, "wall_exceeded": 9})
check("rules B=32: 8 of 32 (exactly 25%) once is not enough", MA._stop_rule(rd, 1) is None)
rolls({"S": S32, "wall_exceeded": 9}, {"S": S32, "wall_exceeded": 2})
check("rules B=32: one rollout over is not enough", MA._stop_rule(rd, 1) is None)
rolls({"S": S32, "wall_exceeded": 9})
check("rules B=32: rollout 0 alone is not enough", MA._stop_rule(rd, 0) is None)
rolls({"S": S32, "wall_exceeded": 3}, {"S": S32, "wall_exceeded": 3})
check("rules B=32: 3 of 32 in each (run20's absolute rule would stop) goes on", MA._stop_rule(rd, 1) is None)
rolls({"S": S32, "dropped_episodes": 8, "failed": 1})
check("rules B=32: 1 group of 8 out of the loss (25%) goes on", MA._stop_rule(rd, 0) is None)
rolls({"S": S32, "dropped_episodes": 16, "failed": 2})
check("rules B=32: 2 groups out of the loss stop", "out of the loss" in (MA._stop_rule(rd, 0) or ""))
S16 = [0.3] * 16
one, ok = {"S": S16, "dropped_episodes": 8, "dropped_groups": 1, "failed": 2}, {"S": S16, "dropped_episodes": 0, "dropped_groups": 0}
rolls(ok, one)
check("rules P=2: 1 group of 2 out once goes on", MA._stop_rule(rd, 1) is None)
rolls(one)
check("rules P=2: 1 group of 2 out in the first rollout goes on", MA._stop_rule(rd, 0) is None)
rolls(one, ok, one)
check("rules P=2: out, in, out goes on", MA._stop_rule(rd, 2) is None)
rolls(one, one)
check("rules P=2: 1 group of 2 out in 2 rollouts in a row stops", "out of the loss" in (MA._stop_rule(rd, 1) or ""))
rolls(ok, {"S": S16, "dropped_episodes": 16, "dropped_groups": 2, "failed": 2})
check("rules P=2: both groups out stops at once", "out of the loss" in (MA._stop_rule(rd, 1) or ""))
rolls({"S": [0.0] * 32}, {"S": [0.0] * 32}, {"S": [0.0] * 32})
check("rules B=32: every episode at 0 in 3 rollouts stops", "S = 0" in (MA._stop_rule(rd, 2) or ""))
del os.environ["MA_RL_WALL_STOP_FRAC"], os.environ["MA_RL_MAX_DROPPED_FRAC"]
rolls({"S": [0.3] * 8, "wall_exceeded": 2, "dropped_episodes": 8}, {"S": [0.3] * 8, "wall_exceeded": 2})
check("rules run20 (fractions unset): more than 1 of 8 in 2 rollouts stops; dropped episodes no rule",
      "more than 1 episode" in (MA._stop_rule(rd, 1) or "") and MA._stop_rule(rd, 0) is None)

# (d) log_rollout's train67 record against a fake router, with the in-process check replaced
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
import threading  # noqa: E402


class Router(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        lps = [[None, body["input_ids"][0], None]] + [[-1.0, t, None] for t in body["input_ids"][1:]]
        data = json.dumps({"meta_info": {"input_token_logprobs": lps}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), Router)
threading.Thread(target=srv.serve_forever, daemon=True).start()
args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=srv.server_port)
out_dir = tempfile.mkdtemp()
os.environ.update({"MA_RL_OUT": out_dir, "MA_AG_TRACE_DIR": os.path.join(out_dir, "traces_live"), "MA_RL_P": "2",
                   "MA_RL_DUMP_EVERY": "5", "MA_RL_DATA": os.path.join(out_dir, "data_train67.jsonl"),
                   "MA_RL_WALL_STOP_FRAC": "0.25", "MA_RL_MAX_DROPPED_FRAC": "0.25", "MA_RL_MAX_SAMPLE_TOKENS": "0"})
open(os.environ["MA_RL_DATA"], "w").write("".join(json.dumps({"prompt": [{"role": "user", "content": p}],
                                                              "metadata": {"pid": p}}) + "\n" for p in ["7", "1", "9", "23"]))


class S:   # what log_rollout reads of a miles Sample, and what _dump_samples writes
    def __init__(self, sess, pid, g, reward, subs, failed=None, n=10):
        self.tokens, self.reward, self.group_index = list(range(n)), reward, g
        self.metadata = {"flint_session": sess, "pid": pid, "fo_lg_submissions": subs, **({"flint_failed": failed} if failed else {})}

    def to_dict(self):
        return {"tokens": self.tokens, "reward": self.reward, "metadata": self.metadata}


def batch(rid, fail=False):
    # P = 2 problems x G = 4 episodes; problem a: S 0.2/0.4/0.0/0.6 (3 of 4 submitted), problem b: all 0 (none)
    a, b = ("7", "1") if rid == 0 else ("9", "23")
    smp = [S(f"{rid}a{i}", a, 0, r, s) for i, (r, s) in enumerate([(0.2, 1), (0.4, 2), (0.0, 0), (0.6, 3)])]
    smp += [S(f"{rid}a0", a, 0, 0.2, 1, n=20)]   # a second leaf of episode a0
    smp += [S(f"{rid}b{i}", b, 1, 0.0, 0, failed="judge infra error" if fail and i == 0 else None) for i in range(4)]
    return smp


inspected, problems_next = [], [{}]


def fake_inspect(d, samples):
    inspected.append((d, len(samples)))
    failed = sum(bool(s.metadata.get("flint_failed")) for s in samples if s.metadata["flint_session"].endswith("0"))
    return {}, {"failed episodes": failed, "judge infra submissions": failed, "rewards not S": 0, **problems_next[0]}, ["table"]


MA._inspect = fake_inspect
dumps = []
real_dump = MA._dump_samples
MA._dump_samples = lambda d, s: dumps.append(d) or real_dump(d, s)
os.makedirs(os.environ["MA_AG_TRACE_DIR"])
MA.log_rollout(0, args, batch(0), {}, 30.0)
rec = json.loads(open(os.path.join(out_dir, "steps.jsonl")).readlines()[-1])
check("log: per-problem S means", rec["S_mean_by_problem"] == {"1": 0.0, "7": 0.3}, rec.get("S_mean_by_problem"))
check("log: fraction of episodes that submitted, overall and per problem",
      rec["submitted_frac"] == 3 / 8 and rec["submitted_frac_by_problem"] == {"1": 0.0, "7": 0.75}, rec.get("submitted_frac"))
check("log: problems seen and planned (the data file's rows 0-1)", rec["problems"] == ["1", "7"] and rec["planned"] == ["7", "1"])
check("log: 8 episodes, 9 samples, 2 groups, 1 with a learning signal",
      rec["episodes"] == 8 and rec["samples"] == 9 and rec["groups"] == 2 and rec["groups_with_signal"] == 1, rec)
check("log: the check runs in this process on the samples; rollout 0 dumped per episode",
      inspected == [(os.path.join(out_dir, "r0"), 9)] and rec["inspect"] == "INSPECT_OK" and rec.get("dumped")
      and sorted(os.listdir(os.path.join(out_dir, "r0", "samples"))) == [f"{i:03d}.pt" for i in range(8)])
os.makedirs(os.environ["MA_AG_TRACE_DIR"])
MA.log_rollout(1, args, batch(1, fail=True), {}, 30.0)
rec = json.loads(open(os.path.join(out_dir, "steps.jsonl")).readlines()[-1])
check("log: a failed episode takes its group out of the loss, the run goes on (1 of 2 groups once: no stop)",
      rec["dropped_groups"] == 1 and rec["dropped_episodes"] == 4 and rec["failed"] == 1
      and "out of the loss" in rec["inspect"] and not os.path.exists(os.path.join(out_dir, "stop_request")), rec)
check("log: rollout 1 not dumped (every 5th)", "dumped" not in rec and dumps == [os.path.join(out_dir, "r0")]
      and rec["planned"] == ["9", "23"])
problems_next[0] = {"top-k errors": 3}
os.makedirs(os.environ["MA_AG_TRACE_DIR"])
try:
    MA.log_rollout(2, args, batch(0), {}, 30.0)
    raised = False
except RuntimeError as e:
    raised = "sample check failed" in str(e)
check("log: any other sample problem ends the run", raised)
srv.shutdown()
done()
