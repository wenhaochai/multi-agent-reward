"""Train/val split of the Frontier-CS v1.1 classic problems (user 2026-10-21: "只用 FCS 1.1", train and val both from the
v1.1 classic, i.e. non-interactive, problems; FrontierSmith is no longer used).
Source: Frontier-CS aec2aff0 (v1.1 checkers and scoring), checked out at project/Frontier-CS-v1.1.
Rules:
  - pool: the 107 problems whose config.yaml type is "default" (interactive ones need an interactor the shell agent
    does not have);
  - always train: 306-315 (the 10 FrontierSmith parity problems, statements 70-85% identical to frontiersmith_{180,...};
    every checkpoint trained on FrontierSmith has seen them) and problems with fewer than 4 test cases (the long-horizon
    game splits cases into provisional and final halves);
  - val: 40 problems drawn from the rest with seed 20261021, stratified by the untrained SFT agent's mean score on the
    old snapshot (data_v11/sft_agent_ev4_by_pid.json, 4 episodes per problem): > 0, = 0, and no score (problems new
    in v1.1); each stratum gives val the share it has of the eligible pool, rounded so the total is 40;
  - train: the other 67.
Prompt: FrontierSmith's build_prompt with the last-block rule (as in data/fcs_val_classic_agent.jsonl), over the v1.1
statement. label = the v1.1 problem directory.
Outputs in data_v11/: fcs11_split.json, fcs11_train67.jsonl, fcs11_val40.jsonl (one-answer form),
fcs11_train67_agent.jsonl, fcs11_val40_agent.jsonl (metadata.fo_game = "agent"), fcs11_val40_agent_smoke8.jsonl."""
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "data_v11"
P11 = Path("/scratch/gpfs/GROUP/USER/project/Frontier-CS-v1.1/algorithmic/problems")
OLD_ROW = Path("/scratch/gpfs/GROUP/USER/project/miles-q38-build/data/fcs_val_classic_agent.jsonl")
OLD_P = Path("/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic/problems")
HEAD = ("You are a competitive programmer. Solve the following problem in C++. Output ONLY the C++ code wrapped in ```cpp "
        "and ```. No explanation. If you write more than one ```cpp block, only the last one is submitted.")
SEED, N_VAL = 20261021, 40


def prompt(statement):
    return [{"role": "user", "content": f"{HEAD}\n\n{statement}\n\nGenerate solution code:"}]


def ptype(d):
    m = re.search(r"^\s*type:\s*(\S+)", (d / "config.yaml").read_text(), re.M)
    return m.group(1) if m else "default"


# the prompt builder reproduces the rows the earlier runs used
old = json.loads(OLD_ROW.read_text().splitlines()[0])
assert prompt((OLD_P / old["metadata"]["pid"] / "statement.txt").read_text()) == old["prompt"], "prompt format drifted"

ids = sorted((d.name for d in P11.iterdir() if d.is_dir()), key=int)
classic = [i for i in ids if ptype(P11 / i) == "default"]
assert len(ids) == 180 and len(classic) == 107, (len(ids), len(classic))
ncase = {i: len(list((P11 / i / "testdata").glob("*.in"))) for i in classic}
dup = [str(i) for i in range(306, 316)]
few = [i for i in classic if ncase[i] < 4]
forced = sorted(set(dup) | set(few), key=int)
elig = [i for i in classic if i not in forced]
sft = json.loads((OUT / "sft_agent_ev4_by_pid.json").read_text())
strata = {"sft>0": [i for i in elig if i in sft and sft[i] > 0], "sft=0": [i for i in elig if i in sft and sft[i] == 0],
          "new": [i for i in elig if i not in sft]}
quota = {k: N_VAL * len(v) / len(elig) for k, v in strata.items()}
take = {k: int(q) for k, q in quota.items()}
for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[:N_VAL - sum(take.values())]:
    take[k] += 1
rng = random.Random(SEED)
val = sorted((i for k, v in strata.items() for i in rng.sample(sorted(v, key=int), take[k])), key=int)
train = sorted((i for i in classic if i not in val), key=int)
assert len(val) == N_VAL and len(train) == 107 - N_VAL and not set(val) & set(forced)

OUT.mkdir(exist_ok=True)
(OUT / "fcs11_split.json").write_text(json.dumps({
    "source": "Frontier-CS aec2aff0 (v1.1)", "seed": SEED, "val": val, "train": train, "forced_train": {"frontiersmith_dup": dup, "fewer_than_4_cases": few},
    "strata": {k: {"eligible": len(v), "val": take[k]} for k, v in strata.items()}}, indent=1) + "\n")


def rows(pids, agent):
    out = []
    for i in pids:
        pr = prompt((P11 / i / "statement.txt").read_text())
        md = {"pid": i, "messages": pr, **({"fo_game": "agent"} if agent else {})}
        out.append(json.dumps({"prompt": pr, "label": str(P11 / i), "metadata": md}, ensure_ascii=False) + "\n")
    return out


for name, pids in (("train67", train), ("val40", val)):
    (OUT / f"fcs11_{name}.jsonl").write_text("".join(rows(pids, False)))
    (OUT / f"fcs11_{name}_agent.jsonl").write_text("".join(rows(pids, True)))
(OUT / "fcs11_val40_agent_smoke8.jsonl").write_text("".join(rows(val[:8], True)))
print("val", len(val), "train", len(train), "forced train", len(forced), {k: (len(v), take[k]) for k, v in strata.items()})
print("val:", " ".join(val))
for name, pids in (("train", train), ("val", val)):
    s = [sft[i] for i in pids if i in sft]
    print(f"{name}: old SFT agent mean {sum(s) / len(s):.1f} over {len(s)} scored problems; cases median "
          f"{sorted(ncase[i] for i in pids)[len(pids) // 2]}")
