"""The prompt file of the train67 RL runs (rl/rl_job.sh train67*): the 67 training problems of Frontier-CS v1.1
(data/fcs11_train67_agent.jsonl), one row per (epoch, problem), each epoch a permutation drawn with a fixed seed.
miles reads the rows in file order (no --rollout-shuffle), P per step, and saves its position with every checkpoint,
so the same file gives a resumed run the same problems for the same step. Problem 112's official checker accepts a
fake output (audit 2026-10-28); 112 stays only if our fixed checker (flint/checkers/112/chk.cc) is in the tree.
Usage: train67_data.py SRC OUT P [SEED] [EPOCHS]; prints the first steps' problems."""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fixed_checker(pid: str) -> bool:
    return os.path.isfile(os.path.join(REPO, "flint", "checkers", pid, "chk.cc"))


def select(rows: list[dict]) -> list[dict]:
    """The training problems: all of them, but 112 without its fixed checker. The row's problem directory goes into
    the metadata too (miles hands the agent only the metadata)."""
    out = []
    for r in rows:
        if r["metadata"]["pid"] == "112" and not fixed_checker("112"):
            continue
        r = json.loads(json.dumps(r))
        r["metadata"]["label"] = r["label"]
        out.append(r)
    return out


def order(n: int, p: int, seed: int, epochs: int) -> list[int]:
    """Row indices for `epochs` epochs: epoch e is a permutation of range(n) drawn with random.Random(seed + e). A step
    (p consecutive rows) that crosses an epoch end would hold a problem twice when the next epoch starts with one
    the previous ended with: such a row swaps with the first later row of its own epoch that is not in the step, so
    every epoch stays a permutation and every step holds p different problems."""
    assert 1 <= p <= n
    seq = []
    for e in range(epochs):
        perm = list(range(n))
        random.Random(seed + e).shuffle(perm)
        seq += perm
    for start in range(0, len(seq) - p + 1, p):
        for i in range(start, start + p):
            if seq[i] in seq[start:i]:
                ep_end = (i // n + 1) * n
                j = next(j for j in range(start + p, ep_end) if seq[j] not in seq[start:start + p])
                seq[i], seq[j] = seq[j], seq[i]
    return seq


def build(src: str, out: str, p: int, seed: int = 67, epochs: int = 30) -> list[dict]:
    rows = select([json.loads(x) for x in open(src)])
    seq = order(len(rows), p, seed, epochs)
    with open(out + ".tmp", "w") as f:
        for i in seq:
            f.write(json.dumps(rows[i]) + "\n")
    os.replace(out + ".tmp", out)
    return [rows[i] for i in seq]


if __name__ == "__main__":
    src, out, p = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 67
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    got = build(src, out, p, seed, epochs)
    pids = [r["metadata"]["pid"] for r in got]
    n = len({*pids})
    print(f"[train67] {n} problems (112 {'kept: fixed checker' if '112' in pids else 'left out: no fixed checker'}), "
          f"{len(pids)} rows, seed {seed}, {p} per step; steps 0-2: {[pids[i * p:(i + 1) * p] for i in range(3)]}")
