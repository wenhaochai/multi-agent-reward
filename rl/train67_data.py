"""The prompt file of the train67 RL runs (rl/rl_job.sh train67*): the 67 training problems of Frontier-CS v1.1
(data/fcs11_train67_agent.jsonl), or a named subset of them, one row per (epoch, problem), each epoch a permutation
drawn with a fixed seed. miles reads the rows in file order (no --rollout-shuffle), P per step, and saves its position
with every checkpoint, so the same file gives a resumed run the same problems for the same step. Problem 112's
official checker accepts a fake output (audit 2026-10-28); 112 stays only if our fixed checker
(flint/checkers/112/chk.cc) is in the tree. A set (data/train67_sets/<NAME>.txt: one pid per line, # comments) keeps
only its problems, in the source's order, before the permutations; a pid of the set that is not among the training
problems is an error. Usage: train67_data.py SRC OUT P [SEED] [EPOCHS] [--set NAME]; prints the first steps'
problems."""
from __future__ import annotations

import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = os.path.join(REPO, "data", "train67_sets")


def fixed_checker(pid: str) -> bool:
    return os.path.isfile(os.path.join(REPO, "flint", "checkers", pid, "chk.cc"))


def load_set(name: str) -> list[str]:
    """The pids of data/train67_sets/<name>.txt, in file order; a name with a path in it, an unknown name, an empty
    set or a pid twice is an error."""
    if not name or os.sep in name or name.startswith("."):
        raise ValueError(f"set name {name!r}: a file name in {SETS} without .txt")
    path = os.path.join(SETS, name + ".txt")
    if not os.path.isfile(path):
        raise ValueError(f"set {name!r}: no file {path}")
    pids = [x.strip() for x in open(path) if x.strip() and not x.lstrip().startswith("#")]
    if not pids or len(set(pids)) != len(pids):
        raise ValueError(f"set {name!r}: empty or a pid twice: {pids}")
    return pids


def select(rows: list[dict], keep: list[str] | None = None) -> list[dict]:
    """The training problems: all of them (or those in keep), but 112 without its fixed checker. The row's problem
    directory goes into the metadata too (miles hands the agent only the metadata)."""
    if keep is not None:
        unknown = sorted(set(keep) - {r["metadata"]["pid"] for r in rows}, key=lambda p: (len(p), p))
        if unknown:
            raise ValueError(f"set pids not among the training problems: {unknown}")
    out = []
    for r in rows:
        if keep is not None and r["metadata"]["pid"] not in keep:
            continue
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


def build(src: str, out: str, p: int, seed: int = 67, epochs: int = 30, set_name: str | None = None) -> list[dict]:
    rows = select([json.loads(x) for x in open(src)], load_set(set_name) if set_name else None)
    seq = order(len(rows), p, seed, epochs)
    with open(out + ".tmp", "w") as f:
        for i in seq:
            f.write(json.dumps(rows[i]) + "\n")
    os.replace(out + ".tmp", out)
    return [rows[i] for i in seq]


if __name__ == "__main__":
    argv, set_name = sys.argv[1:], None
    if "--set" in argv:
        k = argv.index("--set")
        if k + 1 >= len(argv):
            sys.exit("--set needs a NAME")
        set_name = argv[k + 1]
        del argv[k:k + 2]
    src, out, p = argv[0], argv[1], int(argv[2])
    seed = int(argv[3]) if len(argv) > 3 else 67
    epochs = int(argv[4]) if len(argv) > 4 else 30
    try:
        got = build(src, out, p, seed, epochs, set_name)
    except ValueError as e:
        sys.exit(f"[train67] {e}")
    pids = [r["metadata"]["pid"] for r in got]
    n = len({*pids})
    print(f"[train67] {f'set {set_name}: ' if set_name else ''}{n} problems "
          f"(112 {'kept: fixed checker' if '112' in pids else 'left out: no fixed checker'}), "
          f"{len(pids)} rows, seed {seed}, {p} per step; steps 0-2: {[pids[i * p:(i + 1) * p] for i in range(3)]}")
