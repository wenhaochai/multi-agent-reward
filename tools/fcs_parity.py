"""Parity of miles_team/fcs_judge.py against the official Frontier-CS judge: re-judge a stratified sample of the
official batch results (Frontier-CS-Result/batch/algorithmic/results.csv) whose solution file and problem directory
hash-match the local copies, and report |local - official| per solution.
usage: python fcs_parity.py OUT.jsonl [N_PER_BIN] [PROCS]"""
import csv, json, random, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
from typing import Optional, Set


# hash_file / hash_directory: verbatim from Frontier-CS/src/frontier_cs/batch/state.py (the hashes in results.csv)
def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]  # Use first 16 chars for brevity


# Exclude compiled artifacts and temporary files from hash computation
EXCLUDE_EXTENSIONS = {".exe", ".o", ".obj", ".so", ".dll", ".pyc", ".pyo", ".class", ".a", ".lib"}


def hash_directory(path: Path, exclude_extensions: Optional[Set[str]] = None) -> str:
    """
    Compute hash of all relevant files in a directory.

    Args:
        path: Directory path
        exclude_extensions: File extensions to exclude (default: compiled artifacts)

    Returns:
        Combined hash of all file contents and paths
    """
    if exclude_extensions is None:
        exclude_extensions = EXCLUDE_EXTENSIONS

    h = hashlib.sha256()
    files = []

    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        # Skip hidden files and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            continue
        # Exclude compiled artifacts
        if p.suffix in exclude_extensions:
            continue
        files.append(p)

    for p in files:
        # Include relative path in hash (so renames are detected)
        rel_path = p.relative_to(path)
        h.update(str(rel_path).encode("utf-8"))
        h.update(b"\x00")
        # Include file content
        try:
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except (IOError, OSError):
            pass
        h.update(b"\x00")

    return h.hexdigest()[:16]


from miles_team.fcs_judge import judge, load_problem

ALG = Path('/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic')
RES = Path('/scratch/gpfs/GROUP/USER/project/Frontier-CS-Result/batch/algorithmic/results.csv')
VAL = set(open('/scratch/gpfs/GROUP/USER/project/frontiersmith-200/data/frontiercs_val_172_ids.txt').read().split())


def run(row):
    t = time.time()
    r = judge(ALG / 'problems' / row['problem'], open(ALG / 'solutions' / row['solution']).read(), case_workers=4)
    return {**row, 'local': r['score'], 'local_status': r['status'], 'sec': round(time.time() - t, 1)}


def main():
    csv.field_size_limit(1 << 30)
    out, per_bin, procs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25, int(sys.argv[3]) if len(sys.argv) > 3 else 8
    phash, inter = {}, {}
    rows = [r for r in csv.DictReader(open(RES)) if r['status'] == 'success' and r['problem'] in VAL]
    for p in {r['problem'] for r in rows}:
        phash[p] = hash_directory(ALG / 'problems' / p)
        inter[p] = load_problem(ALG / 'problems' / p)['interactive']
    ok = [r for r in rows if phash[r['problem']] == r['problem_hash']
          and (ALG / 'solutions' / r['solution']).exists() and hash_file(ALG / 'solutions' / r['solution']) == r['solution_hash']]
    print(f'{len(rows)} official rows on the 172; {len(ok)} hash-match locally; problems matched {len({r["problem"] for r in ok})}', flush=True)
    bins = defaultdict(list)
    for r in ok:
        s = float(r['score'])
        bins[(inter[r['problem']], 'zero' if s == 0 else 'full' if s >= 100 else 'partial')].append(r)
    rng = random.Random(0)
    sample = []
    for k in sorted(bins):
        rng.shuffle(bins[k])
        sample += [{'solution': r['solution'], 'problem': r['problem'], 'official': float(r['score']),
                    'interactive': k[0], 'bin': k[1]} for r in bins[k][:per_bin]]
        print(k, len(bins[k]), flush=True)
    with ProcessPoolExecutor(procs) as ex, open(out, 'w') as f:
        for res in ex.map(run, sample):
            f.write(json.dumps(res) + '\n'); f.flush()
    res = [json.loads(l) for l in open(out)]
    agg = defaultdict(list)
    for r in res:
        agg[(r['interactive'], r['bin'])].append(abs(r['local'] - r['official']))
    for k, d in sorted(agg.items()):
        print(k, f'n={len(d)} mean|diff|={sum(d)/len(d):.2f} exact(<0.5)={sum(x < 0.5 for x in d)}/{len(d)}')
    d = [abs(r['local'] - r['official']) for r in res]
    print(f'ALL n={len(d)} mean|diff|={sum(d)/len(d):.2f} exact={sum(x < 0.5 for x in d)}/{len(d)}')


if __name__ == '__main__':
    main()
