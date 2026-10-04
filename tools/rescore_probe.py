"""Re-judge a probe's saved answers (runs/<probe>/gen.jsonl) with the current judge and code-block rule
(FCS_EXTRACT, default last) -> runs/<probe>/scores_<rule>.jsonl, written incrementally (a rerun skips done keys).
  python3 tools/rescore_probe.py <probe dir> [<probe dir> ...]"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miles_team.fcs_judge import CASE_WORKERS  # noqa: E402
from miles_team.fcs_rm import EXTRACT, extract_cpp, judge_full  # noqa: E402


def one(r):
    code = extract_cpp(r['response'])
    j = judge_full(r['label'], code)
    return {k: r[k] for k in ('key', 'split', 'pid', 'i')} | {'score': 100 * j['score'], 'status': j['status'],
                                                                'infra': j['infra_error'], 'rule': EXTRACT}


for d in map(Path, sys.argv[1:]):
    out = d / f'scores_{EXTRACT}.jsonl'
    done = {json.loads(l)['key'] for l in open(out)} if out.exists() else set()
    todo = [json.loads(l) for l in open(d / 'gen.jsonl')]
    todo = [r for r in todo if r['key'] not in done]
    print(f'{d.name}: {len(done)} done, {len(todo)} to judge (rule {EXTRACT})', flush=True)
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 8) // CASE_WORKERS)) as ex, open(out, 'a') as f:
        for k, fut in enumerate(as_completed([ex.submit(one, r) for r in todo])):
            f.write(json.dumps(fut.result()) + '\n'); f.flush()
            if k % 200 == 0:
                print(f'  {k}/{len(todo)}', flush=True)
