"""Training data v2 after FrontierCS/FrontierSmith PR #4 (merged 2026-10-17, local frontiersmith-200 ac3046c): the
checkers of the 10 problems that make_filtered_train.py drops are fixed (183 gets reference answers; 18, 52, 58, 100,
106, 108, 125 score improvement over the baseline; 5 and 185 accept the trailing newline), and the parquet's prompt
rows of the 8 problems whose statements changed are updated. Writes data/fcs_train200v2.jsonl and
fcs_train200v2_team.jsonl: the rows of fcs_train200{,_team}.jsonl with prompt (and metadata.messages) taken from the
updated parquet, row order and labels unchanged. Asserts that exactly the expected rows changed. The v1 files stay
for the runs that used them.
  python3 tools/make_fcs_train_v2.py [DATA_DIR]   (needs pandas: in miles.sif)"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

FS = Path('/scratch/gpfs/GROUP/USER/project/frontiersmith-200')
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'data'
CHANGED = {18, 52, 58, 100, 106, 108, 125, 183}   # PR #4: "exactly eight training prompt rows are synchronized"

OLD = "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation."
NEW = OLD + " If you write more than one ```cpp block, only the last one is submitted."   # tools/add_last_block_rule.py


def chat(p):
    """A parquet prompt as our data files hold it: plain role/content dicts, with the last-block rule (2026-10-04)."""
    msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in p]
    for m in msgs:
        if OLD in m["content"] and NEW not in m["content"]:
            m["content"] = m["content"].replace(OLD, NEW)
    return msgs


df = pd.read_parquet(FS / 'data/frontiercs/train_synthetic_200.parquet')
prompts = [chat(p) for p in df['prompt']]
for suffix in ('', '_team'):
    rows = [json.loads(l) for l in (DATA / f'fcs_train200{suffix}.jsonl').read_text().splitlines() if l.strip()]
    assert len(rows) == len(prompts) == 200
    out, diff = [], set()
    for r, p in zip(rows, prompts):
        pid = int(re.search(r'frontiersmith_(\d+)$', r['label'].rstrip('/')).group(1))
        new = p
        if new != r['prompt']:
            diff.add(pid)
        r = {**r, 'prompt': new}
        if 'messages' in (r.get('metadata') or {}):
            r['metadata'] = {**r['metadata'], 'messages': new}
        out.append(r)
    assert diff == CHANGED, f'changed rows {sorted(diff)} != {sorted(CHANGED)}'
    (DATA / f'fcs_train200v2{suffix}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out))
    print(f'fcs_train200v2{suffix}.jsonl: 200 rows, prompts changed for {sorted(diff)}')
