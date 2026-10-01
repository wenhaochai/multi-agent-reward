"""miles prompt data for the Frontier-CS study, aligned with EasyPPO / FrontierSmith:
  data/fcs_train200.jsonl: the 200 FrontierSmith synthetic problems, prompts verbatim from
    frontiersmith-200/data/frontiercs/train_synthetic_200.parquet (row order kept);
  data/fcs_val172.jsonl: the 172 numeric-ID Frontier-CS algorithmic problems of the 2026-02-23 snapshot (55c104b, the
    FrontierSmith full.parquet set; ids in frontiersmith-200/data/frontiercs_val_172_ids.txt), prompt from
    FrontierSmith's prepare_frontiercs_parquet.build_prompt over the local (latest, checker-fixed) statement.txt.
label = the absolute problem directory the judge runs against."""
import json
from pathlib import Path

import pandas as pd

FS = Path('/scratch/gpfs/GROUP/USER/project/frontiersmith-200')
FCS = Path('/scratch/gpfs/GROUP/USER/project/Frontier-CS/algorithmic/problems')
OUT = Path(__file__).resolve().parents[1] / 'data'


def build_prompt(statement: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"""You are a competitive programmer. Solve the following problem in C++. Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.

{statement}

Generate solution code:""",
        }
    ]


def main():
    df = pd.read_parquet(FS / 'data/frontiercs/train_synthetic_200.parquet')
    with open(OUT / 'fcs_train200.jsonl', 'w') as f:
        for _, r in df.iterrows():
            pid = r['reward_model']['ground_truth']
            pdir = FS / 'Frontier-CS/algorithmic/problems' / pid
            assert (pdir / 'config.yaml').exists(), pdir
            f.write(json.dumps({'prompt': [dict(m) for m in r['prompt']], 'label': str(pdir), 'metadata': {'pid': pid}}) + '\n')
    ids = open(FS / 'data/frontiercs_val_172_ids.txt').read().split()
    assert len(ids) == 172
    with open(OUT / 'fcs_val172.jsonl', 'w') as f:
        for pid in ids:
            f.write(json.dumps({'prompt': build_prompt((FCS / pid / 'statement.txt').read_text(encoding='utf-8')),
                                'label': str(FCS / pid), 'metadata': {'pid': pid}}) + '\n')
    print(len(df), len(ids))


if __name__ == '__main__':
    main()
