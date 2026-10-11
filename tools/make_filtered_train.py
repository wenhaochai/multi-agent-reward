"""Training data for NEW runs without the FrontierSmith problems whose checkers do not measure the algorithm
(owner 2026-10-11, after the 2026-10-11 audit; runs in flight keep data/fcs_train200*.jsonl):
  * full marks for no work: 183 (every .ans is empty, so R = G and any feasible output scores 1.0);
  * full marks at the statement's baseline (ratio = min(1, B/C) or 1.0 when O >= B): 18, 52, 58, 100, 106, 108, 125;
  * zero for correct programs: 5, 185 (ouf.readEof() rejects the trailing newline every answer prints).

  python3 tools/make_filtered_train.py [DATA_DIR]   # fcs_train200{,_team}.jsonl -> fcs_train190{,_team}.jsonl
"""
import json
import re
import sys
from pathlib import Path

DROP = {5, 18, 52, 58, 100, 106, 108, 125, 183, 185}
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"


def problem_id(row) -> int:
    m = re.search(r"frontiersmith_(\d+)$", str(row["label"]).rstrip("/"))
    assert m, row["label"]
    return int(m.group(1))


def main():
    for suffix in ("", "_team"):
        src, dst = DATA / f"fcs_train200{suffix}.jsonl", DATA / f"fcs_train190{suffix}.jsonl"
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        ids = [problem_id(r) for r in rows]
        assert len(rows) == 200 and len(set(ids)) == 200 and DROP <= set(ids), (src, len(rows))
        kept = [r for r, i in zip(rows, ids) if i not in DROP]
        dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        print(f"{dst.name}: {len(kept)} of {len(rows)} problems (dropped {sorted(DROP)})")


if __name__ == "__main__":
    main()
