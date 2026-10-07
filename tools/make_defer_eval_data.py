"""An eval set whose samples are judged after training resumes (miles_team.fcs_rm: metadata fcs_defer_score = 1).
Same prompts, labels and order as the input; only the metadata gains the flag.
  python3 tools/make_defer_eval_data.py data/fcs_val172.jsonl data/fcs_val172_defer.jsonl"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as g:
    for line in f:
        r = json.loads(line)
        r["metadata"] = {**(r.get("metadata") or {}), "fcs_defer_score": 1}
        g.write(json.dumps(r) + "\n")
