"""Data for the Frontier-CS team game (miles_team/fcs_team.py) from the solo data (tools/make_fcs_data.py): same
prompts and labels, plus metadata.messages (the raw chat; sample.prompt arrives templated) and, for the solo val set,
metadata.ft_mode = "solo" (one agent on the solo prompt: the trained weights scored as a single agent).
  data/fcs_train200_team.jsonl, data/fcs_val172_team.jsonl, data/fcs_val172_solo.jsonl, and the first 16 val
  problems of each val set as data/fcs_val16_{team,solo}.jsonl for smoke runs"""
import json
import os

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for src, dst, mode, n in [("fcs_train200.jsonl", "fcs_train200_team.jsonl", None, None),
                          ("fcs_val172.jsonl", "fcs_val172_team.jsonl", None, None),
                          ("fcs_val172.jsonl", "fcs_val172_solo.jsonl", "solo", None),
                          ("fcs_val172.jsonl", "fcs_val16_team.jsonl", None, 16),
                          ("fcs_val172.jsonl", "fcs_val16_solo.jsonl", "solo", 16)]:
    rows = [json.loads(l) for l in open(f"{D}/{src}")][:n]
    with open(f"{D}/{dst}.tmp", "w") as f:
        for r in rows:
            md = {**(r.get("metadata") or {}), "messages": r["prompt"]}
            if mode:
                md["ft_mode"] = mode
            f.write(json.dumps({**r, "metadata": md}) + "\n")
    os.replace(f"{D}/{dst}.tmp", f"{D}/{dst}")
    print(dst, len(rows))
