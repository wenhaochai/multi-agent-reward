"""Rewrite the q38 jsonl files for the team rollout: keep prompt/label and add metadata.messages (the raw chat) and
metadata.datasource, which miles_team.team_rollout reads (sample.prompt arrives already templated).
usage: python tools/make_team_data.py   (in place, under data/)"""
import glob, json, os
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for p in sorted(glob.glob(f"{D}/q38mid_train.jsonl") + glob.glob(f"{D}/eval_*.jsonl")):
    rows = [json.loads(l) for l in open(p)]
    with open(p + ".tmp", "w") as f:
        for r in rows:
            r["metadata"] = {"messages": r["prompt"], "datasource": r.get("datasource", "")}
            f.write(json.dumps(r) + "\n")
    os.replace(p + ".tmp", p)
    print(p, len(rows))
