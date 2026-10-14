"""Eval sets for the agentic game (miles_team.fcs_agent): the classic problems of the validation set (interactive ones
need an interactor the agent does not have, and the live judge scores them 0), metadata.fo_game = "agent".
  python3 tools/make_agent_data.py [DATA_DIR]   # fcs_val172_fo_par.jsonl -> fcs_val_classic_agent.jsonl (+ _smoke8)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from miles_team.fcs_judge import load_problem  # noqa: E402

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/gpfs/GROUP/USER/project/miles-q38-build/data")
rows = [json.loads(l) for l in (DATA / "fcs_val172_fo_par.jsonl").read_text().splitlines() if l.strip()]
out = []
for r in rows:
    if load_problem(Path(r["label"]))["interactive"]:
        continue
    r["metadata"] = {**r["metadata"], "fo_game": "agent"}
    out.append(r)
(DATA / "fcs_val_classic_agent.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
(DATA / "fcs_val_classic_agent_smoke8.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[:8]))
print(f"{len(out)} classic of {len(rows)} problems -> fcs_val_classic_agent.jsonl; first 8 -> _smoke8")
