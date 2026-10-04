"""Add the submission rule to every problem prompt in data/*.jsonl (user decision 2026-10-04): after FrontierSmith's
"Output ONLY the C++ code wrapped in ```cpp and ```. No explanation." comes "If you write more than one ```cpp
block, only the last one is submitted." Both the chat in `prompt` and the raw chat in metadata.messages are edited;
the originals are kept in data/pre_lastblock/. Idempotent."""
import json, shutil
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data"
OLD = "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation."
NEW = OLD + " If you write more than one ```cpp block, only the last one is submitted."
(D / "pre_lastblock").mkdir(exist_ok=True)


def fix(msgs):
    n = 0
    for m in msgs or []:
        if isinstance(m, dict) and isinstance(m.get("content"), str) and OLD in m["content"] and NEW not in m["content"]:
            m["content"] = m["content"].replace(OLD, NEW); n += 1
    return n


for f in sorted(D.glob("*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    n = sum(fix(r.get("prompt") if isinstance(r.get("prompt"), list) else None)
            + fix((r.get("metadata") or {}).get("messages")) for r in rows)
    if n:
        if not (D / "pre_lastblock" / f.name).exists():
            shutil.copy2(f, D / "pre_lastblock" / f.name)
        f.with_suffix(".tmp").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        f.with_suffix(".tmp").replace(f)
    print(f"{f.name}: {n} messages updated")
