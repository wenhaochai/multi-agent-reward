"""Code-block rule of miles_team/fcs_rm.py and the val official-score log (real judge, val problem 0).
  python3 tests/test_fcs_extract.py   (in miles.sif)"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import miles_team.fcs_rm as R  # noqa: E402
from miles_team.fcs_judge import FCS_ROOT  # noqa: E402

draft = "#include <cstdio>\nint main(){ int x = y; /* a long first draft that does not compile */ return 0; }\n" + "// pad\n" * 40
final = "#include <cstdio>\nint main(){ puts(\"0\"); }"
resp = f"thinking</think>\n```cpp\n{draft}```\nWait, fix it.\n```cpp\n{final}\n```"
assert R.EXTRACT == "last"
assert R.extract_cpp(resp) == final.strip() and R.extract_cpp(resp, "longest") == draft.strip()
assert R.extract_cpp("x</think>```cpp\nint main(){}\n```") == "int main(){}"
print("[ok] last block by default, longest on request")
d = tempfile.mkdtemp()
args = SimpleNamespace(save=d + "/ckpt")
val = SimpleNamespace(response=resp, label=str(FCS_ROOT / "problems" / "0"), status="", response_length=1, metadata={})
r = asyncio.run(R.fcs_rm(args, val))
rows = [json.loads(l) for l in open(d + "/val_official.jsonl")]
assert len(rows) == 1 and rows[0]["reward"] == r and rows[0]["official"] == 0.0 and not rows[0]["same_code"], rows
print(f"[ok] val official log: reward {r:.3f} (last block), official {rows[0]['official']} (longest = the draft)")
print("ALL OK")
