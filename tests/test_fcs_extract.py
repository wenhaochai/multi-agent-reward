"""Code-block rule of miles_team/fcs_rm.py.
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
print("ALL OK")
