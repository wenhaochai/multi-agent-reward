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
# judge_code keeps its concurrency slot until the judge thread ends, also when the caller is cancelled
import time  # noqa: E402
R._SEM = None
os.environ["FCS_RM_CONCURRENCY"] = "1"
_real = R.judge_full
R.judge_full = lambda d, c: (time.sleep(0.6), {"score": 0.0})[1]


async def _race():
    t0 = time.monotonic()
    a = asyncio.ensure_future(R.judge_code("x", "a"))
    await asyncio.sleep(0.1)
    a.cancel()
    try:
        await a
    except asyncio.CancelledError:
        pass
    await R.judge_code("x", "b")  # must wait for the first thread: 2 x 0.6 s in all
    return time.monotonic() - t0


took = asyncio.run(_race())
R.judge_full = _real
assert took >= 1.15, took
print(f"[ok] a cancelled judge keeps its slot until its thread ends ({took:.2f} s)")
print("ALL OK")
