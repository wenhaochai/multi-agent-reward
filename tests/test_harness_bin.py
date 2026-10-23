"""The long-horizon game's stub `submit` and `post` on PATH (fcs_cmd.HARNESS_BIN): `which submit` finds them inside the
sandbox, running one prints how to call it, and a command without harness=True does not see them."""
import os
import sys
import tempfile

os.environ.setdefault("FCS_CASE_PROC", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from miles_team import fcs_cmd as C  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, detail if not cond else "")


with tempfile.TemporaryDirectory() as work:
    out = C.run_command("which submit post", work, True)
    check("which finds both stubs", out["returncode"] == 0 and out["output"].count(C.HARNESS_BIN) == 2, out)
    out = C.run_command("submit 2>&1 | head -3; echo rc=${PIPESTATUS[0]}", work, True)
    check("submit in a pipeline says how to call it", "whole command" in out["output"] and "rc=2" in out["output"], out)
    out = C.run_command("cat $(which submit)", work, True)
    check("cat $(which submit) is readable", "run by the harness" in out["output"], out)
    out = C.run_command("echo x > f.txt && cat f.txt", work, True)
    check("work dir still writable", out["output"].strip() == "x", out)
    out = C.run_command("touch " + C.HARNESS_BIN + "/z 2>&1; echo rc=$?", work, True)
    check("stub dir not writable", "rc=0" not in out["output"] or C.J.LANDLOCK_ABI == 0, out)
    out = C.run_command("which submit; echo rc=$?", work)
    check("no stubs without harness", "rc=1" in out["output"], out)
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
