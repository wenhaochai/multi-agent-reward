"""The harness's git (fcs_cmd.run_git) and RLIMIT_NPROC. The limit counts every task of the user on the node, and
fcs_judge._nproc_limit() is that count as of up to 5 s ago plus 128; when the user's other work on the node starts
more than 128 tasks inside that window, a git run under that limit cannot fork upload-pack and exits 128 (own-repos
fetches and publishes in smokes w260 and w412). A cached count far below the real one stands in for that: git under
the plain limit fails so, and the harness's git, with fcs_cmd.GIT_NPROC_MARGIN more room, still fetches and publishes."""
import os
import resource
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("FCS_CASE_PROC", "0")
os.environ["MA_LG_N"], os.environ["MA_LG_REPOS"] = "4", "own"
os.environ["TMPDIR"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_agent_long as L  # noqa: E402
from miles_team import fcs_judge as J  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


def real_tasks() -> int:
    return sum(len(os.listdir(f"/proc/{p}/task")) for p in os.listdir("/proc") if p.isdigit()
               and _own(p))


def _own(p):
    try:
        return os.stat(f"/proc/{p}").st_uid == os.getuid()
    except OSError:
        return False


work = tempfile.mkdtemp()
L.own_init(work, [f"agent{k}" for k in range(1, 5)])
d, team = os.path.join(work, "agent2"), os.path.join(work, "team.git")
n = real_tasks()
J._ucount[0], J._ucount[1] = time.monotonic(), n - 60 - J.NPROC_MARGIN   # a limit 60 below; cached: no rescan for 5 s
low = J._nproc_limit()


def plain():   # a git fetch under the limit commands and the judge get
    resource.setrlimit(resource.RLIMIT_NPROC, (low, low))


r = subprocess.run(["git", "-C", d, "fetch", "-q", team, "+refs/heads/*:refs/remotes/team/*"], preexec_fn=plain,
                   capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": work})
check("under the plain limit, git cannot fork and exits 128", r.returncode == 128 and "fork" in r.stderr,
      (low, n, r.returncode, r.stderr[:200]))
r, p = L.fetch_team(work, d), L.publish(work, "agent2")
check("the harness's git fetches and publishes with its margin", r.returncode == 0 and p.returncode == 0
      and J._nproc_limit() == low, (r.returncode, r.stderr[:200], p.returncode, p.stderr[:200]))
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
