"""fcs_cmd.run_command's limits on what a command leaves behind (audit 2026-10-24): a background child that keeps the
output open cannot hold the call past MA_AG_TIMEOUT, every process the command started is gone when the call returns,
and a command cannot reach an abstract unix socket made outside its sandbox. Run it twice, with FCS_CASE_PROC=1 (case
workers, as in the evals) and FCS_CASE_PROC=0 (in the caller); the worktree goes first on sys.path, so the case
workers import this code."""
import os
import socket
import sys
import tempfile
import time
import uuid

os.environ.setdefault("MA_AG_TIMEOUT", "5")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from miles_team import fcs_cmd as C  # noqa: E402

T = C.TIMEOUT
WORKER = C.J.CASE_PROC
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("OK  " if cond else "FAIL"), name, "" if cond else detail)


def alive(tag: str) -> list:
    """pids whose command line holds `tag` (a sleep length no other process uses)."""
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/cmdline", "rb") as f:
                    if tag.encode() in f.read().replace(b"\0", b" "):
                        out.append(int(d))
            except OSError:
                pass
    return out


print("case workers" if WORKER else "in the caller", "| landlock ABI", C.J.LANDLOCK_ABI, "| timeout", T)
base = os.environ.get("TMPDIR") or None
with tempfile.TemporaryDirectory(dir=base) as work:
    r = C.run_command("echo a; echo b >&2; echo c", work)
    check("stdout and stderr stay in order", r["output"] == "a\nb\nc\n" and r["returncode"] == 0, r)
    r = C.run_command("exit 3", work)
    check("return code kept", r["returncode"] == 3 and r["exception_info"] == "", r)
    r = C.run_command("cat /dev/urandom | head -c 4 | wc -c", work)
    check("/dev/urandom readable", r["output"].strip() == "4", r)
    if C.J.LANDLOCK_ABI:   # what fcs_agent_long's v2 prompt says about /tmp
        r = C.run_command("ls /tmp; touch /tmp/fcs_probe_$$ && echo WROTE", work)
        check("/tmp can be neither listed nor written", "Permission denied" in r["output"] and "WROTE" not in r["output"], r)
    t = time.monotonic()
    r = C.run_command("sleep 100", work)
    dt = time.monotonic() - t
    check("a long command still times out", r["returncode"] == -1 and "timed out" in r["exception_info"]
          and T - 0.5 < dt < T + 3, (dt, r))
    r = C.run_command("(sleep 30; touch late_file) & echo started", work)
    time.sleep(1)
    check("background child in the group killed", "started" in r["output"]
          and not os.path.exists(os.path.join(work, "late_file")), r)
    cases = [("setsid sleep 25.1 &", "25.1", ""),
             ("(setsid sleep 25.2 &); echo started", "25.2", "started"),
             ("timeout 25.3 sleep 25.3; echo after", "25.3", None),
             ("(timeout 25.4 sleep 25.4 >/dev/null &); echo bg", "25.4", "bg"),
             ("nohup timeout 26.5 sleep 26.5 > log 2>&1 & echo x", "26.5", "x")]
    if WORKER:   # an emptied environment escapes the tag, not the subreaper
        cases.append(("(env -i setsid sleep 25.6 < /dev/null &); echo e", "25.6", "e"))
    for cmd, tag, want in cases:
        t = time.monotonic()
        r = C.run_command(cmd, work)
        dt = time.monotonic() - t
        time.sleep(0.3)
        left = alive(f"sleep {tag}")
        timed_out = want is None
        check(f"{cmd!r}: returns in {dt:.1f} s, nothing left", dt < T + 6 and (dt < 3 or timed_out) and not left
              and (want is None or want in r["output"]) and (r["returncode"] == -1) == timed_out, (dt, left, r))
        for p in left:
            try:
                os.kill(p, 9)
            except ProcessLookupError:
                pass
    if C.J.LANDLOCK_ABI >= 6:
        name = "\0fcs_test_" + uuid.uuid4().hex
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(name)
        srv.listen(1)
        code = ("import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
                f"try:\n    s.connect({name!r}); print('CONNECTED')\nexcept OSError as e:\n    print('REFUSED', e.errno)")
        r = C.run_command(f"python3 -c \"{code}\"", work)
        check("no connect to an abstract socket made outside the sandbox", "REFUSED" in r["output"], r)
        r = C.run_command("python3 -c \"import socket; a, b = socket.socketpair(); a.send(b'x'); print(b.recv(1))\"", work)
        check("a command's own sockets still work", "b'x'" in r["output"], r)
        srv.close()
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
