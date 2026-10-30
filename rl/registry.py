"""The file registry that pairs an RL trainer job with engine jobs on other nodes (stage 0 of the elastic design,
tmp/elastic_rl/design.md). Standard library only: it runs on the host (python3.9) and in the container.

Layout of a run directory RUN:
  RUN/engines/<job>.json  one per engine job: {"job", "node", "state", "engines": [{"url", "tp", "gpus"}], "log",
                          "start", "registered", "heartbeat"}; state "loading" from the job's start (so a waiting
                          trainer knows an engine is coming), "serving" only after every engine of the job answered
                          /health_generate (only serving engines are given to miles); the heartbeat is rewritten
                          every 30 s; the file is removed when the job ends.
  RUN/trainer.json        the trainer job: {"job", "node", "ip", "start", "heartbeat"}, rewritten every 30 s.
  RUN/STOP                the run is over: an engine job that sees it exits 0; a job that finds it at its start
                          exits 3 (delete it to reuse the directory). The trainer writes it when it ends; in pair mode
                          an engine that never saw a trainer writes it too, so that a trainer starting later fails at
                          once instead of waiting.
Every write goes to a temporary file in the same directory and is renamed over the target, so a reader never sees a
partial file (rename is atomic on GPFS within a directory). Times are Unix seconds from the writer's clock.

CLI (exit 0 unless stated):
  register RUN JOB NODE URL,TP,GPUS [...] [--state loading|serving] [--log PATH]
                                                       write RUN/engines/JOB.json
  beat FILE                                            rewrite FILE's heartbeat (exit 1 if FILE is gone)
  deregister RUN JOB                                   remove RUN/engines/JOB.json
  live RUN [--max-age S]                               print the live engines as JSON lines
  wait RUN [--min N] [--timeout S] [--load-timeout S] wait for >= N serving engines of one TP; print
                                                       "ADDRS<TAB>TP<TAB>NGPUS<TAB>JOBS". Exit 2 when no engine
                                                       record (loading or serving) is fresh after --timeout, or none
                                                       serves --load-timeout after the first one was seen; 3 on STOP
  trainer-beat RUN JOB NODE IP START                   write RUN/trainer.json
  engine-check RUN SINCE [--stale S] [--grace S] [--seen FILE]
                                                       what an engine job should do now: prints "serve", "stop"
                                                       (RUN/STOP exists), "gone" (a trainer heartbeat was seen and is
                                                       now older than --stale) or "alone" (no trainer heartbeat seen
                                                       --grace seconds after SINCE, the engine job's start); --seen
                                                       FILE remembers that a fresh heartbeat was seen
  trainer-check RUN JOBS... [--max-age S]              exit 0 when every engine job named is live and serving
"""
import argparse
import json
import os
import sys
import time

FRESH = 90                # a heartbeat older than this is not fresh (3 missed 30-s beats): a dead engine, no trainer
TRAINER_STALE = 900       # an engine job leaves when the trainer's heartbeat, once seen, is older than 15 min
TRAINER_GRACE = 1200      # ... or when it saw no trainer heartbeat 20 min after its own start
ENGINE_MAX_AGE = FRESH


def _write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):   # gone, or (never, given the rename) half written
        return None


def engine_file(run, job):
    return os.path.join(run, "engines", f"{job}.json")


def parse_engine(spec):
    """URL,TP,GPUS -> {"url", "tp", "gpus"}; GPUS is a ':'-separated list of the node's device ids."""
    url, tp, gpus = spec.split(",")
    url = url if "://" in url else f"http://{url}"
    host_port = url.split("://", 1)[1].rstrip("/")
    assert ":" in host_port and not host_port.startswith("127.") and not host_port.startswith("localhost"), \
        f"an engine must register a routable host:port, got {url}"
    g = [int(x) for x in gpus.split(":") if x != ""]
    assert int(tp) == len(g), f"TP {tp} but {len(g)} GPUs in {spec}"
    return {"url": url.rstrip("/"), "tp": int(tp), "gpus": g}


def register(run, job, node, engines, state="serving", log=None, now=None):
    assert state in ("loading", "serving"), state
    now = time.time() if now is None else now
    os.makedirs(os.path.join(run, "engines"), exist_ok=True)
    rec = {"job": job, "node": node, "state": state, "engines": engines, "log": log, "start": now, "registered": now,
           "heartbeat": now}
    old = _read_json(engine_file(run, job))
    if old:   # a re-registration of the same job (loading -> serving) keeps its first start time
        rec["start"] = old.get("start", now)
    _write_json(engine_file(run, job), rec)
    return rec


def beat(path, now=None):
    rec = _read_json(path)
    if rec is None:
        return False
    rec["heartbeat"] = time.time() if now is None else now
    _write_json(path, rec)
    return True


def deregister(run, job):
    try:
        os.remove(engine_file(run, job))
    except FileNotFoundError:
        pass


def records(run, max_age=ENGINE_MAX_AGE, now=None, states=("serving",)):
    """The engine files with a fresh heartbeat and one of ``states``, in a fixed order (registration time, then job).
    A directory that cannot be listed gives none."""
    now = time.time() if now is None else now
    d = os.path.join(run, "engines")
    try:
        names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        return []
    recs = [r for r in (_read_json(os.path.join(d, n)) for n in names) if r]
    recs = [r for r in recs if now - float(r.get("heartbeat", 0)) <= max_age and r.get("state", "serving") in states
            and not r.get("leaving")]
    recs.sort(key=lambda r: (float(r.get("registered", 0)), str(r.get("job"))))
    return recs


def live_engines(run, max_age=ENGINE_MAX_AGE, now=None):
    """The serving engines of every fresh engine file, each with its job added."""
    return [dict(e, job=r["job"]) for r in records(run, max_age, now) for e in r.get("engines", [])]


def pick(engines, n_min):
    """>= n_min engines of one TP size (miles asserts one size per engine group): the TP size with the most engines,
    ties to the larger TP. Returns (addrs, tp, total GPUs) or None."""
    by_tp = {}
    for e in engines:
        by_tp.setdefault(e["tp"], []).append(e)
    if not by_tp:
        return None
    tp, es = max(by_tp.items(), key=lambda kv: (len(kv[1]), kv[0]))
    if len(es) < n_min:
        return None
    urls, jobs = [], []
    for e in es:
        if e["url"] not in urls:
            urls.append(e["url"])
        if e["job"] not in jobs:
            jobs.append(e["job"])
    return [u.split("://", 1)[1] for u in urls], tp, tp * len(urls), jobs


def trainer_beat(run, job, node, ip, start, now=None):
    now = time.time() if now is None else now
    os.makedirs(run, exist_ok=True)
    _write_json(os.path.join(run, "trainer.json"),
                {"job": job, "node": node, "ip": ip, "start": float(start), "heartbeat": now})


def engine_check(run, since, stale=TRAINER_STALE, grace=TRAINER_GRACE, seen_file=None, now=None):
    """serve | stop | gone | alone (see the module docstring). A trainer counts as seen when its heartbeat is fresh
    (<= FRESH s); once seen, it is gone when its heartbeat is older than ``stale`` (or the file is gone). Without
    ``seen_file`` nothing is remembered, so a seen trainer whose heartbeat is no longer fresh counts as not seen.
    The sighting and the staleness are separate tests: with ``stale`` < FRESH (the dry run), a beat aged between the
    two is a sighting and at once a dead trainer, not "no trainer yet" (which would end in "alone" and a STOP)."""
    now = time.time() if now is None else now
    if os.path.exists(os.path.join(run, "STOP")):
        return "stop"
    t = _read_json(os.path.join(run, "trainer.json"))
    age = now - float(t["heartbeat"]) if t and "heartbeat" in t else None
    seen = age is not None and age <= FRESH
    if seen and seen_file and not os.path.exists(seen_file):
        with open(seen_file, "w") as f:
            f.write(f"{t.get('job')} {now}\n")
    if seen or (seen_file and os.path.exists(seen_file)):
        return "gone" if age is None or age > stale else "serve"
    return "alone" if now - float(since) > grace else "serve"


def trainer_check(run, jobs, max_age=ENGINE_MAX_AGE, now=None):
    """True when every engine job in ``jobs`` has a fresh, serving record."""
    live = {r["job"] for r in records(run, max_age, now)}
    return all(j in live for j in jobs)


def wait(run, n_min=1, timeout=1200, load_timeout=1800, max_age=ENGINE_MAX_AGE, poll=15, clock=time.time,
         sleep=time.sleep, out=sys.stdout):
    """The trainer's wait (see ``wait`` in the module docstring). Returns the exit code."""
    t0, first = clock(), None
    while True:
        if os.path.exists(os.path.join(run, "STOP")):
            print(f"STOP exists in {run}", file=sys.stderr)
            return 3
        got = pick(live_engines(run, max_age, clock()), n_min)
        if got:
            print(f"{' '.join(got[0])}\t{got[1]}\t{got[2]}\t{' '.join(got[3])}", file=out)
            return 0
        now = clock()
        if first is None and records(run, max_age, now, ("loading", "serving")):
            first = now
            print(f"an engine is loading ({now - t0:.0f} s after the start); waiting up to {load_timeout:.0f} s for it",
                  file=sys.stderr)
        if first is None and now - t0 > timeout:
            print(f"no engine record in {run}/engines after {timeout:.0f} s", file=sys.stderr)
            return 2
        if first is not None and now - first > load_timeout:
            print(f"no {n_min} serving engine(s) {load_timeout:.0f} s after the first record", file=sys.stderr)
            return 2
        sleep(poll)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("register"); s.add_argument("run"); s.add_argument("job"); s.add_argument("node"); s.add_argument("engines", nargs="+")
    s.add_argument("--state", default="serving", choices=["loading", "serving"]); s.add_argument("--log")
    s = sub.add_parser("beat"); s.add_argument("file")
    s = sub.add_parser("deregister"); s.add_argument("run"); s.add_argument("job")
    s = sub.add_parser("live"); s.add_argument("run"); s.add_argument("--max-age", type=float, default=ENGINE_MAX_AGE)
    s = sub.add_parser("wait"); s.add_argument("run"); s.add_argument("--min", type=int, default=1)
    s.add_argument("--timeout", type=float, default=1200); s.add_argument("--load-timeout", type=float, default=1800)
    s.add_argument("--max-age", type=float, default=ENGINE_MAX_AGE)
    s.add_argument("--poll", type=float, default=15)
    s = sub.add_parser("trainer-beat"); s.add_argument("run"); s.add_argument("job"); s.add_argument("node"); s.add_argument("ip"); s.add_argument("start")
    s = sub.add_parser("trainer-check"); s.add_argument("run"); s.add_argument("jobs", nargs="+"); s.add_argument("--max-age", type=float, default=ENGINE_MAX_AGE)
    s = sub.add_parser("engine-check"); s.add_argument("run"); s.add_argument("since", type=float)
    s.add_argument("--stale", type=float, default=TRAINER_STALE); s.add_argument("--grace", type=float, default=TRAINER_GRACE)
    s.add_argument("--seen")
    a = p.parse_args(argv)
    if a.cmd == "register":
        print(json.dumps(register(a.run, a.job, a.node, [parse_engine(x) for x in a.engines], a.state, a.log)))
    elif a.cmd == "beat":
        return 0 if beat(a.file) else 1
    elif a.cmd == "deregister":
        deregister(a.run, a.job)
    elif a.cmd == "live":
        for e in live_engines(a.run, a.max_age):
            print(json.dumps(e))
    elif a.cmd == "wait":
        return wait(a.run, a.min, a.timeout, a.load_timeout, a.max_age, a.poll)
    elif a.cmd == "trainer-check":
        return 0 if trainer_check(a.run, a.jobs, a.max_age) else 1
    elif a.cmd == "trainer-beat":
        trainer_beat(a.run, a.job, a.node, a.ip, a.start)
    elif a.cmd == "engine-check":
        print(engine_check(a.run, a.since, a.stale, a.grace, a.seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
