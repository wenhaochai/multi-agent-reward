"""Run the long game (flint.game) on a data file against served Flint: the game, the shell, the submit command and
the judge run here; only the model calls go to the OpenAI-compatible chat endpoints (flint.chat).

  python3 run_eval.py --base-url http://127.0.0.1:8000[,http://...] --model flint --data data/fcs11_val40_agent.jsonl \
      --out <dir> [--episodes 1] [--concurrency 32]
Game settings come from the environment (flint.game: MA_LG_N, MA_LG_TOKENS, ...; MA_AG_TRACE_DIR defaults to
<out>/traces). Output: <out>/results.jsonl, one line per finished episode (resumable).

One driver per --out: the driver holds a POSIX lock on <out>/.lock while it runs, and a second driver on the same
directory exits 1 at once.

Checkpoints (default on, --no-ckpt off): each episode saves its state to <out>/state/<pid>_e<episode>.{pkl,r<round>.tgz}
at the end of a round every MA_LG_CKPT_S seconds (300) and after every round with a submission (game.Checkpoint), and
the next run resumes it there. A transport abort keeps the state; a crash or a finished row removes it. The run stops
early, pausing every episode at the end of its round (up to STOP_GRACE_S, then cancelled at its last save), 10 min
before SLURM_JOB_END_TIME, or when a server is down (its GPUs idle; the requeue starts fresh ones).

Exit status (wq requeues a --partial job on 99): 0 when every episode has a clean score or has failed
--max-episode-tries times; 99 when some episode can still be retried or resumed, so the next run takes only those;
1 when every episode this run started crashed (a bug: fail fast instead of looping), the run wrote no row and saved no
checkpoint, or another driver holds --out. An exception in one episode (a 400 reply, a bad JSON reply, a bug)
becomes that episode's aborted row and does not stop the others."""
import argparse
import asyncio
import fcntl
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from flint import Aborted
from flint import game as G
from flint import sandbox as C
from flint.chat import ChatSession, Client, HttpAborted

WATCH_S, STOP_MARGIN_S, STOP_GRACE_S = 60, 600, 420   # stop checks; stop this long before the end; pause wait


def _clean(r: dict) -> bool:
    """A row whose score counts: S is set and no submission met a judge infrastructure error (its 0 is not the
    model's), so a rerun keeps it and skips the episode."""
    return r.get("S") is not None and not (r.get("info") or {}).get("fo_judge_infra_error")


def _read(path: Path) -> list:
    """results.jsonl's rows; a line cut short by a kill is skipped (its episode reruns)."""
    rows = []
    for line in (path.read_text().splitlines() if path.exists() else []):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def _append(path: Path, rec: dict) -> None:
    """One row at the end of results.jsonl. A line a kill cut short is ended first, so it does not swallow this row
    (_read then skips the cut line alone)."""
    with open(path, "a+b") as f:
        lead = b""
        if f.seek(0, os.SEEK_END):
            f.seek(-1, os.SEEK_END)
            lead = b"" if f.read(1) == b"\n" else b"\n"
        f.write(lead + (json.dumps(rec) + "\n").encode())


def outcome(want: list, recs: list, ran: list, max_tries: int, episodes: int, kept: bool = False) -> tuple[str, int]:
    """The summary line and the exit status. A key (pid, episode) of `want` is complete with a clean row, or given up
    after max_tries rows that are not clean (aborted, crashed or judge-infra); the others are retried by the next run
    (exit 99). The mean covers the clean rows only, and the line says how many keys it leaves out. Exit 1 when every
    row of this run is a crash and no other episode of it kept work for the rerun (kept: paused, cancelled, or a
    checkpoint still on disk)."""
    by: dict = {}
    for r in recs:
        by.setdefault((r["pid"], r["episode"]), []).append(r)
    ok, retry, given_up, infra = {}, [], [], 0
    for k in want:
        clean = [r for r in by.get(k, []) if _clean(r)]
        if clean:
            ok[k] = clean[-1]
            continue
        (given_up if len(by.get(k, [])) >= max_tries else retry).append(k)
        infra += any(r.get("S") is not None for r in by.get(k, []))

    def mean(xs):
        return 100 * sum(r["S"] for r in xs) / len(xs) if xs else float("nan")
    by_ep = " ".join(f"ep{e} {mean([r for (_, ep), r in ok.items() if ep == e]):.2f}" for e in range(episodes))
    line = (f"{'INCOMPLETE' if retry else 'DONE'} {len(ok)}/{len(want)} clean episodes (to rerun {len(retry)}, given "
            f"up after {max_tries} tries {len(given_up)}, judge-infra among them {infra}); mean final score "
            f"{mean(list(ok.values())):.2f} over clean episodes; {by_ep}")
    # a problem with one test case scores the case the agent saw (game.split_cases): the held-out mean leaves those out
    ho = [r for r in ok.values() if not (r.get("info") or {}).get("fo_lg_shared_case")]
    if ho:
        line += (f"\nHELD-OUT {len(ho)} clean episodes ({len(ok) - len(ho)} with a single shared case left out), mean "
                 f"final score {mean(ho):.2f}")
    if ran and not kept and not any(_clean(r) for r in ran) and all(r.get("crash") for r in ran):
        return line + f"\nFAILED: all {len(ran)} episodes of this run crashed", 1
    return line, 99 if retry else 0


def make_client(args) -> Client:
    return Client([u.strip().rstrip("/") for u in args.base_url.split(",") if u.strip()], args.model, args.temperature,
                  args.top_p, args.top_k, route=not args.no_route, request_timeout=args.request_timeout)


def lock_out(out_dir: Path):
    """The lock that keeps a second driver off this results directory, also on another node: GPFS applies fcntl locks
    on every node and flock locks only on one (tested on della-vis1 and della-vis2, 2026-10-28). Held while the returned
    file stays open."""
    f = open(out_dir / ".lock", "a")
    try:
        fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(f"another driver holds {out_dir} (lock {out_dir / '.lock'}): refusing")
    return f


async def main(args, client: Client | None = None) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_out(out_dir)   # noqa: F841 (held until main returns)
    os.environ.setdefault("MA_AG_TRACE_DIR", str(out_dir / "traces"))
    client = client or make_client(args)
    stop: list = [None]   # why the run stops early: "time" or "down"
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    res_path = out_dir / "results.jsonl"
    old = _read(res_path)
    done = {(r["pid"], r["episode"]) for r in old if _clean(r)}
    tries = Counter((r["pid"], r["episode"]) for r in old if not _clean(r))
    want = [(r["metadata"]["pid"], e) for e in range(args.episodes) for r in rows]
    todo = [(i, e, r) for e in range(args.episodes) for i, r in enumerate(rows)
            if (r["metadata"]["pid"], e) not in done and tries[(r["metadata"]["pid"], e)] < args.max_episode_tries]
    n_cap = sum(k not in done and tries[k] >= args.max_episode_tries for k in want)
    print(f"{len(rows)} problems x {args.episodes} episodes; {len(done)} done, {len(todo)} to run, {n_cap} given up "
          f"after {args.max_episode_tries} tries; model {args.model}; "
          f"N={G.N} tokens={G.TOKENS} ctx_chars={G.CTX_CHARS} ctx_low={G.CTX_LOW} turn={G.TURN} max_len={G.MAX_LEN} "
          f"harness={G.harness_tag()}", flush=True)
    if todo:   # each command gets a private /tmp (sandbox.private_tmp): on a node that refuses the namespace every
        # command of every episode would fail, so the run stops here (exit 1, not a requeue)
        probe = os.path.join(os.environ.get("TMPDIR", "/tmp"), "agent_long", f"probe_{os.getpid()}")
        os.makedirs(probe, exist_ok=True)
        r = await asyncio.get_running_loop().run_in_executor(G.POOL, C.run_command, "echo ok > /tmp/p && cat /tmp/p",
                                                             probe, True)
        G.rm_work(probe)
        if r["output"] != "ok\n":
            print(f"PRIVATE_TMP_FAIL {r}", flush=True)
            return 1

    def ckpt_for(pid, e):
        return G.Checkpoint(out_dir / "state" / f"{str(pid).replace('/', '_')}_e{e}")
    if not args.no_ckpt:
        for k in done:   # a kill between an episode's row and its clear left the state
            ckpt_for(*k).clear()
        n_res = sum(ckpt_for(r["metadata"]["pid"], e).exists() for _, e, r in todo)
        print(f"{n_res} of them resume from a checkpoint", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    await client.open()
    ran, cks, running, paused = [], [], {}, [0]   # this run's rows and checkpoints; episodes in flight

    def new_session(inp, name):
        return ChatSession(client, inp, name, G.MAX_LEN)

    async def one(i, e, r):
        async with sem:
            if client.down or stop[0]:   # a server died or the run stops: the requeue (exit 99) runs it on fresh servers
                return
            t0 = time.time()
            key = (r["metadata"]["pid"], e)
            ck = None if args.no_ckpt else ckpt_for(*key)
            if ck is not None:
                cks.append(ck)
            running[key] = ck
            url = min(client.urls, key=client.active.get)
            client.active[url] += 1
            inp = SimpleNamespace(evaluation=True, sampling_params={}, base_url=url, moved_at=t0, moves=0,
                                  sample=SimpleNamespace(index=i, group_index=e, label=r["label"], metadata=r["metadata"]))
            rec = {"pid": r["metadata"]["pid"], "episode": e, "model": args.model}
            try:
                _, S, info = await G.episode(inp, r["metadata"]["messages"], r["label"], new_session, ckpt=ck)
                rec.update(S=S, info=info)
            except G.Paused as ex:   # the run stops: the state is saved, no row; the rerun resumes it
                paused[0] += 1
                rec = None
                print(f"[{key[0]} ep{e}] {ex}", flush=True)
            except HttpAborted as ex:
                rec.update(S=None, aborted=str(ex))
                if ex.url in client.down:   # every server is down: not the episode's failure, no row; the rerun resumes
                    rec = None
            except Aborted as ex:
                rec.update(S=None, aborted=str(ex))
            except Exception as ex:   # one episode's bug or bad reply; CancelledError (a BaseException) still stops all
                traceback.print_exc()
                rec.update(S=None, aborted=f"{type(ex).__name__}: {ex}"[:500], crash=True,
                           traceback=traceback.format_exc()[-4000:])
            finally:
                client.active[inp.base_url] -= 1   # the server it ended on
                running.pop(key, None)
            if rec is None:
                return
            rec["moves"] = inp.moves
            rec["minutes"] = round(((ck.prior_s if ck is not None else 0.0) + time.time() - t0) / 60, 1)   # all runs
            _append(res_path, rec)
            ran.append(rec)
            if ck is not None and (rec["S"] is not None or rec.get("crash")):   # a transport abort resumes
                ck.clear()
            print(f"[{rec['pid']} ep{e}] S={rec.get('S')} subs={(rec.get('info') or {}).get('fo_lg_submissions')} "
                  f"tokens={(rec.get('info') or {}).get('fo_lg_tokens')} {rec['minutes']} min moves={inp.moves}"
                  + (f" ABORTED {rec['aborted'][:200]}" if rec.get("S") is None else ""), flush=True)

    tasks = [asyncio.create_task(one(i, e, r)) for i, e, r in todo]

    async def watch():
        while True:
            await asyncio.sleep(WATCH_S)
            end = os.environ.get("SLURM_JOB_END_TIME", "")
            if end.isdigit() and time.time() > int(end) - STOP_MARGIN_S:
                stop[0] = "time"
            elif client.down:
                stop[0] = "down"
            if stop[0]:
                break
        print(f"stopping ({stop[0]}): {len(running)} episodes in flight pause at the end of their round", flush=True)
        for ck in running.values():
            if ck is not None:
                ck.stop = True
        t_end = time.time() + STOP_GRACE_S
        while any(ck is not None for ck in running.values()) and time.time() < t_end:
            await asyncio.sleep(min(WATCH_S, 5))
        for t in tasks:   # the rest lose the work since their last save
            t.cancel()

    watcher = asyncio.create_task(watch())
    try:
        if tasks:
            await asyncio.wait(tasks)
    finally:
        watcher.cancel()
        await client.close()
    for t in tasks:
        if not t.cancelled() and t.exception() is not None:   # the driver's own bug (one() catches the episode's)
            raise t.exception()
    n_cut = sum(t.cancelled() for t in tasks)
    # a crash clears its own state, so only a state still on disk is work the rerun resumes
    kept = bool(paused[0] or n_cut or any(ck.saves and ck.exists() for ck in cks))
    line, code = outcome(want, _read(res_path), ran, args.max_episode_tries, args.episodes, kept)
    print(line, flush=True)
    if client.down:
        print(f"servers down: {sorted(client.down)}", flush=True)
    if stop[0] or paused[0] or n_cut:
        print(f"stopped ({stop[0]}): {paused[0]} episodes paused at a checkpoint, {n_cut} cancelled", flush=True)
    if code == 99 and not ran and not sum(ck.saves for ck in cks):   # no row, no checkpoint: a rerun would repeat
        print("no progress in this run", flush=True)
        code = 1
    if code:
        print(f"exit {code}" + (": rerun the same command to resume" if code == 99 else ""), flush=True)
    return code


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="one URL, or several comma-separated (one per server replica)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (smoke)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--max-episode-tries", type=int, default=3,
                    help="an episode whose rows are all aborted or judge-infra after this many runs is given up")
    ap.add_argument("--no-route", action="store_true", help="episodes stay on their first server unless it is down")
    ap.add_argument("--no-ckpt", action="store_true", help="no episode checkpoints: a stopped run restarts its episodes")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse())))
