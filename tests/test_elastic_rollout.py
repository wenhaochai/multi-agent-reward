"""flint.elastic_rollout: the fully async producer's in-flight budget from the router's routed engines, its staleness
window and wake-up, and the warm-up engine's join and leave. Run in
miles-v0.1.3.sif with miles-elastic on the path: python3 -m pytest -q tests/test_elastic_rollout.py"""
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flint import elastic_rollout as er   # noqa: E402


def _router(workers):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            b = json.dumps({"workers": [{"url": u} for u in workers]}).encode()
            self.send_response(200 if self.path == "/workers" else 404)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_routed_engines_counts_each_engine_once():
    workers = ["http://172.17.0.1:30000", "http://172.17.0.2:20040/", "http://172.17.0.3:20040@0", "http://172.17.0.3:20040@1"]
    srv = _router(workers)
    try:
        assert er.routed_engines("127.0.0.1", srv.server_address[1]) == 3
    finally:
        srv.shutdown()


def test_an_unreachable_router_gives_none():
    assert er.routed_engines("127.0.0.1", 9, timeout=0.5) is None


def test_the_budget_is_whole_groups_per_routed_engine():
    assert er.in_flight_groups(1, 16, 8) == 2
    assert er.in_flight_groups(3, 16, 8) == 6
    assert er.in_flight_groups(0, 16, 8) == 2     # no engine routed yet: as one
    assert er.in_flight_groups(1, 4, 8) == 1      # at least one group


def test_the_producer_follows_the_router_and_keeps_the_last_count(monkeypatch):
    p = er.ElasticFullyAsyncRolloutFn.__new__(er.ElasticFullyAsyncRolloutFn)
    p.args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=1, n_samples_per_prompt=8)
    p._per_engine, p._engines, p._checked, p._budget = 16, 1, float("-inf"), None
    counts = iter([1, 3, None])
    monkeypatch.setattr(er, "routed_engines", lambda ip, port: next(counts))
    clock = iter([0.0, 30.0, 61.0, 200.0])
    monkeypatch.setattr(er.time, "monotonic", lambda: next(clock))
    assert p._max_in_flight_groups() == 2    # t=0: 1 engine
    assert p._max_in_flight_groups() == 2    # t=30: not re-read
    assert p._max_in_flight_groups() == 6    # t=61: 3 engines
    assert p._max_in_flight_groups() == 6    # t=200: the router does not answer, keep 3


def test_router_workers_lists_each_engine_once():
    workers = ["http://172.17.0.3:20040@1", "http://172.17.0.1:30000", "http://172.17.0.3:20040@0", "http://172.17.0.2:20040/"]
    srv = _router(workers)
    try:
        assert er.router_workers("127.0.0.1", srv.server_address[1]) == [
            "http://172.17.0.1:30000", "http://172.17.0.2:20040", "http://172.17.0.3:20040"]
    finally:
        srv.shutdown()
    assert er.router_workers("127.0.0.1", 9, timeout=0.5) is None


def test_the_staleness_window():
    assert er.window_groups(None, 2, True) is None           # no bound: no window
    assert er.window_groups(1, 2, True) == 4                  # a drain waits: this step's P and the next step's P
    assert er.window_groups(1, 2, False) == 2                 # the step trains: only the next step's P
    assert er.window_groups(2, 3, False) == 6


def test_the_window_counts_groups_in_flight_and_finished_not_trained_on():
    p = er.ElasticFullyAsyncRolloutFn.__new__(er.ElasticFullyAsyncRolloutFn)
    p.args = SimpleNamespace(max_weight_staleness=1, rollout_batch_size=2)
    p._output = SimpleNamespace(_buffer=[object()])
    p._draining = True
    assert p._window_free(2) == 1          # 4 - 2 in flight - 1 finished
    p._draining = False
    assert p._window_free(2) == -1         # 2 - 2 - 1: full
    p.args.max_weight_staleness = None
    assert p._window_free(10) is None


class _Inner:
    sample_done_callback = "cb"

    def __init__(self, answer=True):
        self.answer, self.asked, self.waited = answer, 0, 0

    def has_capacity(self, *, pending_groups, group_budget):
        self.asked += 1
        return self.answer

    def on_submit(self, groups):
        pass

    async def wait_for_progress(self, pendings):
        self.waited += 1
        return await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)


def test_a_full_window_stops_submissions_without_asking_the_scheduler():
    free = {"n": 0}
    inner = _Inner(answer=True)
    s = er.PacedScheduler(inner, lambda pending: free["n"], asyncio.Event(), recheck_s=0.05)
    assert s.sample_done_callback == "cb"
    assert not s.has_capacity(pending_groups=4, group_budget=8) and inner.asked == 0
    free["n"] = 1
    assert s.has_capacity(pending_groups=3, group_budget=8) and inner.asked == 1
    inner.answer = False
    assert not s.has_capacity(pending_groups=3, group_budget=8)
    s2 = er.PacedScheduler(_Inner(), lambda pending: None, asyncio.Event())   # no window: the scheduler decides
    assert s2.has_capacity(pending_groups=100, group_budget=1)


def test_the_producer_wakes_on_a_finished_group_a_wake_or_the_recheck():
    async def main():
        wake = asyncio.Event()
        inner = _Inner()
        s = er.PacedScheduler(inner, lambda pending: 1, wake, recheck_s=0.2)
        s.has_capacity(pending_groups=1, group_budget=8)            # window open: the scheduler's own wait
        quick = asyncio.ensure_future(asyncio.sleep(0.01))
        slow = asyncio.ensure_future(asyncio.sleep(10))
        done, pending = await s.wait_for_progress({quick, slow})
        assert done == {quick} and pending == {slow} and inner.waited == 1
        # a wake (a drain began, an engine joined): back at once, nothing done
        asyncio.get_running_loop().call_later(0.02, wake.set)
        t0 = time.monotonic()
        done, pending = await s.wait_for_progress({slow})
        assert done == set() and pending == {slow} and time.monotonic() - t0 < 0.15
        # nothing happens: back after recheck_s
        t0 = time.monotonic()
        done, pending = await s.wait_for_progress({slow})
        assert done == set() and pending == {slow} and 0.15 < time.monotonic() - t0 < 1.0
        # no group in flight (the window was full): back after recheck_s, or at a wake
        done, pending = await s.wait_for_progress(set())
        assert done == set() and pending == set()
        # a full window waits on whole groups, not on the scheduler's sample wake-ups
        full = er.PacedScheduler(inner, lambda pending: 0, wake, recheck_s=0.2)
        assert not full.has_capacity(pending_groups=1, group_budget=8)
        quick = asyncio.ensure_future(asyncio.sleep(0.01))
        waited = inner.waited
        done, pending = await full.wait_for_progress({quick, slow})
        assert done == {quick} and inner.waited == waited
        assert not slow.done()
        slow.cancel()
    asyncio.run(main())


W_URL, OWN_URL = "http://172.17.0.9:20040", "http://172.17.0.1:30000"


def _warmup(tmp_path, *, checksums=None, launched=True, serving=True, **kw):
    from flint.engine_registry import registry
    d = tmp_path / "warmup"
    (d / "engines").mkdir(parents=True)
    if launched:
        (d / "launched").write_text("123w\n")
    if serving:
        registry.register(str(d), "123w", "node", [{"url": W_URL, "tp": 4, "gpus": [0, 1, 2, 3]}], state="serving")
    calls, changes = [], []

    async def add(url):
        calls.append(("add", url))

    async def remove(url):
        calls.append(("remove", url))

    w = er.Warmup(d, workers=lambda: [OWN_URL, W_URL], add=add, remove=remove,
                  checksums=checksums or (lambda url: {"rank0/w": "a1", "rank1/w": "b2"}), version=lambda url: "1",
                  set_version=lambda url, v: calls.append(("version", url, v)), on_change=lambda: changes.append(1),
                  poll_s=0.01, **kw)
    return w, d, calls, changes


def _gone(d, mib="2048", after=0.05):
    async def write():
        await asyncio.sleep(after)
        (d / "gone").write_text(json.dumps({"job": "123w", "time": 0, "max_gpu_mib": mib}))
    return asyncio.ensure_future(write())


def test_the_warmup_engine_joins_with_equal_weights_and_leaves_at_the_first_step(tmp_path):
    w, d, calls, changes = _warmup(tmp_path)

    async def main():
        w.start()
        await w._task
        assert w.url == W_URL and calls == [("version", W_URL, "1"), ("add", W_URL)] and changes == [1]
        g = _gone(d)
        await w.release()
        await g
        assert calls[-1] == ("remove", W_URL) and changes == [1, 1] and (d / "leave").exists()
        await w.release()                     # later drains: nothing more
        w.start()
        assert calls[-1] == ("remove", W_URL) and w._task.done()
    asyncio.run(main())


def test_the_first_groups_wait_for_the_warmup_engine_at_most_the_hold(tmp_path):
    now = [100.0]
    w, d, calls, changes = _warmup(tmp_path / "a", clock=lambda: now[0], hold_s=300.0, serving=False)
    assert w.holding()                         # before the first drain too
    w._hold_until = None

    async def main():
        w.start()
        assert w.holding()
        now[0] += 299
        assert w.holding()
        now[0] += 2
        assert not w.holding(), "the hold ends"
        w._task.cancel()
    asyncio.run(main())

    # it joined: no more hold
    w, d, calls, changes = _warmup(tmp_path / "b")

    async def joined():
        w.start()
        assert w.holding()
        await w._task
        assert w.url == W_URL and not w.holding()
    asyncio.run(joined())

    # no warm-up job, one that was sent away, one that ended, one released
    w, d, calls, changes = _warmup(tmp_path / "c", launched=False)
    assert not w.holding()
    w, d, calls, changes = _warmup(tmp_path / "d", serving=False)
    (d / "leave").write_text("1 sent away")
    assert not w.holding()
    w, d, calls, changes = _warmup(tmp_path / "e", serving=False)
    (d / "gone").write_text("{}")
    assert not w.holding()
    w, d, calls, changes = _warmup(tmp_path / "f", serving=False)
    w._left = True
    assert not w.holding()


def test_a_holding_warmup_closes_the_window(tmp_path):
    w, d, calls, changes = _warmup(tmp_path, serving=False)
    p = er.ElasticFullyAsyncRolloutFn.__new__(er.ElasticFullyAsyncRolloutFn)
    p.args = SimpleNamespace(max_weight_staleness=1, rollout_batch_size=2)
    p._output = SimpleNamespace(_buffer=[])
    p._draining = True
    p._warmup = w
    assert p._window_free(0) == 0
    w._left = True
    assert p._window_free(0) == 4
    p._warmup = None
    assert p._window_free(0) == 4


def test_the_warmup_engine_waits_for_the_router_to_list_the_own_engine(tmp_path):
    w, d, calls, changes = _warmup(tmp_path)
    lists = iter([None, [], [W_URL]])
    w._workers = lambda: next(lists, [OWN_URL])    # POST /workers answers before the worker is listed

    async def main():
        w.start()
        await w._task
        assert w.url == W_URL and calls == [("version", W_URL, "1"), ("add", W_URL)] and not (d / "leave").exists()
    asyncio.run(main())


def test_a_router_that_never_lists_the_own_engine_sends_the_warmup_engine_away(tmp_path):
    w, d, calls, changes = _warmup(tmp_path, join_s=0.05)
    w._workers = lambda: []

    async def main():
        w.start()
        await w._task
        assert w.url is None and calls == [] and (d / "leave").exists()
    asyncio.run(main())


def test_a_warmup_engine_with_other_weights_is_sent_away(tmp_path):
    w, d, calls, changes = _warmup(tmp_path, checksums=lambda url: {"rank0/w": "a1" if url == W_URL else "zz"})

    async def main():
        w.start()
        await w._task
        assert w.url is None and calls == [] and (d / "leave").exists()
        g = _gone(d)
        await w.release()
        await g
        assert calls == []
    asyncio.run(main())


def test_no_warmup_without_launched(tmp_path):
    w, d, calls, changes = _warmup(tmp_path, launched=False)

    async def main():
        w.start()
        assert w._task is None
        await w.release()
        assert not (d / "leave").exists() and calls == []
    asyncio.run(main())


def test_the_first_step_waits_until_the_warmup_engine_is_gone_and_its_gpus_are_free(tmp_path):
    w, d, calls, changes = _warmup(tmp_path, serving=False, gone_s=0.3)

    async def main():
        w.start()                             # still loading
        await asyncio.sleep(0.03)
        try:
            await w.release()                 # no gone within gone_s
            raise AssertionError("release returned without gone")
        except RuntimeError as e:
            assert "not gone" in str(e)
        assert (d / "leave").exists() and w._task.done() and calls == []
    asyncio.run(main())

    w, d, calls, changes = _warmup(tmp_path / "b", serving=False)

    async def high():
        w.start()
        g = _gone(d, mib="40000")
        try:
            await w.release()
            raise AssertionError("release returned with the GPUs still full")
        except RuntimeError as e:
            assert "40000" in str(e)
        await g
    asyncio.run(high())
