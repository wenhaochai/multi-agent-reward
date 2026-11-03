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


# OldestFirstDataBuffer: fake groups with one weight-version span, the real miles get and stale filter
def _group(version, index=0):
    span = SimpleNamespace(version=str(version), abs_start=0, abs_end=10)
    sample = SimpleNamespace(all_weight_version_spans=[span], index=index)
    return SimpleNamespace(group=[sample], prompt_group=[sample], version=version)


def _args(s=1, p=2, factor=2.0):
    return SimpleNamespace(max_weight_staleness=s, rollout_batch_size=p, async_data_buffer_capacity_factor=factor, dynamic_sampling_filter_path=None)


def _buffer(cls=None, **kw):
    cls = cls or er.OldestFirstDataBuffer
    dropped = []
    b = cls(SimpleNamespace(args=_args(**kw), unused_handler_fn=lambda group, why: dropped.append((group, why))))
    b._preput_filter = lambda entry: True     # aborted, missing reward, dynamic filter: miles' own
    b.dropped = dropped
    return b


def test_a_drain_takes_the_oldest_version_first_not_the_first_finished():
    async def go():
        b = _buffer(s=2, p=2, factor=3.0)
        for v in (5, 3, 4, 3):
            await b.put(_group(v))
        got = await b.get(num_groups=2, current_version=5)
        assert [e.version for e in got] == [3, 3]
        assert [e.version for e in b._buffer] == [4, 5]
    asyncio.run(go())


def test_a_drain_waits_for_a_group_at_its_last_chance_and_takes_it(monkeypatch):
    monkeypatch.setattr(er, "DEADLINE_POLL_S", 0.01)

    async def go():
        b = _buffer(s=1, p=2)
        await b.put(_group(5)); await b.put(_group(5))
        running = [4]
        b.in_flight_versions = lambda: list(running)
        t = asyncio.ensure_future(b.get(num_groups=2, current_version=5))
        await asyncio.sleep(0.05)
        assert not t.done()                      # 2 groups are in, one at version 4 is on its way
        await b.put(_group(4))                   # the worker's put comes before it takes the group off its list
        await asyncio.sleep(0.03)
        assert not t.done()
        running.clear()
        got = await asyncio.wait_for(t, 1)
        assert [e.version for e in got] == [4, 5]
        assert [e.version for e in b._buffer] == [5]
        assert b.get_metrics()["rollout/fully_async/deadline_wait_s"] > 0
    asyncio.run(go())


def test_a_group_past_or_before_its_last_chance_holds_no_drain_up(monkeypatch):
    async def go():
        b = _buffer(s=1, p=2)
        await b.put(_group(5)); await b.put(_group(5))
        b.in_flight_versions = lambda: [3, 5]    # 3 is lost anyway, 5 has another step
        got = await asyncio.wait_for(b.get(num_groups=2, current_version=5), 1)
        assert len(got) == 2
    asyncio.run(go())


def test_a_drain_leaves_older_groups_their_places_before_their_last_step(monkeypatch):
    """s = 2, P = 2, at the drain of version 5: 4 running groups of version 4 have the places of this drain and the next (5 and 6): taking 2 younger finished groups now would leave 4
    for the 2 places at 6, so this drain waits for 2 of them, though none is at its last chance."""
    monkeypatch.setattr(er, "DEADLINE_POLL_S", 0.01)

    async def go():
        b = _buffer(s=2, p=2, factor=3.0)
        await b.put(_group(5)); await b.put(_group(5))
        running = [4, 4, 4, 4]
        b.in_flight_versions = lambda: list(running)
        t = asyncio.ensure_future(b.get(num_groups=2, current_version=5))
        await asyncio.sleep(0.05)
        assert not t.done()
        await b.put(_group(4)); running.pop()
        await asyncio.sleep(0.05)
        assert not t.done()                      # one is in, 3 run: 4 for 2 + 2 places, this drain must take 2
        await b.put(_group(4)); running.pop()
        got = await asyncio.wait_for(t, 1)
        assert [e.version for e in got] == [4, 4]
    asyncio.run(go())


def test_with_slack_a_drain_takes_younger_groups_and_does_not_wait(monkeypatch):
    async def go():
        b = _buffer(s=2, p=2, factor=3.0)
        await b.put(_group(5)); await b.put(_group(5))
        b.in_flight_versions = lambda: [4, 4]    # 2 older groups, 2 + 2 places left: no need to wait
        got = await asyncio.wait_for(b.get(num_groups=2, current_version=5), 1)
        assert [e.version for e in got] == [5, 5]
    asyncio.run(go())


def test_without_a_staleness_bound_nothing_waits():
    async def go():
        b = _buffer(s=None, p=2)
        await b.put(_group(1)); await b.put(_group(0))
        b.in_flight_versions = lambda: [0, 1, 2]
        got = await asyncio.wait_for(b.get(num_groups=2, current_version=2), 1)
        assert [e.version for e in got] == [0, 1]
    asyncio.run(go())


def test_a_buffer_smaller_than_the_window_is_refused():
    try:
        _buffer(s=2, p=2, factor=2.0)
    except AssertionError as e:
        assert "window" in str(e)
    else:
        raise AssertionError("a buffer of 4 groups under a window of 6 was accepted")


async def _run_window(cls, s, p, seed, steps=120):
    """The window rule with random group durations: what the real producer does. Returns (the lags of the groups trained on, the groups dropped as stale)."""
    import random
    rng = random.Random(seed)
    b = _buffer(cls, s=s, p=p, factor=float(s + 1))
    version, draining, lags, running = 0, True, [], {}
    b.in_flight_versions = lambda: list(running.values())
    stop = asyncio.Event()

    async def one(v):
        await asyncio.sleep(rng.choice([0.001, 0.002, 0.003, 0.02]))
        return _group(v)

    async def producer():
        active = set()
        while not stop.is_set():
            while len(active) + len(b._buffer) < (s + (1 if draining else 0)) * p:
                t = asyncio.ensure_future(one(version))
                running[t] = version
                active.add(t)
            if not active:
                await asyncio.sleep(0.001)
                continue
            done, active = await asyncio.wait(active, timeout=0.001, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                await b.put(t.result())
                del running[t]
        for t in active:
            t.cancel()

    w = asyncio.ensure_future(producer())
    for _ in range(steps):
        draining = True
        got = await b.get(num_groups=p, current_version=version)
        lags += [version - e.version for e in got]
        draining = False
        await asyncio.sleep(0.004)
        version += 1
    stop.set()
    await w
    return lags, len(b.dropped)


def test_under_the_window_no_group_is_ever_dropped(monkeypatch):
    monkeypatch.setattr(er, "DEADLINE_POLL_S", 0.001)
    for s, p in ((1, 2), (2, 2), (3, 2), (1, 4), (2, 1), (2, 4), (3, 1)):
        for seed in range(6):
            lags, dropped = asyncio.run(_run_window(er.OldestFirstDataBuffer, s, p, seed))
            assert dropped == 0, (s, p, seed)
            assert max(lags) <= s


def test_miles_fifo_buffer_does_drop_them_in_the_same_loop():
    """So the test above can fail."""
    total = sum(asyncio.run(_run_window(er.DefaultDataBuffer, s, 2, seed))[1] for s in (1, 2) for seed in range(3))
    assert total > 0


# The real ElasticFullyAsyncRolloutFn, miles' worker loop and drain, real Sample objects (from the audit's harness); only generate_and_rm_group and GenerateState are fakes. 1 simulated min = MIN s.
MIN = 0.002


class _Engine:
    def __init__(self, seed, hang=None):
        import random
        self.rng, self.version, self.hang = random.Random(seed), 1, hang

    async def generate(self, state, group, sampling_params, evaluation=False, sample_done_callback=None):
        from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall
        if self.hang == group[0].group_index:
            await asyncio.Event().wait()

        async def one(sample, d):
            await asyncio.sleep(0.5 * MIN)
            first = self.version
            await asyncio.sleep(d * MIN)
            spans = [WeightVersionSpan(version=str(v), abs_start=j, abs_end=j + 1) for j, v in enumerate(range(first, self.version + 1))]
            sample.weight_versions = [WeightVersionsPerCall(spans=spans)]
            sample.status, sample.reward, sample.metadata = Sample.Status.COMPLETED, 1.0, {}
            if sample_done_callback is not None:
                sample_done_callback()

        await asyncio.gather(*[one(x, min(60.0, max(10.0, self.rng.lognormvariate(3.35, 0.45)))) for x in group])
        return group


class _Data:
    def __init__(self):
        self.n = 0

    def get_samples(self, k):
        from miles.utils.types import Sample
        self.n += 1
        return [[Sample(group_index=self.n, index=self.n * 100 + i, prompt="p", label="l") for i in range(8)]]


def _producer(monkeypatch, engine, s=2, p=2, order=True, per_engine=None):
    from argparse import Namespace
    import miles.rollout.fully_async_rollout as fa
    from miles.rollout.base_types import RolloutFnConstructorInput
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState

    class State(GenerateState):
        def __init__(self, args):
            self.args, self.sampling_params, self.aborted = args, {}, False
            self.generate_fn_semaphore, self.generate_function = asyncio.Semaphore(4096), None

    monkeypatch.setattr(fa, "GenerateState", State)
    monkeypatch.setattr(fa, "generate_and_rm_group", engine.generate)
    monkeypatch.setattr(er, "routed_engines", lambda ip, port, timeout=2.0: 1)
    monkeypatch.setattr(er, "DEADLINE_POLL_S", 0.002)
    monkeypatch.setenv("MA_ASYNC_SAMPLES_PER_ENGINE", str(per_engine or (s + 1) * p * 8))
    a = Namespace(rollout_global_dataset=True, debug_rollout_only=False, debug_skip_weight_update=False, rollout_batch_size=p, n_samples_per_prompt=8, max_weight_staleness=s,
                  async_max_concurrent_samples=None, async_data_buffer_capacity_factor=float(s + 1), async_unused_samples_handler="drop",
                  custom_async_data_buffer_path="flint.elastic_rollout.OldestFirstDataBuffer" if order else None, enable_sample_ownership_checker=False,
                  custom_async_data_buffer_path_per_model=None, megatron_config=None, rollout_submission_granularity=None, dynamic_sampling_filter_path=None, reward_key=None,
                  rollout_sample_filter_path=None, buffer_filter_path=None, sglang_router_ip="127.0.0.1", sglang_router_port=1, sglang_router_request_timeout_secs=14400, eval_num_gpus=0,
                  namespaced_radix_cache=True, update_weights_interval=1)
    return er.ElasticFullyAsyncRolloutFn(RolloutFnConstructorInput(args=a, data_source=_Data()))


async def _train(fn, engine, steps):
    from miles.rollout.base_types import RolloutFnTrainInput
    from miles.rollout.filter_hub.common_filters import group_weight_version_stats
    lags, stale = [], 0
    for r in range(steps):
        out = await fn(RolloutFnTrainInput(rollout_id=r, weight_version=engine.version))
        stale += out.metrics["rollout/fully_async/stale_groups_filtered"]
        lags += [engine.version - group_weight_version_stats(g).oldest_version for g in out.samples]
        await asyncio.sleep(engine.rng.uniform(5, 14) * MIN)
        engine.version += 1
        await asyncio.sleep(MIN)
    return lags, stale, fn


async def _stop(fn):
    fn._worker_error_cancel_targets.clear()
    await fn.dispose()
    for x in list(fn._running_tasks):
        x.task.cancel()


def test_the_real_producer_drops_no_group_under_the_window(monkeypatch):
    for s, p in ((1, 2), (2, 2), (2, 1), (3, 2)):
        for seed in range(2):
            async def go():
                e = _Engine(seed)
                fn = _producer(monkeypatch, e, s, p)
                assert isinstance(fn._output, er.OldestFirstDataBuffer) and fn._output.in_flight_versions == fn._in_flight_versions
                lags, stale, fn = await _train(fn, e, 40)
                assert len(fn._started) <= (s + 1) * p      # finished tasks leave
                await _stop(fn)
                return lags, stale
            lags, stale = asyncio.run(go())
            assert stale == 0, (s, p, seed)
            assert max(lags) <= s and min(lags) >= 0, (s, p, seed, lags)


def test_miles_buffer_drops_groups_in_the_same_loop(monkeypatch):
    """So the test above can fail; a producer without the rule behaves like this one."""
    total = 0
    for seed in range(4):
        async def go():
            e = _Engine(seed)
            fn = _producer(monkeypatch, e, 2, 2, order=False)
            assert not fn._deadlines
            _, stale, fn = await _train(fn, e, 40)
            assert fn._started == {}     # nothing is kept for a rule that is off: a finished task holds its samples
            await _stop(fn)
            return stale
        total += asyncio.run(go())
    assert total > 0


def test_a_hung_group_holds_drains_only_until_the_limit(monkeypatch):
    monkeypatch.setenv("MA_LG_WALL_MIN", "0")
    monkeypatch.setattr(er, "GROUP_GRACE_S", 0.3)

    async def go():
        e = _Engine(1, hang=1)       # group 1 starts at version 1 and never ends
        fn = _producer(monkeypatch, e, 2, 2)
        lags, stale, fn = await asyncio.wait_for(_train(fn, e, 6), 30)
        await _stop(fn)
        return lags
    assert len(asyncio.run(go())) == 12


def test_a_failed_worker_cancels_the_drain_that_waits(monkeypatch):
    class Bad(_Engine):
        async def generate(self, state, group, sampling_params, evaluation=False, sample_done_callback=None):
            if group[0].group_index == 7:      # while the drain of version 3 waits for group 1, at its last chance
                while self.version < 3:
                    await asyncio.sleep(MIN)
                await asyncio.sleep(2 * MIN)
                raise RuntimeError("the group raised")
            return await super().generate(state, group, sampling_params, evaluation, sample_done_callback)

    async def go():
        e = Bad(1, hang=1)
        fn = _producer(monkeypatch, e, 2, 2)
        try:
            await asyncio.wait_for(_train(fn, e, 30), 30)
            return "went on"
        except asyncio.CancelledError:
            return "cancelled"
        finally:
            await _stop(fn)
    assert asyncio.run(go()) == "cancelled"
