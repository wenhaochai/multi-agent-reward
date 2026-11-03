"""The fully async producer of an async RL run (rl/rl_inner.sh async=1): miles' FullyAsyncRolloutFn, plus
  - an in-flight budget from the engines the router serves: MA_ASYNC_SAMPLES_PER_ENGINE (default 16) episodes per
    routed engine, in whole GRPO groups, at least one group. The routed engines (sgl-router GET /workers, read at most
    once a minute, and at once when the warm-up engine joins or leaves) are the run's own engine, every engine of an
    engine job that has received the weights (flint/engine_registry.py) and the warm-up engine (below). A failed read
    keeps the last count;
  - a staleness window (with --max-weight-staleness s; miles pushes the weights after every step): the groups in
    flight plus the finished groups not yet trained on stay at most (s + 1) P while the trainer waits for a step's
    data and s P while it trains (P = --rollout-batch-size, the groups of one step). A group started before a push is
    then trained on at most s steps after it, not dropped as stale; only a group that finishes after later ones still
    can be. Engines beyond what the window fills lower the episodes per engine;
  - a wake-up when a drain starts or ends, when the warm-up engine joins or leaves, and every RECHECK_S: a budget or a
    window that grew is used then, not at the next finished episode (an engine that joined used to wait for one);
  - the warm-up engine of a fresh start (MA_WARMUP_DIR; rl/elastic_engine.sh EL_WARMUP=1, which rl/rl_job.sh starts on
    the actor's 4 GPUs, idle until the first step's data is in). It loads the base checkpoint, which only on a fresh
    start is what the first weight push gave the run's own engine; miles never pushes to it (NCCL cannot place an
    engine rank on an actor rank's GPU). Once it serves, its weights are compared with the own engine's tensor by
    tensor (SGLang /weights_checker hashes); if every tensor is equal it takes the own engine's weight version and
    joins the router. When the first step's data is in, the producer takes it out of the router, asks it to leave
    (MA_WARMUP_DIR/leave: it finishes its running requests, at most 10 min, and ends) and hands the data to the trainer
    only once it is gone with its GPUs free (MA_WARMUP_DIR/gone), so the step trains on those GPUs.

miles (flint-grpo-elastic): --fully-async --rollout-function-path flint.elastic_rollout.ElasticFullyAsyncRolloutFn.
"""

import asyncio
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn

logger = logging.getLogger(__name__)

RECHECK_S = 60.0
WARMUP_JOIN_S = 1800.0     # the warm-up engine serves within 30 min of the first drain, or it is sent away
WARMUP_GONE_S = 900.0      # ... and is gone within 15 min of its leave (a drain of at most 10 min, then its GPUs)
WARMUP_GPU_MIB = 12288     # the most its GPUs may hold once it is gone (the actor keeps 2-7 GiB on each: w894)
WARMUP_POLL_S = 5.0
CHECKSUM_TIMEOUT_S = 600.0


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_json(url: str, timeout: float):
    with _opener().open(url, timeout=timeout) as r:
        return json.loads(r.read())


def _post_json(url: str, body: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    with _opener().open(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def _engine_url(url: str) -> str:
    """An engine's url as the router lists it (once per dp rank as url@rank) -> the url."""
    url = url.rstrip("/")
    head, _, tail = url.rpartition("@")
    return head if head and tail.isdigit() else url


def router_workers(router_ip: str, router_port: int, timeout: float = 2.0) -> list[str] | None:
    """The engines the router routes to, each once; None if the router does not answer."""
    try:
        workers = _get_json(f"http://{router_ip}:{router_port}/workers", timeout)["workers"]
        return sorted({_engine_url(w["url"]) for w in workers})
    except (OSError, ValueError, KeyError, TypeError):
        return None


def routed_engines(router_ip: str, router_port: int, timeout: float = 2.0) -> int | None:
    """How many engines the router routes to (an engine listed once per dp rank as url@rank counts once)."""
    urls = router_workers(router_ip, router_port, timeout)
    return None if urls is None else len(urls)


def in_flight_groups(engines: int, samples_per_engine: int, group_size: int) -> int:
    return max(1, samples_per_engine * max(1, engines) // group_size)


def window_groups(staleness: int | None, batch: int, draining: bool) -> int | None:
    """The groups that may be in flight or finished and not yet trained on: (s + 1) P while a drain waits for a step's
    data, s P while the step trains; None without a staleness bound."""
    if staleness is None:
        return None
    return (int(staleness) + (1 if draining else 0)) * int(batch)


def engine_checksums(url: str, timeout: float = CHECKSUM_TIMEOUT_S) -> dict[str, str]:
    """{rank<r>/<tensor>: hash} of an engine's weights (SGLang /weights_checker, every TP rank)."""
    from miles.utils.audit_utils.checksum_utils import merge_inference_engine_ranks

    body = _post_json(f"{url}/weights_checker", {"action": "checksum", "allow_quant_error": False, "selector": "all"},
                      timeout)
    return merge_inference_engine_ranks(body)


def engine_weight_version(url: str, timeout: float = 30.0) -> str:
    error = None
    for path in ("/model_info", "/get_weight_version"):   # newer SGLang answers /model_info
        try:
            return str(_get_json(f"{url}{path}", timeout)["weight_version"])
        except (OSError, ValueError, KeyError) as e:
            error = e
    raise error


def set_engine_weight_version(url: str, version: str, timeout: float = 30.0) -> None:
    body = _post_json(f"{url}/update_weight_version", {"new_version": version, "abort_all_requests": False}, timeout)
    assert body.get("success", True), f"{url}/update_weight_version: {body}"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _touch(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


class PacedScheduler:
    """miles' submission scheduler (whole groups, or sample backfill) plus the staleness window and the wake-up. While
    the window is full, the producer waits for a group to finish, a wake or RECHECK_S, never for one sample: a
    finished sample frees an engine slot, not a window slot (those free at a drain)."""

    def __init__(self, inner, window_free, wake: asyncio.Event, recheck_s: float = RECHECK_S):
        self._inner = inner
        self._window_free = window_free   # callable(pending_groups) -> free window slots, None without a window
        self._wake = wake
        self._recheck_s = recheck_s
        self._full = False

    @property
    def sample_done_callback(self):
        return self._inner.sample_done_callback

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        free = self._window_free(pending_groups)
        self._full = free is not None and free <= 0
        return not self._full and self._inner.has_capacity(pending_groups=pending_groups, group_budget=group_budget)

    def on_submit(self, groups) -> None:
        self._inner.on_submit(groups)

    async def wait_for_progress(self, pendings: set) -> tuple[set, set]:
        # nothing ran since has_capacity saw the current state (the worker loop does not await in between), so a wake
        # set before now is already seen
        self._wake.clear()
        waiters = {asyncio.ensure_future(self._wake.wait())}
        progress = None
        if pendings:
            progress = asyncio.ensure_future(
                asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED) if self._full
                else self._inner.wait_for_progress(pendings))
            waiters.add(progress)
        try:
            done, _ = await asyncio.wait(waiters, timeout=self._recheck_s, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()
        if progress is not None and progress in done:
            return progress.result()
        return set(), set(pendings)


class Warmup:
    """The warm-up engine of a fresh start (module docstring). Its files under DIR (rl/rl_job.sh, rl/elastic_engine.sh
    EL_WARMUP=1):
      launched            the warm-up job's id, written when rl/rl_job.sh starts it
      engines/<job>.json  its record (rl/registry.py: "loading", then "serving" with the engine's url)
      leave               written here: the engine leaves (finishes its running requests, at most 10 min) and ends
      gone                written by the engine job once it has ended and its GPUs are free: {"job", "time", "max_gpu_mib"}
    I/O is injected: workers() -> the router's engines; add(url) and remove(url) coroutines (the router); in a thread,
    checksums(url) -> {tensor: hash}, version(url) -> str, set_version(url, version); on_change() when the routed
    engines change."""

    def __init__(self, directory, *, workers, add, remove, checksums, version, set_version, on_change,
                 clock=time.monotonic, join_s: float = WARMUP_JOIN_S, gone_s: float = WARMUP_GONE_S,
                 poll_s: float = WARMUP_POLL_S):
        self.dir = Path(directory)
        self._workers, self._add, self._remove = workers, add, remove
        self._checksums, self._version, self._set_version = checksums, version, set_version
        self._on_change, self._clock = on_change, clock
        self._join_s, self._gone_s, self._poll_s = join_s, gone_s, poll_s
        self.url: str | None = None
        self._left = False
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    def launched(self) -> bool:
        return (self.dir / "launched").exists()

    def start(self) -> None:
        if self._task is None and not self._left and self.launched():
            self._task = asyncio.ensure_future(self._join())

    def _serving_url(self) -> str | None:
        from flint.engine_registry import registry

        recs = registry.records(str(self.dir))   # fresh, serving, not leaving
        return recs[0]["engines"][0]["url"] if recs and recs[0].get("engines") else None

    def _send_leave(self, why: str) -> None:
        if not (self.dir / "leave").exists():
            _touch(self.dir / "leave", f"{time.time():.0f} {why}\n")

    async def _join(self) -> None:
        try:
            t0 = self._clock()
            while (url := self._serving_url()) is None:
                if self._left:
                    return
                if (self.dir / "gone").exists():
                    logger.warning(f"warm-up engine: ended before it served (its log: {self.dir}/engine.log)")
                    return
                if self._clock() - t0 > self._join_s:
                    logger.warning(f"warm-up engine: not serving {self._join_s / 60:.0f} min after the first drain; sent away")
                    self._send_leave("not serving in time")
                    return
                await asyncio.sleep(self._poll_s)
            # the router takes a worker in asynchronously (POST /workers answers before it is listed: w912 read an empty
            # list right after the first weight push), so the run's own engine may take a moment to show
            while not (own := [u for u in (self._workers() or []) if u != url]):
                if self._left:
                    return
                if self._clock() - t0 > self._join_s:
                    logger.warning(f"warm-up engine {url}: the router lists no engine to compare it with; sent away")
                    self._send_leave("no engine to compare with")
                    return
                await asyncio.sleep(self._poll_s)
            a, b = await asyncio.gather(asyncio.to_thread(self._checksums, url), asyncio.to_thread(self._checksums, own[0]))
            keys = set(a) | set(b)
            diff = sorted(k for k in keys if a.get(k) != b.get(k))
            if diff or not keys:
                logger.warning(f"CONSISTENCY WARNING warm-up engine {url}: {len(diff)} of {len(keys)} weight tensors "
                               f"differ from {own[0]}'s (e.g. {diff[:3]}); sent away")
                self._send_leave("weights differ from the run's own engine")
                return
            version = await asyncio.to_thread(self._version, own[0])
            await asyncio.to_thread(self._set_version, url, version)
            async with self._lock:
                if self._left:
                    return
                await self._add(url)
                self.url = url
            logger.info(f"warm-up engine {url} joined the router: all {len(keys)} weight tensors equal {own[0]}'s, "
                        f"weight version {version}")
            self._on_change()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("warm-up engine: the join failed; sent away")
            self._send_leave("the join failed")

    async def release(self) -> None:
        """The first step's data is in: out of the router, leave, and return once it is gone with its GPUs free."""
        if self._left or not self.launched():
            self._left = True
            return
        self._left = True
        t0 = self._clock()
        async with self._lock:   # a join that is adding it finishes first
            if self.url is not None:
                try:
                    await self._remove(self.url)
                except Exception:
                    logger.exception(f"warm-up engine {self.url}: taking it out of the router failed; it leaves anyway")
                self._on_change()
        self._send_leave("the first step's data is in")
        if self._task is not None and not self._task.done():
            self._task.cancel()
        while not (self.dir / "gone").exists():
            if self._clock() - t0 > self._gone_s:
                raise RuntimeError(f"the warm-up engine is not gone {self._gone_s / 60:.0f} min after its leave, so the "
                                   f"actor's GPUs may still hold its memory: see {self.dir}")
            await asyncio.sleep(self._poll_s)
        mib = (_read_json(self.dir / "gone") or {}).get("max_gpu_mib")
        if str(mib).isdigit() and int(mib) >= WARMUP_GPU_MIB:
            raise RuntimeError(f"the warm-up engine is gone, but its GPUs (the actor's) still hold up to {mib} MiB")
        logger.info(f"warm-up engine {self.url or '(never routed)'}: gone {self._clock() - t0:.0f} s after the first "
                    f"step's data came in; its GPUs hold at most {mib} MiB")


class ElasticFullyAsyncRolloutFn(FullyAsyncRolloutFn):
    def __init__(self, input):
        super().__init__(input)
        self._per_engine = int(os.environ.get("MA_ASYNC_SAMPLES_PER_ENGINE", "16"))
        self._engines = 1
        self._checked = float("-inf")
        self._budget = None
        self._draining = False
        self._wake = asyncio.Event()
        if self.args.max_weight_staleness is not None:
            assert self.args.update_weights_interval == 1, "the staleness window counts one weight push per step"
        self._scheduler = PacedScheduler(self._scheduler, self._window_free, self._wake)
        self._warmup = None
        if d := os.environ.get("MA_WARMUP_DIR"):
            from miles.backends.sglang_utils.sglang_api_client import WorkerType
            from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient, use_legacy_router_api

            ip, port = self.args.sglang_router_ip, self.args.sglang_router_port
            client, legacy = SGLangRouterApiClient(router_url=f"http://{ip}:{port}"), use_legacy_router_api(self.args)
            self._warmup = Warmup(
                d, workers=lambda: router_workers(ip, port),
                add=lambda url: client.add_worker(url, WorkerType.REGULAR, legacy),
                remove=lambda url: client.remove_worker(url, legacy),
                checksums=engine_checksums, version=engine_weight_version, set_version=set_engine_weight_version,
                on_change=self._engines_changed)

    def _engines_changed(self) -> None:
        self._checked = float("-inf")
        self._wake.set()

    def _max_in_flight_groups(self) -> int:
        now = time.monotonic()
        if now - self._checked >= RECHECK_S:
            self._checked = now
            n = routed_engines(self.args.sglang_router_ip, self.args.sglang_router_port)
            if n is not None:
                self._engines = max(1, n)
        budget = in_flight_groups(self._engines, self._per_engine, self.args.n_samples_per_prompt)
        if budget != self._budget:
            logger.info(f"elastic producer: {self._engines} routed engine(s) x {self._per_engine} episodes: "
                        f"{budget} group(s) of {self.args.n_samples_per_prompt} in flight")
            self._budget = budget
        return budget

    def _window_free(self, pending_groups: int) -> int | None:
        w = window_groups(self.args.max_weight_staleness, self.args.rollout_batch_size, self._draining)
        return None if w is None else w - pending_groups - len(getattr(self._output, "_buffer", ()))

    async def _drain(self, input):
        if self._warmup is not None:
            self._warmup.start()
        self._draining = True
        self._wake.set()
        try:
            out = await super()._drain(input)
        finally:
            self._draining = False
            self._wake.set()
        if self._warmup is not None:
            await self._warmup.release()
        return out
