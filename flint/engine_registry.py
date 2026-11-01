"""Engines in other jobs as extra cells of a colocated RL run (anchored elastic RL, tmp/elastic_rl/anchored.md).

miles (flint-grpo-elastic): --extra-inference-engine-provider-path flint.engine_registry.provider, with
MA_ENGINE_REGISTRY = the run directory. The run keeps its own colocated engines and never waits for these. Each engine
of an engine job (rl/engine_job.sh) that registered in RUN/engines (the rl/registry.py layout) is one cell while
  - its record is fresh (heartbeat <= rl/registry.py FRESH, 90 s), says "serving" (the job saw /health_generate, the
    world size and a one-token /generate pass) and does not say "leaving";
  - its tensor-parallel size is the run's own (--rollout-num-gpus-per-engine): one serving shape per run;
  - its /health answers (with miles' engine environment, SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false, /health does
    not touch the scheduler, so a weight push does not make it fail). It is announced after one answer and withdrawn
    after FAIL_LIMIT failures in a row (~30 s at the 10-s poll).
miles routes a new cell only after the next weight push (the cell waits in PendingWeights), so an engine serves the
run's current weights from its first request. A withdrawn cell is unregistered from the router; its sessions' next
turns go to another engine, which prefills their history again (same tokens, the KV cache is lost).
A registry that cannot be listed (a scratch stall) keeps the last good set. Cell ids are never reused within a run, so a
restarted engine at the same address is a new cell: workers_hash = job|url.
"""

import asyncio
import importlib.util
import logging
import os
import time
from pathlib import Path

import httpx

from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.polling_reconcile_loop import PollingReconcileLoop
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

logger = logging.getLogger(__name__)

POOL_ID = "elastic-engine"
POLL_S = 10.0
FAIL_LIMIT = 3
PROBE_TIMEOUT_S = 5.0
OFFSET_STEP = 8   # cell k's gpu_offset is the actor's GPU count + 8 k: past the colocated engines, unique, stable


def _load_registry():
    """rl/registry.py of this repository (the one definition of the registry files and of a fresh heartbeat)."""
    path = Path(__file__).resolve().parent.parent / "rl" / "registry.py"
    spec = importlib.util.spec_from_file_location("ma_engine_registry_files", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


registry = _load_registry()


def provider(args, *, capability=None) -> "RegistryEngineProvider":
    run = os.environ.get("MA_ENGINE_REGISTRY")
    assert run, "flint.engine_registry needs MA_ENGINE_REGISTRY, the run directory the engine jobs register in"
    return RegistryEngineProvider(args, run=run)


async def _http_health(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S, trust_env=False) as client:
            return (await client.get(f"{url}/health")).status_code == 200
    except (httpx.HTTPError, OSError):
        return False


class RegistryEngineProvider(BaseWorkerProvider):
    def __init__(self, args, *, run: str, poll_s: float = POLL_S, probe=_http_health, clock=time.time) -> None:
        self._run = run
        self._poll_s = poll_s
        self._probe = probe
        self._clock = clock
        self._tp = int(args.rollout_num_gpus_per_engine)
        self._offset0 = int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
        self._api_key = getattr(args, "sglang_api_key", None)
        self._index: dict[tuple[str, str], int] = {}   # (job, url) -> cell index, never reused in this run
        self._fails: dict[tuple[str, str], int] = {}
        self._announced: dict[tuple[str, str], CellInfo] = {}
        # every cell and worker ever announced: miles still asks about a cell while it removes it
        self._infos: dict[str, CellInfo] = {}
        self._addrs: dict[str, NamedHostAndPorts] = {}   # worker name -> addresses
        self._last: list[tuple[str, dict]] | None = None
        self._skipped_tp: set[tuple[str, str]] = set()

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        assert worker_name in self._addrs, f"{worker_name} is not an engine of {self._run}/engines"
        return self._addrs[worker_name]

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [WorkerInfo(name=n, generation=0, self_addrs=self._addrs[n], gpu_ids=[]) for n in self._infos[c].worker_names]
            for c in cell_ids
        ]

    async def watch_cells(self, reconcile):
        return await PollingReconcileLoop(list_cells=self.list_cells, poll_interval_seconds=self._poll_s).start(reconcile)

    def _read(self) -> list[tuple[str, dict]] | None:
        """The serving engines of fresh, not-leaving records, as (job, engine); None if the directory cannot be listed."""
        d = os.path.join(self._run, "engines")
        try:
            os.listdir(d)
        except FileNotFoundError:
            return []
        except OSError:
            return None
        return [(str(r["job"]), e) for r in registry.records(self._run, now=self._clock()) for e in r.get("engines", [])]

    async def list_cells(self) -> dict[str, CellInfo]:
        engines = self._read()
        if engines is None:
            logger.warning(f"cannot list {self._run}/engines; keeping the {len(self._last or [])} engines seen last")
            engines = self._last or []
        else:
            self._last = engines
        candidates = []
        for job, e in engines:
            key = (job, e["url"])
            if int(e["tp"]) != self._tp:
                if key not in self._skipped_tp:
                    self._skipped_tp.add(key)
                    logger.warning(f"engine {e['url']} of job {job} is TP{e['tp']}, the run serves TP{self._tp}: skipped")
                continue
            candidates.append((key, e))
        alive = await asyncio.gather(*[self._probe(e["url"]) for _, e in candidates])
        present = set()
        for (key, e), ok in zip(candidates, alive, strict=True):
            present.add(key)
            self._fails[key] = 0 if ok else self._fails.get(key, 0) + 1
            if key in self._announced:
                if self._fails[key] >= FAIL_LIMIT:
                    logger.warning(f"engine {e['url']} of job {key[0]}: /health failed {FAIL_LIMIT} times; withdrawn")
                    self._withdraw(key)
            elif ok:
                self._announce(key, e)
        for key in [k for k in self._announced if k not in present]:   # record gone, stale or leaving
            logger.info(f"engine {key[1]} of job {key[0]}: no longer registered as serving; withdrawn")
            self._withdraw(key)
        return {info.cell_id: info for info in self._announced.values()}

    def _announce(self, key: tuple[str, str], e: dict) -> None:
        index = self._index.setdefault(key, len(self._index))
        cell_id = compute_cell_id(pool_id=POOL_ID, cell_index=index)
        worker = compute_worker_name(pool_id=POOL_ID, cell_index=index)
        host, port = e["url"].split("://", 1)[1].rsplit(":", 1)
        self._addrs[worker] = dict(primary=HostAndPort(host=host, port=int(port)))
        self._announced[key] = self._infos[cell_id] = CellInfo(
            cell_id=cell_id,
            pool_id=POOL_ID,
            alive=True,
            worker_names=[worker],
            workers_hash=f"{key[0]}|{key[1]}",
            meta=dict(
                model_id="default",
                worker_type="regular",
                num_gpus_per_engine=int(e["tp"]),
                gpu_offset=self._offset0 + OFFSET_STEP * index,
                sglang_api_key=self._api_key,
                needs_offload=False,
                update_weights=True,
            ),
        )
        logger.info(f"engine {e['url']} of job {key[0]} (TP{e['tp']}): cell {cell_id}")

    def _withdraw(self, key: tuple[str, str]) -> None:
        self._announced.pop(key, None)
        self._fails.pop(key, None)
