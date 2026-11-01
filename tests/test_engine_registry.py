"""flint.engine_registry: engine-job records -> extra miles cells. Run in miles-v0.1.3.sif with miles-elastic on the path:
python3 -m pytest -q tests/test_engine_registry.py
"""
import asyncio
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flint import engine_registry as er   # noqa: E402

reg = er.registry
ARGS = Namespace(rollout_num_gpus_per_engine=4, actor_num_nodes=1, actor_num_gpus_per_node=8, sglang_api_key=None)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _provider(run, clock, health):
    async def probe(url):
        return health.get(url, True)
    return er.RegistryEngineProvider(ARGS, run=str(run), poll_s=0.01, probe=probe, clock=clock)


def _register(run, job, urls, clock, tp=4, state="serving"):
    engines = [reg.parse_engine(f"{u},{tp},{':'.join(str(g) for g in range(i * tp, (i + 1) * tp))}") for i, u in enumerate(urls)]
    reg.register(str(run), job, "node", engines, state=state, now=clock())


def _cells(p):
    return asyncio.run(p.list_cells())


def test_serving_engines_become_cells_past_the_colocated_gpus(tmp_path):
    """Each serving engine of a job is one cell, after the actor's GPUs, with the metadata miles needs."""
    clock = Clock()
    _register(tmp_path, "w1", ["172.17.0.1:20040", "172.17.0.1:20140"], clock)
    _register(tmp_path, "w2", ["172.17.0.2:21040"], clock, state="loading")
    p = _provider(tmp_path, clock, {})
    cells = _cells(p)
    assert sorted(cells) == ["elastic-engine-00000", "elastic-engine-00001"]
    metas = [cells[c].meta for c in sorted(cells)]
    assert [m["gpu_offset"] for m in metas] == [8, 16]
    assert all(m["needs_offload"] is False and m["update_weights"] is True and m["num_gpus_per_engine"] == 4 for m in metas)
    assert cells["elastic-engine-00000"].workers_hash == "w1|http://172.17.0.1:20040"
    addrs = asyncio.run(p.get_addrs("elastic-engine-00001-00000"))
    assert addrs["primary"].host == "172.17.0.1" and addrs["primary"].port == 20140
    (infos,) = p.get_worker_infos(cell_ids=["elastic-engine-00000"])
    assert infos[0].name == "elastic-engine-00000-00000"


def test_stale_leaving_or_gone_records_are_withdrawn_and_ids_are_not_reused(tmp_path):
    """An engine leaves the cell set when its heartbeat is stale, it says leaving, or its file is gone; a restarted
    engine at the same address is a new cell."""
    clock = Clock()
    _register(tmp_path, "w1", ["172.17.0.1:20040"], clock)
    _register(tmp_path, "w2", ["172.17.0.2:20040"], clock)
    _register(tmp_path, "w3", ["172.17.0.3:20040"], clock)
    p = _provider(tmp_path, clock, {})
    assert len(_cells(p)) == 3
    clock.t += reg.FRESH + 1                       # w1 stops beating; w2 and w3 beat
    for j in ("w2", "w3"):
        reg.beat(reg.engine_file(str(tmp_path), j), now=clock())
    f = reg.engine_file(str(tmp_path), "w2")       # w2 announces it is leaving
    rec = json.load(open(f)); rec["leaving"] = clock(); json.dump(rec, open(f, "w"))
    assert sorted(c.workers_hash for c in _cells(p).values()) == ["w3|http://172.17.0.3:20040"]
    reg.deregister(str(tmp_path), "w3")
    assert _cells(p) == {}
    # miles still asks about a cell while it removes it
    assert p.get_worker_infos(cell_ids=["elastic-engine-00000"])[0][0].name == "elastic-engine-00000-00000"
    _register(tmp_path, "w4", ["172.17.0.1:20040"], clock)   # a new job at w1's old address
    assert list(_cells(p)) == ["elastic-engine-00003"]


def test_health_failures_withdraw_after_the_limit_and_a_new_engine_needs_one_answer(tmp_path):
    clock = Clock()
    _register(tmp_path, "w1", ["172.17.0.1:20040"], clock)
    health = {"http://172.17.0.1:20040": False}
    p = _provider(tmp_path, clock, health)
    assert _cells(p) == {}                          # not announced before one answer
    health["http://172.17.0.1:20040"] = True
    assert len(_cells(p)) == 1
    health["http://172.17.0.1:20040"] = False
    for _ in range(er.FAIL_LIMIT - 1):
        assert len(_cells(p)) == 1                  # tolerated
    assert _cells(p) == {}                          # the FAIL_LIMIT-th failure in a row withdraws it


def test_engines_of_another_tp_size_are_skipped(tmp_path):
    clock = Clock()
    _register(tmp_path, "w1", ["172.17.0.1:20040"], clock, tp=8)
    _register(tmp_path, "w2", ["172.17.0.2:20040"], clock, tp=4)
    assert [c.workers_hash for c in _cells(_provider(tmp_path, clock, {})).values()] == ["w2|http://172.17.0.2:20040"]


def test_an_unlistable_registry_keeps_the_last_set(tmp_path, monkeypatch):
    clock = Clock()
    _register(tmp_path, "w1", ["172.17.0.1:20040"], clock)
    p = _provider(tmp_path, clock, {})
    assert len(_cells(p)) == 1
    real = os.listdir

    def stalled(path):
        if str(path).endswith("engines"):
            raise OSError(5, "Input/output error")
        return real(path)

    monkeypatch.setattr(er.os, "listdir", stalled)
    assert len(_cells(p)) == 1
    monkeypatch.setattr(er.os, "listdir", real)
    assert not (tmp_path / "missing").exists()
    assert _cells(_provider(tmp_path / "missing", clock, {})) == {}   # no directory yet: no engine


def test_the_polling_loop_adds_and_removes_cells(tmp_path):
    """Through miles' PollingReconcileLoop: a join, then a leave, reach the reconcile callback."""
    clock = Clock()
    p = _provider(tmp_path, clock, {})
    events = []

    async def reconcile(cell_id, info):
        events.append((cell_id, info is not None))

    async def run():
        stop = await p.watch_cells(reconcile)
        _register(tmp_path, "w1", ["172.17.0.1:20040"], clock)
        await asyncio.sleep(0.1)
        reg.deregister(str(tmp_path), "w1")
        await asyncio.sleep(0.1)
        await stop()

    asyncio.run(run())
    assert events == [("elastic-engine-00000", True), ("elastic-engine-00000", False)]


def test_the_provider_needs_the_registry_directory(monkeypatch):
    monkeypatch.delenv("MA_ENGINE_REGISTRY", raising=False)
    with pytest.raises(AssertionError, match="MA_ENGINE_REGISTRY"):
        er.provider(ARGS)
