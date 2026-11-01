"""flint.elastic_rollout: the fully async producer's in-flight budget from the router's routed engines. Run in
miles-v0.1.3.sif with miles-elastic on the path: python3 -m pytest -q tests/test_elastic_rollout.py"""
import json
import sys
import threading
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
