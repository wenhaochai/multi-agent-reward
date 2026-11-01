"""The fully async producer of an elastic RL run (rl/rl_inner.sh async=1): miles' FullyAsyncRolloutFn that keeps as
many episodes in flight as the engines the router serves can take, MA_ASYNC_SAMPLES_PER_ENGINE (default 16) per
routed engine, in whole GRPO groups, at least one group.

miles (flint-grpo-elastic): --fully-async --rollout-function-path flint.elastic_rollout.ElasticFullyAsyncRolloutFn.
The routed engines (sgl-router GET /workers, read at most once a minute) are the run's own engine and every engine of
an engine job that has received the weights (flint/engine_registry.py). An engine that joins raises the budget at the
next completed episode; one that leaves lowers it, and the groups already in flight finish (their next turns go to the
other engines). A failed read keeps the last count.
"""

import json
import logging
import os
import time
import urllib.request

from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn

logger = logging.getLogger(__name__)

RECHECK_S = 60.0


def routed_engines(router_ip: str, router_port: int, timeout: float = 2.0) -> int | None:
    """How many engines the router routes to (an engine listed once per dp rank as url@rank counts once)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{router_ip}:{router_port}/workers", timeout=timeout) as r:
            urls = [w["url"].rstrip("/") for w in json.loads(r.read())["workers"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return len({u.rsplit("@", 1)[0] if u.rsplit("@", 1)[-1].isdigit() else u for u in urls})


def in_flight_groups(engines: int, samples_per_engine: int, group_size: int) -> int:
    return max(1, samples_per_engine * max(1, engines) // group_size)


class ElasticFullyAsyncRolloutFn(FullyAsyncRolloutFn):
    def __init__(self, input):
        super().__init__(input)
        self._per_engine = int(os.environ.get("MA_ASYNC_SAMPLES_PER_ENGINE", "16"))
        self._engines = 1
        self._checked = float("-inf")
        self._budget = None

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
