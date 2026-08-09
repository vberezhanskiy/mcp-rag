"""In-process metrics — counters, histograms, last-value gauges.

Lightweight, zero-deps. Exposed via the ``metrics`` MCP tool so users can
see latency, cache hit rates, and tool-call frequencies without standing
up Prometheus. The values are also picked up by graph_stats output.

Designed to be called from a wrapper around tool dispatch:

    metrics = Metrics()
    with metrics.timer("tool.search_code"):
        ...
    metrics.inc("tool.search_code.calls")
    metrics.set_gauge("faiss.size", graph.faiss_index.ntotal)

Snapshot retrieval is O(metrics-count) — keep counter names bounded.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from threading import Lock
from typing import Deque, Iterator


class Metrics:
    """Process-local registry. Thread-safe via a single lock."""

    def __init__(self, histogram_window: int = 200) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        # Histogram backing: a bounded ring buffer of recent samples per key.
        # Mean / p50 / p95 / max are computed at snapshot time — cheap for
        # ~hundreds of samples and avoids a streaming-quantile dependency.
        self._hist: dict[str, Deque[float]] = {}
        self._gauges: dict[str, float] = {}
        self._window = histogram_window

    def inc(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + n

    def observe(self, key: str, value: float) -> None:
        with self._lock:
            buf = self._hist.get(key)
            if buf is None:
                buf = deque(maxlen=self._window)
                self._hist[key] = buf
            buf.append(float(value))

    def set_gauge(self, key: str, value: float) -> None:
        with self._lock:
            self._gauges[key] = float(value)

    @contextmanager
    def timer(self, key: str) -> Iterator[None]:
        """Context manager: measures wall-clock seconds, records as ms."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(f"{key}.ms", (time.perf_counter() - t0) * 1000.0)

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            hist_summary: dict[str, dict[str, float]] = {}
            for key, buf in self._hist.items():
                if not buf:
                    continue
                samples = sorted(buf)
                n = len(samples)
                hist_summary[key] = {
                    "count": n,
                    "mean": round(sum(samples) / n, 3),
                    "p50": round(samples[n // 2], 3),
                    "p95": round(samples[max(0, int(n * 0.95) - 1)], 3),
                    "max": round(samples[-1], 3),
                }
        return {"counters": counters, "histograms": hist_summary, "gauges": gauges}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._hist.clear()
            self._gauges.clear()


# Module-level default instance — dispatch wrappers can grab this without
# threading the registry through every call site.
default_metrics = Metrics()
