"""Metrics abstraction with Prometheus + in-memory backends.

Callers record metrics via ``counter("name", labels, value)`` and
``observe("name", labels, value)`` without needing to know which backend
is active.  When ``prometheus_client`` is installed and
:func:`configure_metrics(prometheus=True)` is called, metrics are exported
via the Prometheus client.  Otherwise values are aggregated in a simple
thread-safe in-memory registry suitable for tests and low-volume use.

The in-memory fallback ensures that calls never raise in production even
when Prometheus is misconfigured.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class _InMemoryRegistry:
    """Thread-safe in-memory metrics store.

    Stores counters as sums and histograms as ``(count, sum, min, max)``
    tuples keyed by ``(name, frozenset(labels.items()))``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, frozenset], float] = {}
        self._hist: Dict[Tuple[str, frozenset], Tuple[int, float, float, float]] = {}

    @staticmethod
    def _key(name: str, labels: Optional[Dict[str, str]]) -> Tuple[str, frozenset]:
        lbl = frozenset((labels or {}).items())
        return (name, lbl)

    def counter(self, name: str, labels: Optional[Dict[str, str]], value: float) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, labels: Optional[Dict[str, str]], value: float) -> None:
        key = self._key(name, labels)
        with self._lock:
            prev = self._hist.get(key)
            if prev is None:
                self._hist[key] = (1, value, value, value)
            else:
                c, s, mn, mx = prev
                self._hist[key] = (c + 1, s + value, min(mn, value), max(mx, value))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": {
                    f"{n}{{{','.join(f'{k}={v}' for k, v in sorted(labels))}}}": v
                    for (n, labels), v in self._counters.items()
                },
                "histograms": {
                    f"{n}{{{','.join(f'{k}={v}' for k, v in sorted(labels))}}}": {
                        "count": c,
                        "sum": s,
                        "min": mn,
                        "max": mx,
                        "avg": s / c if c else 0.0,
                    }
                    for (n, labels), (c, s, mn, mx) in self._hist.items()
                },
            }


# Module-level state (single-process).  Multi-process Prometheus requires
# ``prometheus_client.multiprocess`` — configure externally if needed.
_inmem = _InMemoryRegistry()
_prom_enabled = False
_prom_counters: Dict[str, Any] = {}
_prom_histograms: Dict[str, Any] = {}
_enabled = False


def configure_metrics(*, enabled: bool = True, prometheus: bool = False) -> None:
    """Enable metric recording and optionally the Prometheus backend.

    Calling with ``enabled=False`` turns recording into a no-op.
    """
    global _enabled, _prom_enabled
    _enabled = enabled
    if prometheus:
        try:
            import prometheus_client  # noqa: F401

            _prom_enabled = True
        except ImportError:
            logger.warning(
                "prometheus_client not installed; metrics fall back to in-memory. "
                "Install via: pip install prometheus-client"
            )
            _prom_enabled = False
    else:
        _prom_enabled = False


def _get_prom_counter(name: str, labels: Optional[Dict[str, str]]):
    from prometheus_client import Counter

    c = _prom_counters.get(name)
    if c is None:
        label_names = sorted((labels or {}).keys())
        c = Counter(name, name, label_names)
        _prom_counters[name] = c
    return c


def _get_prom_histogram(name: str, labels: Optional[Dict[str, str]]):
    from prometheus_client import Histogram

    h = _prom_histograms.get(name)
    if h is None:
        label_names = sorted((labels or {}).keys())
        h = Histogram(name, name, label_names)
        _prom_histograms[name] = h
    return h


def counter(name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
    """Increment a counter metric."""
    if not _enabled:
        return
    try:
        if _prom_enabled:
            c = _get_prom_counter(name, labels)
            if labels:
                c.labels(**labels).inc(value)
            else:
                c.inc(value)
        _inmem.counter(name, labels, value)
    except Exception as exc:  # pragma: no cover - metrics must never raise
        logger.debug("metrics counter failed: %s", exc)


def observe(name: str, labels: Optional[Dict[str, str]] = None, value: float = 0.0) -> None:
    """Record an observation on a histogram metric."""
    if not _enabled:
        return
    try:
        if _prom_enabled:
            h = _get_prom_histogram(name, labels)
            if labels:
                h.labels(**labels).observe(value)
            else:
                h.observe(value)
        _inmem.observe(name, labels, value)
    except Exception as exc:  # pragma: no cover
        logger.debug("metrics observe failed: %s", exc)


def histogram(name: str, labels: Optional[Dict[str, str]] = None, value: float = 0.0) -> None:
    """Alias for :func:`observe`."""
    observe(name, labels, value)


def get_metrics_registry() -> _InMemoryRegistry:
    """Return the in-memory registry (always populated alongside Prometheus)."""
    return _inmem
