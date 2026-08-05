"""Async task lifecycle tracking.

The orchestrator and pool spawn many fire-and-forget tasks via
``asyncio.create_task``.  Without a registry these tasks:

- swallow exceptions (log-only via the "never-awaited" warning),
- prevent graceful shutdown because nothing cancels them,
- leak references if the creating object is GC'd.

:class:`TaskRegistry` addresses all three.  It wraps ``create_task`` with a
done-callback that logs unhandled exceptions, tracks every task in a set,
and exposes :meth:`cancel_all` for shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Optional, Set

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Track background tasks and cancel them on shutdown.

    Args:
        name: Label used in log messages and metric labels.
    """

    def __init__(self, name: str = "koa") -> None:
        self._name = name
        self._tasks: Set[asyncio.Task] = set()
        self._closed = False

    def create_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: Optional[str] = None,
    ) -> asyncio.Task:
        """Create and track a task; log exceptions on completion."""
        if self._closed:
            # Don't silently drop work after shutdown — surface it.
            raise RuntimeError(f"TaskRegistry[{self._name}] is closed; cannot schedule {name!r}")
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "[TaskRegistry:%s] Task %r raised unhandled exception: %r",
                self._name,
                task.get_name(),
                exc,
                exc_info=exc,
            )
            try:
                from .metrics import counter

                counter("koa_task_exception_total", {"registry": self._name}, 1)
            except Exception:
                pass

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    @property
    def closed(self) -> bool:
        return self._closed

    async def cancel_all(self, timeout: float = 5.0) -> None:
        """Cancel every tracked task and wait up to ``timeout`` seconds.

        Idempotent: safe to call multiple times.  After this returns the
        registry is marked closed and further ``create_task`` calls raise.
        """
        self._closed = True
        tasks = list(self._tasks)
        if not tasks:
            return
        logger.info("[TaskRegistry:%s] Cancelling %d tasks", self._name, len(tasks))
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[TaskRegistry:%s] Timeout cancelling %d tasks",
                self._name,
                len(self._tasks),
            )


# Module-level default registry — use for generic fire-and-forget work that
# is not tied to a specific orchestrator lifecycle.
_default = TaskRegistry("default")


def get_task_registry() -> TaskRegistry:
    """Return the process-wide default task registry."""
    return _default
