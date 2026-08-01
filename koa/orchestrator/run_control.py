"""Per-request run control: interruption and mid-run steering.

A long task is only usable if the user can stop it and redirect it. Both are
out-of-band signals against a run that is already in flight, so they live in a
small object the caller holds a handle to rather than in the loop's arguments.

``RunControl`` is the handle for one in-flight request. ``RunControlRegistry``
maps a tenant to its current handle so an HTTP route (or any other surface) can
find the run to signal.

Cancellation is cooperative: the ReAct loop checks ``cancelled`` at its
boundaries (before an LLM call, before executing a tool batch, between turns)
and unwinds cleanly, compensating any tool call that will not run so the
transcript never carries an orphaned tool_call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RunControl:
    """Interruption + steering signals for a single in-flight request."""

    def __init__(self) -> None:
        self._cancel = asyncio.Event()
        self._cancel_reason: Optional[str] = None
        self._steering: List[str] = []

    # -- interruption --------------------------------------------------

    def request_interrupt(self, reason: str = "user stop") -> None:
        """Ask the run to stop at its next boundary. Idempotent."""
        self._cancel_reason = reason
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def cancel_reason(self) -> str:
        return self._cancel_reason or "user stop"

    async def wait_cancelled(self) -> None:
        await self._cancel.wait()

    # -- steering ------------------------------------------------------

    def queue_steering(self, text: str) -> None:
        """Queue a user message to be injected at the next turn boundary."""
        if text and text.strip():
            self._steering.append(text.strip())

    @property
    def has_steering(self) -> bool:
        return bool(self._steering)

    def drain_steering(self) -> List[str]:
        pending, self._steering = self._steering, []
        return pending

    # -- helpers -------------------------------------------------------

    async def race(self, coro, interrupted):
        """Await *coro*, resolving early with *interrupted* if the run is stopped.

        The pending task is cancelled on the interrupt path so it cannot keep
        running (or emit) after the loop has unwound.
        """
        task = asyncio.ensure_future(coro)
        cancel_wait = asyncio.ensure_future(self._cancel.wait())
        try:
            done, _ = await asyncio.wait(
                {task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                return task.result()
            task.cancel()
            return interrupted
        finally:
            cancel_wait.cancel()


class RunControlRegistry:
    """Tracks the current RunControl per tenant so surfaces can signal a run."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunControl] = {}

    def start(self, tenant_id: str) -> RunControl:
        """Register a fresh control for *tenant_id*, replacing any stale one."""
        control = RunControl()
        self._runs[tenant_id] = control
        return control

    def get(self, tenant_id: str) -> Optional[RunControl]:
        return self._runs.get(tenant_id)

    def finish(self, tenant_id: str, control: Optional[RunControl] = None) -> None:
        """Drop the control for *tenant_id* once its run has unwound.

        When *control* is given it is only dropped if it is still the current
        one, so a run finishing late cannot evict its successor.
        """
        current = self._runs.get(tenant_id)
        if current is None:
            return
        if control is not None and current is not control:
            return
        self._runs.pop(tenant_id, None)

    def request_interrupt(self, tenant_id: str, reason: str = "user stop") -> bool:
        """Signal the tenant's in-flight run. Returns False if there is none."""
        control = self._runs.get(tenant_id)
        if control is None:
            return False
        control.request_interrupt(reason)
        logger.info("[RunControl] interrupt requested for tenant=%s (%s)", tenant_id, reason)
        return True

    def queue_steering(self, tenant_id: str, text: str) -> bool:
        """Queue a steering message for the tenant's in-flight run."""
        control = self._runs.get(tenant_id)
        if control is None:
            return False
        control.queue_steering(text)
        logger.info("[RunControl] steering queued for tenant=%s", tenant_id)
        return True
