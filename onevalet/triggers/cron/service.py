"""CronService — timer-based scheduler with CRUD API, matching OpenClaw's CronService."""

import asyncio
import logging
import time
from typing import Callable, List, Optional

from .executor import CronExecutor
from .models import (
    CronEvent,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRunEntry,
)
from .run_log import CronRunLog
from .schedule import compute_job_next_run_at_ms, now_ms, recompute_next_runs
from .store import CronJobStore

logger = logging.getLogger(__name__)

# Maximum sleep interval before checking again
MAX_SLEEP_S = 60.0

# Minimum sleep to avoid busy-spin
MIN_SLEEP_S = 0.1


class CronService:
    """Main cron scheduler.

    Timer-based: sleeps until the next job is due (capped at 60s),
    then fires all due jobs. Can be woken immediately when jobs are
    added, updated, or removed.
    """

    def __init__(
        self,
        store: CronJobStore,
        executor: CronExecutor,
        run_log: Optional[CronRunLog] = None,
        on_event: Optional[Callable[[CronEvent], None]] = None,
    ):
        self._store = store
        self._executor = executor
        self._run_log = run_log
        self._on_event = on_event
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the cron scheduler."""
        if self._running:
            return

        # Load store if not loaded
        await self._store.load()

        # Clear stale running markers
        all_jobs = self._store.list(include_disabled=True)
        cleared = self._executor.clear_stuck_jobs(all_jobs)
        if cleared:
            logger.info(f"Cleared {cleared} stuck running markers on startup")
            await self._store.save()

        # Recompute schedules
        recompute_next_runs(all_jobs)
        await self._store.save()

        # Start timer loop
        self._running = True
        self._loop_task = asyncio.create_task(self._timer_loop())
        logger.info(f"CronService started ({len(all_jobs)} jobs loaded)")

    async def stop(self) -> None:
        """Stop the cron scheduler."""
        self._running = False
        self._wake.set()  # wake the loop so it can exit
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("CronService stopped")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def add(self, input: CronJobCreate) -> CronJob:
        """Create a new cron job."""
        job = input.to_job()

        # Compute initial next_run_at_ms
        job.state.next_run_at_ms = compute_job_next_run_at_ms(job, now_ms())

        self._store.add(job)
        await self._store.save()
        self._reschedule()

        self._emit(CronEvent(
            job_id=job.id, action="added",
            next_run_at_ms=job.state.next_run_at_ms,
        ))

        logger.info(f"Added cron job {job.id} ({job.name})")
        return job

    async def update(self, job_id: str, patch: CronJobPatch) -> CronJob:
        """Update an existing cron job."""
        job = self._store.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        schedule_changed = patch.schedule is not None
        enabled_changed = patch.enabled is not None

        patch.apply(job)

        # Recompute schedule if relevant fields changed
        if schedule_changed or enabled_changed:
            if job.enabled:
                job.state.next_run_at_ms = compute_job_next_run_at_ms(job, now_ms())
                job.state.schedule_error_count = 0
            else:
                job.state.next_run_at_ms = None

        self._store.update(job)
        await self._store.save()
        self._reschedule()

        self._emit(CronEvent(
            job_id=job.id, action="updated",
            next_run_at_ms=job.state.next_run_at_ms,
        ))

        logger.info(f"Updated cron job {job.id}")
        return job

    async def remove(self, job_id: str) -> bool:
        """Delete a cron job."""
        removed = self._store.remove(job_id)
        if removed:
            await self._store.save()
            self._reschedule()

            self._emit(CronEvent(job_id=job_id, action="removed"))

            # Clean up run log
            if self._run_log:
                await self._run_log.delete_log(job_id)

            logger.info(f"Removed cron job {job_id}")
        return removed

    async def run(self, job_id: str, mode: str = "force") -> CronRunEntry:
        """Manually trigger a cron job.

        Args:
            job_id: The job ID to run.
            mode: "force" to always run, "due" to only run if due.
        """
        job = self._store.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if mode == "due":
            nra = job.state.next_run_at_ms
            if nra is not None and nra > now_ms():
                return CronRunEntry(
                    ts=now_ms(), job_id=job_id, action="finished",
                    status="skipped", summary="Not yet due",
                )

        return await self._executor.execute(job)

    def get_job(self, job_id: str) -> Optional[CronJob]:
        return self._store.get(job_id)

    def list_jobs(
        self,
        user_id: Optional[str] = None,
        include_disabled: bool = False,
    ) -> List[CronJob]:
        return self._store.list(user_id=user_id, include_disabled=include_disabled)

    def find_job(self, hint: str, user_id: Optional[str] = None) -> Optional[CronJob]:
        """Find a job by ID or name hint."""
        return self._store.find_by_hint(hint, user_id)

    async def status(self) -> dict:
        """Return scheduler status summary."""
        all_jobs = self._store.list(include_disabled=True)
        enabled = [j for j in all_jobs if j.enabled]
        running = [j for j in all_jobs if j.state.running_at_ms is not None]
        next_due = self._store.get_next_due_time()

        return {
            "running": self._running,
            "total_jobs": len(all_jobs),
            "enabled_jobs": len(enabled),
            "running_jobs": len(running),
            "next_due_at_ms": next_due,
            "next_due_in_seconds": (
                max(0, (next_due - now_ms()) / 1000) if next_due else None
            ),
        }

    async def get_runs(self, job_id: str, limit: int = 20) -> List[CronRunEntry]:
        """Get run history for a job."""
        if not self._run_log:
            return []
        return await self._run_log.get_runs(job_id, limit=limit)

    # ------------------------------------------------------------------
    # Timer loop
    # ------------------------------------------------------------------

    async def _timer_loop(self) -> None:
        """Main scheduler loop: sleep until next due job, then fire."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron timer tick error: {e}")
                await asyncio.sleep(1)  # brief recovery pause

    async def _tick(self) -> None:
        """Single timer tick: compute sleep, wait, fire due jobs."""
        next_due = self._store.get_next_due_time()
        now = now_ms()

        if next_due is not None:
            delay_ms = next_due - now
            sleep_s = max(MIN_SLEEP_S, min(delay_ms / 1000, MAX_SLEEP_S))
        else:
            sleep_s = MAX_SLEEP_S

        # Sleep, but wake immediately if _wake event is set
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=sleep_s)
        except asyncio.TimeoutError:
            pass

        if not self._running:
            return

        await self._fire_due_jobs()

    async def _fire_due_jobs(self) -> None:
        """Find and execute all due jobs."""
        now = now_ms()
        due_jobs: List[CronJob] = []

        for job in self._store.list(include_disabled=False):
            if job.state.running_at_ms is not None:
                continue
            nra = job.state.next_run_at_ms
            if nra is not None and nra <= now:
                due_jobs.append(job)

        if not due_jobs:
            return

        logger.info(f"Firing {len(due_jobs)} due cron job(s)")

        # Execute concurrently (respecting per-job max_concurrent_runs)
        tasks = [self._safe_execute(job) for job in due_jobs]
        await asyncio.gather(*tasks)

    async def _safe_execute(self, job: CronJob) -> None:
        """Execute a job with error isolation."""
        try:
            await self._executor.execute(job)
        except Exception as e:
            logger.error(f"Cron job {job.id} execution error: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reschedule(self) -> None:
        """Wake the timer loop to recalculate sleep."""
        self._wake.set()

    def _emit(self, event: CronEvent) -> None:
        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.debug(f"Event handler error: {e}")
