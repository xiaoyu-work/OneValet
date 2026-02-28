"""CronExecutor — executes cron jobs through the orchestrator with backoff and concurrency."""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional, Tuple

from .delivery import CronDeliveryHandler, DeliveryResult
from .models import (
    AgentTurnPayload,
    AtSchedule,
    CronEvent,
    CronJob,
    CronRunEntry,
    SessionTarget,
    SystemEventPayload,
    WakeMode,
)
from .run_log import CronRunLog
from .schedule import compute_job_next_run_at_ms, MIN_REFIRE_GAP_MS

if TYPE_CHECKING:
    from .store import CronJobStore
    from ...orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# Exponential backoff schedule in seconds (matching OpenClaw)
BACKOFF_SCHEDULE_S = [30, 60, 300, 900, 3600]

# Clear running marker if job has been running longer than this
STUCK_JOB_TIMEOUT_MS = 2 * 60 * 60 * 1000  # 2 hours

# Default job execution timeout
DEFAULT_JOB_TIMEOUT_S = 120


def _now_ms() -> int:
    return int(time.time() * 1000)


class CronExecutor:
    """Executes cron jobs through the orchestrator.

    Supports two session targets:
    - MAIN: Injects a system event into the main session
    - ISOLATED: Dedicated orchestrator call with fresh context

    Includes exponential backoff on consecutive errors, stuck job
    detection, and concurrency control.
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        store: "CronJobStore",
        run_log: CronRunLog,
        delivery: CronDeliveryHandler,
        on_event: Optional[callable] = None,
    ):
        self._orchestrator = orchestrator
        self._store = store
        self._run_log = run_log
        self._delivery = delivery
        self._on_event = on_event

    async def execute(self, job: CronJob) -> CronRunEntry:
        """Execute a cron job, handling concurrency, backoff, and delivery."""
        now = _now_ms()

        # Guard: skip if already running
        if job.state.running_at_ms is not None:
            logger.warning(f"Job {job.id} already running, skipping")
            return CronRunEntry(
                ts=now, job_id=job.id, action="finished", status="skipped",
                summary="Already running",
            )

        # Mark as running
        job.state.running_at_ms = now
        await self._store.save()

        # Emit started event
        self._emit(CronEvent(job_id=job.id, action="started", run_at_ms=now))

        # Execute
        start_ms = now
        status = "ok"
        error_msg: Optional[str] = None
        summary: Optional[str] = None

        try:
            timeout = self._resolve_timeout(job)
            summary, error_msg = await asyncio.wait_for(
                self._execute_core(job),
                timeout=timeout,
            )
            if error_msg:
                status = "error"
        except asyncio.TimeoutError:
            status = "error"
            error_msg = f"Job timed out after {self._resolve_timeout(job)}s"
            logger.warning(f"Job {job.id} timed out")
        except Exception as e:
            status = "error"
            error_msg = str(e)
            logger.error(f"Job {job.id} execution failed: {e}")

        end_ms = _now_ms()
        duration_ms = end_ms - start_ms

        # Clear running marker
        job.state.running_at_ms = None

        # Apply result to job state
        self._apply_result(job, status, error_msg, duration_ms, end_ms)

        # Delivery
        delivery_result = DeliveryResult(status="not-requested")
        if status == "ok" and summary:
            try:
                run_entry_for_delivery = CronRunEntry(
                    ts=end_ms, job_id=job.id, status=status, summary=summary,
                )
                delivery_result = await self._delivery.deliver(
                    job, summary, run_entry_for_delivery,
                )
            except Exception as e:
                delivery_result = DeliveryResult(
                    delivered=False, status="not-delivered", error=str(e),
                )
                logger.warning(f"Delivery failed for job {job.id}: {e}")

        # Update delivery state
        job.state.last_delivery_status = delivery_result.status
        job.state.last_delivery_error = delivery_result.error
        job.state.last_delivered = delivery_result.delivered

        # Handle auto-deletion for one-shot jobs
        deleted = False
        if (
            status == "ok"
            and job.delete_after_run
            and isinstance(job.schedule, AtSchedule)
        ):
            self._store.remove(job.id)
            deleted = True
            logger.info(f"Auto-deleted one-shot job {job.id}")

        # Compute next run (if not deleted)
        if not deleted:
            next_run = compute_job_next_run_at_ms(job, end_ms)
            # Apply MIN_REFIRE_GAP for cron schedules
            if next_run is not None and next_run - end_ms < MIN_REFIRE_GAP_MS:
                next_run = end_ms + MIN_REFIRE_GAP_MS
            job.state.next_run_at_ms = next_run

        # Save
        await self._store.save()

        # Build run entry
        run_entry = CronRunEntry(
            ts=end_ms,
            job_id=job.id,
            action="finished",
            status=status,
            error=error_msg,
            summary=summary,
            delivered=delivery_result.delivered,
            delivery_status=delivery_result.status,
            delivery_error=delivery_result.error,
            run_at_ms=start_ms,
            duration_ms=duration_ms,
            next_run_at_ms=job.state.next_run_at_ms if not deleted else None,
        )

        # Append to run log
        await self._run_log.append(run_entry)

        # Emit finished event
        self._emit(CronEvent(
            job_id=job.id, action="finished",
            status=status, error=error_msg, summary=summary,
            duration_ms=duration_ms,
            delivered=delivery_result.delivered,
            delivery_status=delivery_result.status,
            next_run_at_ms=run_entry.next_run_at_ms,
        ))

        return run_entry

    async def _execute_core(self, job: CronJob) -> Tuple[Optional[str], Optional[str]]:
        """Dispatch to main or isolated execution. Returns (summary, error)."""
        if job.session_target == SessionTarget.MAIN:
            return await self._execute_main(job)
        else:
            return await self._execute_isolated(job)

    def _build_metadata(self, job: CronJob, **extra) -> dict:
        """Build metadata dict for orchestrator calls."""
        meta = {
            "source": "cron",
            "cron_job_id": job.id,
            "cron_job_name": job.name,
            "wake_mode": job.wake_mode.value,
            **extra,
        }
        # Signal the orchestrator to inject notify_user tool
        if job.delivery and job.delivery.conditional:
            meta["cron_conditional_delivery"] = True
        return meta

    async def _execute_main(self, job: CronJob) -> Tuple[Optional[str], Optional[str]]:
        """Main mode: inject system event into main session."""
        if not isinstance(job.payload, SystemEventPayload):
            return None, "Main session requires systemEvent payload"

        text = job.payload.text
        message = f"[Cron: {job.name}] {text}"

        try:
            result = await self._orchestrator.handle_message(
                tenant_id=job.user_id,
                message=message,
                metadata=self._build_metadata(job),
            )
            return self._extract_summary(job, result, text), None
        except Exception as e:
            return None, str(e)

    async def _execute_isolated(self, job: CronJob) -> Tuple[Optional[str], Optional[str]]:
        """Isolated mode: dedicated orchestrator call with fresh context."""
        if not isinstance(job.payload, AgentTurnPayload):
            return None, "Isolated session requires agentTurn payload"

        message = job.payload.message

        try:
            result = await self._orchestrator.handle_message(
                tenant_id=job.user_id,
                message=message,
                metadata=self._build_metadata(job, isolated=True),
            )
            return self._extract_summary(job, result, ""), None
        except Exception as e:
            return None, str(e)

    def _extract_summary(self, job: CronJob, result, fallback: str) -> Optional[str]:
        """Extract summary from orchestrator result, respecting conditional delivery.

        For conditional delivery jobs, only return a summary if the agent
        explicitly called notify_user. This prevents delivery of results
        when the monitored condition was not met.
        """
        if job.delivery and job.delivery.conditional:
            # Only deliver the explicit notification message
            notification = result.metadata.get("cron_notification") if result.metadata else None
            if notification:
                return notification
            # Condition not met — return None to suppress delivery
            return None
        return result.raw_message or fallback

    def _apply_result(
        self,
        job: CronJob,
        status: str,
        error: Optional[str],
        duration_ms: int,
        end_ms: int,
    ) -> None:
        """Update job state based on execution result."""
        job.state.last_run_at_ms = end_ms
        job.state.last_run_status = status
        job.state.last_duration_ms = duration_ms

        if status == "ok":
            job.state.consecutive_errors = 0
            job.state.last_error = None
        else:
            job.state.consecutive_errors += 1
            job.state.last_error = error
            # Apply exponential backoff
            self._apply_backoff(job, end_ms)

    def _apply_backoff(self, job: CronJob, end_ms: int) -> None:
        """Apply exponential backoff to push next_run_at_ms forward."""
        idx = min(job.state.consecutive_errors - 1, len(BACKOFF_SCHEDULE_S) - 1)
        backoff_ms = BACKOFF_SCHEDULE_S[idx] * 1000
        backed_off_next = end_ms + backoff_ms

        # next_run_at should be at least backed_off_next
        natural_next = compute_job_next_run_at_ms(job, end_ms)
        if natural_next is not None:
            job.state.next_run_at_ms = max(natural_next, backed_off_next)
        else:
            job.state.next_run_at_ms = backed_off_next

        logger.info(
            f"Job {job.id} backoff: {BACKOFF_SCHEDULE_S[idx]}s "
            f"(consecutive_errors={job.state.consecutive_errors})"
        )

    def _resolve_timeout(self, job: CronJob) -> int:
        """Resolve execution timeout for a job."""
        if isinstance(job.payload, AgentTurnPayload) and job.payload.timeout_seconds:
            return job.payload.timeout_seconds
        return DEFAULT_JOB_TIMEOUT_S

    def _emit(self, event: CronEvent) -> None:
        """Emit a cron event if handler is registered."""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.debug(f"Event handler error: {e}")

    def clear_stuck_jobs(self, jobs: list) -> int:
        """Detect and reset stuck jobs (running > 2h). Returns count cleared."""
        now = _now_ms()
        count = 0
        for job in jobs:
            if job.state.running_at_ms is not None:
                if now - job.state.running_at_ms > STUCK_JOB_TIMEOUT_MS:
                    logger.warning(f"Clearing stuck running marker on job {job.id}")
                    job.state.running_at_ms = None
                    count += 1
        return count
