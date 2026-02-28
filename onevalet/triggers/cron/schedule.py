"""Cron schedule computation — at/every/cron with timezone and stagger."""

import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from typing import List, Optional

from .models import (
    AtSchedule,
    CronJob,
    CronScheduleSpec,
    EverySchedule,
    Schedule,
)

logger = logging.getLogger(__name__)

# Safety gap after cron fire to prevent spin-loops (matching OpenClaw)
MIN_REFIRE_GAP_MS = 2_000


def now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)


def _parse_iso_to_ms(iso_str: str) -> Optional[int]:
    """Parse an ISO 8601 datetime string to milliseconds since epoch."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _resolve_timezone(tz: Optional[str]) -> Optional[object]:
    """Resolve an IANA timezone string to a tzinfo object."""
    if not tz:
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz)
    except ImportError:
        try:
            import pytz
            return pytz.timezone(tz)
        except ImportError:
            logger.warning(f"No timezone library available for tz={tz}")
            return None


# ---------------------------------------------------------------------------
# Core schedule computation
# ---------------------------------------------------------------------------

def compute_next_run_at_ms(schedule: Schedule, now_ms_val: int) -> Optional[int]:
    """Compute the next run time in ms for a schedule, given current time."""

    if isinstance(schedule, AtSchedule):
        at_ms = _parse_iso_to_ms(schedule.at)
        if at_ms is None:
            return None
        return at_ms if at_ms > now_ms_val else None

    if isinstance(schedule, EverySchedule):
        every = max(1, schedule.every_ms)
        anchor = schedule.anchor_ms if schedule.anchor_ms is not None else now_ms_val
        anchor = max(0, anchor)
        if now_ms_val < anchor:
            return anchor
        elapsed = now_ms_val - anchor
        steps = max(1, math.ceil((elapsed + 1) / every))
        return anchor + steps * every

    if isinstance(schedule, CronScheduleSpec):
        return _compute_cron_next_ms(schedule.expr, schedule.tz, now_ms_val)

    return None


def _compute_cron_next_ms(expr: str, tz: Optional[str], now_ms_val: int) -> Optional[int]:
    """Compute next cron fire time using croniter."""
    try:
        from croniter import croniter
    except ImportError:
        logger.warning("croniter not installed, cron schedules disabled")
        return None

    try:
        tzinfo = _resolve_timezone(tz)
        now_dt = datetime.fromtimestamp(now_ms_val / 1000, tz=timezone.utc)
        if tzinfo:
            now_dt = now_dt.astimezone(tzinfo)

        cron = croniter(expr, now_dt)
        next_dt = cron.get_next(datetime)
        next_ms = int(next_dt.timestamp() * 1000)

        # Guard: if croniter returned same or past time, advance 1 second
        if next_ms <= now_ms_val:
            next_second_ms = (now_ms_val // 1000) * 1000 + 1000
            now_dt2 = datetime.fromtimestamp(next_second_ms / 1000, tz=timezone.utc)
            if tzinfo:
                now_dt2 = now_dt2.astimezone(tzinfo)
            cron2 = croniter(expr, now_dt2)
            next_dt2 = cron2.get_next(datetime)
            next_ms = int(next_dt2.timestamp() * 1000)

        return next_ms
    except Exception as e:
        logger.warning(f"Cron schedule computation failed for '{expr}': {e}")
        return None


# ---------------------------------------------------------------------------
# Stagger computation
# ---------------------------------------------------------------------------

def compute_stagger_offset_ms(job_id: str, stagger_ms: int) -> int:
    """Compute a deterministic stagger offset using SHA256(job_id) % stagger_ms."""
    if stagger_ms <= 0:
        return 0
    digest = hashlib.sha256(job_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], byteorder="big")
    return value % stagger_ms


def compute_staggered_cron_next_ms(job: CronJob, now_ms_val: int) -> Optional[int]:
    """Compute next cron fire with deterministic stagger applied."""
    schedule = job.schedule
    if not isinstance(schedule, CronScheduleSpec):
        return compute_next_run_at_ms(schedule, now_ms_val)

    stagger_ms = schedule.stagger_ms or 0
    offset_ms = compute_stagger_offset_ms(job.id, stagger_ms)

    if offset_ms <= 0:
        return compute_next_run_at_ms(schedule, now_ms_val)

    # Shift cursor backwards by offset, find base fire, add offset
    cursor_ms = max(0, now_ms_val - offset_ms)
    for _ in range(4):
        base_next = _compute_cron_next_ms(schedule.expr, schedule.tz, cursor_ms)
        if base_next is None:
            return None
        shifted = base_next + offset_ms
        if shifted > now_ms_val:
            return shifted
        # Move cursor forward for retry
        cursor_ms = max(cursor_ms + 1, base_next + 1000)

    return None


# ---------------------------------------------------------------------------
# Job-level computation
# ---------------------------------------------------------------------------

def compute_job_next_run_at_ms(job: CronJob, now_ms_val: int) -> Optional[int]:
    """Compute next run time for a job, including stagger and fast-path for 'every'."""
    schedule = job.schedule

    # "every" fast path: if we ran recently, next = last_run + interval
    if isinstance(schedule, EverySchedule):
        every = max(1, schedule.every_ms)
        last = job.state.last_run_at_ms
        if last is not None and last > now_ms_val - every:
            return last + every
        return compute_next_run_at_ms(schedule, now_ms_val)

    # "at" — skip if already ran successfully
    if isinstance(schedule, AtSchedule):
        if job.state.last_run_status == "ok" and job.state.last_run_at_ms:
            return None
        return compute_next_run_at_ms(schedule, now_ms_val)

    # "cron" — use stagger
    if isinstance(schedule, CronScheduleSpec):
        result = compute_staggered_cron_next_ms(job, now_ms_val)
        if result is None:
            # Retry from next whole second
            next_second = (now_ms_val // 1000) * 1000 + 1000
            result = compute_staggered_cron_next_ms(job, next_second)
        return result

    return None


def recompute_next_runs(jobs: List[CronJob], now_ms_val: Optional[int] = None) -> None:
    """Bulk recompute next_run_at_ms for jobs that need it.

    Only fills missing or already-past values to avoid silently advancing
    past-due jobs without execution.
    """
    if now_ms_val is None:
        now_ms_val = now_ms()

    for job in jobs:
        if not job.enabled:
            continue
        nra = job.state.next_run_at_ms
        if nra is None or nra <= now_ms_val:
            try:
                job.state.next_run_at_ms = compute_job_next_run_at_ms(job, now_ms_val)
                job.state.schedule_error_count = 0
            except Exception as e:
                job.state.schedule_error_count += 1
                logger.warning(f"Schedule computation failed for job {job.id}: {e}")
                if job.state.schedule_error_count >= 3:
                    job.enabled = False
                    logger.error(f"Job {job.id} auto-disabled after {job.state.schedule_error_count} schedule errors")
