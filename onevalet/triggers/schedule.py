"""Schedule trigger utilities — cron, interval, one-time."""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def compute_next_schedule(
    cron: Optional[str] = None,
    interval_minutes: Optional[int] = None,
    run_at: Optional[str] = None,
    last_run: Optional[datetime] = None,
) -> Optional[datetime]:
    """Compute next run time for a schedule trigger.

    Args:
        cron: Cron expression (e.g. "0 8 * * *")
        interval_minutes: Interval in minutes
        run_at: ISO datetime for one-time triggers
        last_run: Last execution time

    Returns:
        Next run datetime, or None if no more runs
    """
    now = datetime.now()

    if run_at:
        dt = datetime.fromisoformat(run_at) if isinstance(run_at, str) else run_at
        return dt if dt > now else None

    if interval_minutes:
        base = last_run or now
        return base + timedelta(minutes=interval_minutes)

    if cron:
        try:
            from croniter import croniter

            return croniter(cron, now).get_next(datetime)
        except ImportError:
            logger.warning("croniter not installed; cron triggers unavailable")
            return None
        except Exception as e:
            logger.warning(f"Invalid cron expression '{cron}': {e}")
            return None

    return None


def is_due(next_run: Optional[datetime]) -> bool:
    """Check if a scheduled time has arrived."""
    if next_run is None:
        return False
    return datetime.now() >= next_run
