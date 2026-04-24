"""Cron seed — idempotent registration of per-user sensing/reflection jobs.

Call ``ensure_sensing_cron_jobs(cron_service, user_id, tz)`` during user
onboarding (and optionally from a periodic reconciler).  Re-running is safe:
we look up existing jobs by name+user and skip if present.

The three jobs we seed:

============================  =====================  =====================
Name                          Agent                  Default schedule (tz)
============================  =====================  =====================
sensing.daily_aggregate       DailyAggregatorAgent   30 0 * * *   (00:30)
sensing.weekly_reflection     WeeklyReflectorAgent   0 3 * * 1    (Mon 03)
sensing.memory_proactive      MemoryProactiveAgent   0 9 * * *    (09:00)
============================  =====================  =====================

Users can later edit schedules via the /cron REST endpoints; habit_discovery
may re-write them based on observed active hours.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from koa.triggers.cron.models import (
    AgentTurnPayload,
    CronJobCreate,
    CronScheduleSpec,
    SessionTarget,
    WakeMode,
)

logger = logging.getLogger(__name__)


_SEEDS = [
    {
        "name": "sensing.daily_aggregate",
        "description": "Nightly rollup of messages/health/motion into daily_logs.",
        "agent_id": "DailyAggregatorAgent",
        "expr": "30 0 * * *",
        "message": "Run DailyAggregatorAgent for the prior day.",
    },
    {
        "name": "sensing.weekly_reflection",
        "description": "Weekly LLM reflection; extracts episodes + long-term facts.",
        "agent_id": "WeeklyReflectorAgent",
        "expr": "0 3 * * 1",  # Monday 03:00 local
        "message": "Run WeeklyReflectorAgent over the previous Mon-Sun.",
    },
    {
        "name": "sensing.memory_proactive",
        "description": "Daily memory-driven proactive check (anniversaries, streaks).",
        "agent_id": "MemoryProactiveAgent",
        "expr": "0 9 * * *",
        "message": "Run MemoryProactiveAgent for today.",
    },
]


async def ensure_sensing_cron_jobs(
    cron_service,
    user_id: str,
    *,
    tz: Optional[str] = None,
) -> List[str]:
    """Create missing sensing/reflection cron jobs for a user.

    Returns the names of jobs that were newly created (empty list if the
    user already had all of them).
    """
    if not cron_service or not user_id:
        return []

    try:
        existing = cron_service.list_jobs(user_id=user_id, include_disabled=True)
    except Exception as e:
        logger.warning("list_jobs failed for %s: %s", user_id, e)
        existing = []
    existing_names = {j.name for j in existing}

    created: List[str] = []
    for seed in _SEEDS:
        if seed["name"] in existing_names:
            continue
        job = CronJobCreate(
            name=seed["name"],
            description=seed["description"],
            user_id=user_id,
            agent_id=seed["agent_id"],
            schedule=CronScheduleSpec(expr=seed["expr"], tz=tz or None),
            session_target=SessionTarget.ISOLATED,
            wake_mode=WakeMode.NEXT_HEARTBEAT,
            payload=AgentTurnPayload(message=seed["message"]),
            enabled=True,
            max_concurrent_runs=1,
        )
        try:
            await cron_service.add(job)
            created.append(seed["name"])
            logger.info("Seeded cron job '%s' for user %s", seed["name"], user_id)
        except Exception as e:
            logger.warning("Failed to seed '%s' for %s: %s", seed["name"], user_id, e)
    return created
