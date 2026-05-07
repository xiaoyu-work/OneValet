"""DailyAggregatorAgent — nightly rollup of a day into a Momex daily_log episode.

Thin @valet wrapper around ``aggregate_day`` from the memory lifecycle
package. Scheduled once per user per local night (e.g. 00:30 local time)
via CronService; see ``koa.builtin_agents.reflection.cron_seed``.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from koa import valet
from koa.memory.lifecycle.daily_log_aggregator import aggregate_day
from koa.memory.lifecycle.episode_memory import EpisodeMemory
from koa.result import AgentStatus
from koa.standard_agent import StandardAgent

logger = logging.getLogger(__name__)


@valet(domain="reflection", expose_as_tool=False)
class DailyAggregatorAgent(StandardAgent):
    """Rolls up yesterday's per-user signals into a Momex daily_log episode."""

    tools = ()
    max_turns = 1

    async def on_running(self, msg):
        hints = self.context_hints or {}
        db = hints.get("db")
        momex = hints.get("momex")
        user_id = hints.get("user_id") or (self.metadata or {}).get("user_id")
        if not db or not user_id:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="nothing_to_report",
                metadata={"run_status": "skipped", "reason": "no_context"},
            )

        now, tz_name = self._user_now()
        target = now.date() - timedelta(days=1)

        episode_memory = EpisodeMemory(momex) if momex is not None else None
        try:
            payload = await aggregate_day(
                db,
                user_id,
                target,
                tz_name,
                episode_memory=episode_memory,
            )
        except Exception as e:
            logger.exception("daily aggregate failed for %s %s: %s", user_id, target, e)
            return self.make_result(
                status=AgentStatus.ERROR,
                raw_message=str(e),
                error_message=str(e),
                metadata={"run_status": "error", "reason": str(e)},
            )

        summary = f"Aggregated {target.isoformat()} for user."

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=summary,
            data={
                "local_date": target.isoformat(),
                "message_count": (payload or {}).get("messages", {}).get("total", 0),
            },
            metadata={"run_status": "ok", "local_date": target.isoformat()},
        )
