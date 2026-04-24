"""DailyAggregatorAgent — nightly rollup of daily_logs.

Thin @valet wrapper around ``aggregate_day`` from the memory lifecycle
package.  Scheduled to run once per user per local night (e.g. 00:30 local
time) via CronService; see ``koa.builtin_agents.reflection.cron_seed``.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from koa import valet
from koa.standard_agent import StandardAgent
from koa.memory.lifecycle.daily_log_aggregator import aggregate_day

logger = logging.getLogger(__name__)


@valet(domain="reflection", expose_as_tool=False)
class DailyAggregatorAgent(StandardAgent):
    """Rolls up yesterday's per-user signals into ``daily_logs``."""

    tools = ()
    max_turns = 1

    async def on_running(self, msg):
        db = (self.context_hints or {}).get("db")
        user_id = (self.context_hints or {}).get("user_id") or (self.metadata or {}).get("user_id")
        if not db or not user_id:
            return self.make_result(status="skipped", reason="no_context")

        now, _tz = self._user_now()
        target = now.date() - timedelta(days=1)
        try:
            row = await aggregate_day(db, user_id, target)
        except Exception as e:
            logger.exception("daily aggregate failed for %s %s: %s", user_id, target, e)
            return self.make_result(status="error", reason=str(e))

        return self.make_result(
            status="ok",
            local_date=target.isoformat(),
            message_count=(row or {}).get("message_count", 0),
            summary=f"Aggregated {target.isoformat()} for user.",
        )
