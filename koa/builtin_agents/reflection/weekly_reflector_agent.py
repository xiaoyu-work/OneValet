"""WeeklyReflectorAgent — cron-driven weekly memory reflection.

Triggered by CronService at Monday 03:00 local per user. Wraps
``koa.memory.lifecycle.weekly_reflector.run_weekly_reflection`` so that the
result is fed back into the standard ``true_memory_proposals`` pipeline.
Episodes are persisted as Momex entries with ``kind=episode``.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from koa import valet
from koa.memory.lifecycle.episode_memory import EpisodeMemory
from koa.memory.lifecycle.weekly_reflector import run_weekly_reflection
from koa.result import AgentStatus
from koa.standard_agent import StandardAgent

logger = logging.getLogger(__name__)


@valet(domain="reflection", expose_as_tool=False)
class WeeklyReflectorAgent(StandardAgent):
    """Runs the weekly LLM reflection and emits long-term memory proposals."""

    tools = ()
    max_turns = 1

    async def on_running(self, msg):
        hints = self.context_hints or {}
        momex = hints.get("momex")
        user_id = hints.get("user_id") or (self.metadata or {}).get("user_id")
        if not momex or not user_id:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="nothing_to_report",
                metadata={"run_status": "skipped", "reason": "no_context"},
            )

        now, _tz = self._user_now()
        # Reflect over the previous full week ending yesterday.
        week_end = now.date() - timedelta(days=1)

        llm_client = self.llm_client
        if llm_client is None:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="nothing_to_report",
                metadata={"run_status": "skipped", "reason": "no_llm_client"},
            )

        async def _llm_call(system_prompt: str, user_prompt: str) -> str:
            resp = await llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                config={"response_format": {"type": "json_object"}},
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        episode_memory = EpisodeMemory(momex)

        reflection = await run_weekly_reflection(
            user_id,
            week_end,
            llm_call=_llm_call,
            episode_memory=episode_memory,
            existing_true_memory=hints.get("true_memory"),
        )
        if reflection is None:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="nothing_to_report",
                metadata={"run_status": "skipped", "reason": "no_data_or_llm_failed"},
            )

        result_metadata = {
            "run_status": "ok",
            "highlight": reflection.highlight,
            "episodes_written": reflection.episodes_written,
            "week_start": reflection.week_start.isoformat(),
            "week_end": reflection.week_end.isoformat(),
            "mood_trend": reflection.mood_trend,
        }
        if reflection.fact_proposals:
            result_metadata["true_memory_proposals"] = reflection.fact_proposals

        summary = (
            f"Reflected on {reflection.week_start}..{reflection.week_end}: "
            f"{reflection.episodes_written} episodes, "
            f"{len(reflection.fact_proposals)} facts, "
            f"mood={reflection.mood_trend}."
        )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=summary,
            data={
                "highlight": reflection.highlight,
                "episodes_written": reflection.episodes_written,
                "fact_proposals": reflection.fact_proposals,
            },
            metadata=result_metadata,
        )
