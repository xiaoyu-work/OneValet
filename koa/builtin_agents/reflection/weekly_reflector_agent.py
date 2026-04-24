"""WeeklyReflectorAgent — cron-driven weekly memory reflection.

Triggered by CronService at Monday 03:00 local per user.  Wraps
``koa.memory.lifecycle.weekly_reflector.run_weekly_reflection`` so that the
result is fed back into the standard ``true_memory_proposals`` pipeline
(which koi-backend's worker consumes to upsert facts).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, List, Optional

from koa import valet
from koa.standard_agent import StandardAgent
from koa.memory.lifecycle.weekly_reflector import run_weekly_reflection

logger = logging.getLogger(__name__)


@valet(domain="reflection", expose_as_tool=False)
class WeeklyReflectorAgent(StandardAgent):
    """Runs the weekly LLM reflection and emits long-term memory proposals."""

    tools = ()
    max_turns = 1

    async def on_running(self, msg):
        db = (self.context_hints or {}).get("db")
        user_id = (self.context_hints or {}).get("user_id") or (self.metadata or {}).get("user_id")
        if not db or not user_id:
            return self.make_result(status="skipped", reason="no_context")

        now, _tz = self._user_now()
        # Reflect over the *previous* full week (Mon-Sun).  If today is Mon,
        # that window is yesterday..yesterday-6.
        week_end = now.date() - timedelta(days=1)

        llm_client = self.llm_client
        if llm_client is None:
            return self.make_result(status="skipped", reason="no_llm_client")

        async def _llm_call(system_prompt: str, user_prompt: str) -> str:
            resp = await llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                config={"response_format": {"type": "json_object"}},
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        embedder = _get_embedder(self.context_hints)

        reflection = await run_weekly_reflection(
            db, user_id, week_end, llm_call=_llm_call, embedder=embedder,
        )
        if reflection is None:
            return self.make_result(status="skipped", reason="no_data_or_llm_failed")

        # Forward fact proposals through the existing pipeline.
        if reflection.fact_proposals:
            if self.metadata is None:
                self.metadata = {}
            existing = self.metadata.get("true_memory_proposals", [])
            self.metadata["true_memory_proposals"] = existing + reflection.fact_proposals

        return self.make_result(
            status="ok",
            summary=(
                f"Reflected on {reflection.week_start}..{reflection.week_end}: "
                f"{len(reflection.episodes_created)} episodes, "
                f"{len(reflection.fact_proposals)} facts, "
                f"mood={reflection.mood_trend}."
            ),
            highlight=reflection.highlight,
            episodes=reflection.episodes_created,
        )


def _get_embedder(hints: Optional[dict]):
    """Return an embedder from context_hints if one is wired, else None.

    The orchestrator may attach ``embedder`` — a zero-arg callable taking a
    string and returning Awaitable[Optional[List[float]]].  If absent, we
    still produce episodes but with NULL embeddings; recall falls back to
    keyword search until a backfill job populates them.
    """
    if not hints:
        return None
    return hints.get("embedder")
