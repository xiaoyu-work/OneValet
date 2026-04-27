"""MotionAgent — skeleton.

TODO (Phase 1 follow-up):
  * Read ``motion_segments`` for local_date, compute:
      - commute detection (long automotive segments between known locations)
      - walk/run minutes vs activity_minutes from HealthAgent (reconcile)
      - first_movement_at, last_movement_at → daily routine fingerprint
  * Write a "daily routine shape" fact when it stabilizes across a week.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from koa import valet

from .base import SensingAgent, SensingResult

logger = logging.getLogger(__name__)


@valet(domain="sensing", expose_as_tool=False)
class MotionAgent(SensingAgent):
    """Daily motion segment analysis (skeleton)."""

    SOURCE_TABLE = "motion_segments"

    async def analyze(self, db: Any, user_id: str, local_date: date, tz_name: str) -> SensingResult:
        try:
            row = await db.fetchrow(
                """SELECT COUNT(*) AS n FROM tenant_default.motion_segments
                   WHERE user_id = $1 AND started_at::date = $2""",
                user_id,
                local_date,
            )
            n = int(row["n"]) if row else 0
        except Exception as e:
            logger.debug("motion count query failed: %s", e)
            n = 0
        # TODO: real analysis; for now just surface a count.
        return SensingResult(notes=f"MotionAgent stub: {n} segments for {local_date}")
