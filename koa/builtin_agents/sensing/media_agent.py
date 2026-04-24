"""MediaAgent — consumes Photos metadata (EXIF + GPS + album) to feed
``episode_store`` with visual-anchored episodes.

Runs nightly: scans the last 24h of photo metadata, clusters by time/place,
proposes episode drafts (e.g., "Saturday hike at Mt. Tam — 47 photos").

Phase-3 skeleton — the iOS side (photos metadata upgrade / PhotoKit read)
is separately gated on Photos usage permission.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List

from koa import valet
from koa.builtin_agents.sensing.base import SensingAgent, SensingResult

logger = logging.getLogger(__name__)


@valet(domain="sensing", expose_as_tool=False)
class MediaAgent(SensingAgent):
    """Photos-metadata sensing agent (Phase 3 skeleton)."""

    tools = ()
    max_turns = 1

    async def analyze(
        self,
        *,
        db,
        user_id: str,
        for_date: date,
    ) -> SensingResult:
        # TODO(phase-3): query `photo_metadata` table (to be added) for
        # items where captured_at::date = for_date, cluster by
        # (time-window 30min, geo-cell 500m), propose episodes for clusters
        # with >=10 photos or >=2 distinct faces.
        logger.debug("MediaAgent.analyze stub for %s %s", user_id, for_date)
        return SensingResult(
            state={"last_scanned_date": for_date.isoformat()},
            proposals=[],
        )
