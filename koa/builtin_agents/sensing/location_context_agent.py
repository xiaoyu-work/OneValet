"""LocationContextAgent — skeleton.

TODO (Phase 1 follow-up):
  * Combine ``user_locations`` (existing geofence) + motion_segments +
    eventkit.location to classify each local hour into {home, work, commute,
    gym, elsewhere}.
  * Populate ``user_state.primary_location`` with the modal category.
  * Detect routine breaks ("user was at 'elsewhere' all day Tuesday;
    usually at work") → propose a fact.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from koa import valet
from .base import SensingAgent, SensingResult


@valet(domain="sensing", expose_as_tool=False)
class LocationContextAgent(SensingAgent):
    """Combines geofence + motion to classify daily location pattern (skeleton)."""

    SOURCE_TABLE = "user_locations"

    async def analyze(self, db: Any, user_id: str, local_date: date, tz_name: str) -> SensingResult:
        # TODO: modal location classification.
        return SensingResult(notes="LocationContextAgent stub: no-op")
