"""DeviceStateAgent — snapshots timezone, locale, battery, focus mode, DND.

Consumed by other agents to adapt behavior (e.g., suppress non-urgent push
during focus mode; switch tone when it's 2am local).

Phase-3 skeleton.  The iOS side pushes a single payload to
``POST /api/device/state`` (route TODO).  This agent just normalizes the
latest snapshot into user_state.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict

from koa import valet
from koa.builtin_agents.sensing.base import SensingAgent, SensingResult

logger = logging.getLogger(__name__)


@valet(domain="sensing", expose_as_tool=False)
class DeviceStateAgent(SensingAgent):
    """Reads latest device state snapshot and surfaces key flags."""

    tools = ()
    max_turns = 1

    async def analyze(
        self,
        *,
        db,
        user_id: str,
        for_date: date,
    ) -> SensingResult:
        # TODO(phase-3): SELECT latest row from `device_state_snapshots`
        # (to be added) and compute flags like
        # ['focus_work', 'low_battery', 'tz_changed'].
        logger.debug("DeviceStateAgent.analyze stub for %s %s", user_id, for_date)
        return SensingResult(
            state={"focus_mode": None, "battery": None, "tz_changed": False},
            proposals=[],
        )
