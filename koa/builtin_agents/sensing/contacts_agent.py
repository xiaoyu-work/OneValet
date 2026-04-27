"""ContactsAgent — skeleton.

TODO (Phase 1 follow-up):
  * Promote ``device_contacts`` rows into ``entities`` (entity_type='person').
  * Extract birthdays into ``important_dates`` (existing table).
  * Resolve "Mom/Dad/Jay" references in messages → contact.fingerprint.
  * Surface relationship annotations into user_profile.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from koa import valet

from .base import SensingAgent, SensingResult


@valet(domain="sensing", expose_as_tool=False)
class ContactsAgent(SensingAgent):
    """Promotes device contacts into the entity graph (skeleton)."""

    SOURCE_TABLE = "device_contacts"

    async def analyze(self, db: Any, user_id: str, local_date: date, tz_name: str) -> SensingResult:
        # TODO: diff device_contacts vs entities, upsert into entities.
        return SensingResult(notes="ContactsAgent stub: no-op")
