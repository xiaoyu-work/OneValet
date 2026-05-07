"""User settings — generic key/value store keyed by ``user_id``.

Used today for:

* ``two_way_calendar_sync`` — gates reverse edits/deletes on Google or
  EventKit calendar rows initiated from the iOS app. Default ``False``.

The store is intentionally generic so future per-user UI prefs can land
without another migration.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


KEY_TWO_WAY_CALENDAR_SYNC = "two_way_calendar_sync"


async def get_user_setting(
    db,
    user_id: str,
    key: str,
    default: Any = None,
) -> Any:
    """Return the JSON-decoded value for ``(user_id, key)`` or ``default``."""
    row = await db.fetchrow(
        "SELECT value FROM tenant_default.user_settings "
        "WHERE user_id = $1 AND key = $2",
        user_id,
        key,
    )
    if row is None:
        return default
    raw = row["value"]
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default
    return raw


async def set_user_setting(
    db,
    user_id: str,
    key: str,
    value: Any,
) -> None:
    """Upsert ``(user_id, key) -> value`` (JSON-encoded)."""
    await db.execute(
        """
        INSERT INTO tenant_default.user_settings (user_id, key, value, updated_at)
        VALUES ($1, $2, $3::jsonb, NOW())
        ON CONFLICT (user_id, key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = NOW()
        """,
        user_id,
        key,
        json.dumps(value),
    )


async def is_two_way_sync_enabled(db, user_id: str) -> bool:
    """Convenience: ``two_way_calendar_sync`` toggle, default ``False``."""
    val = await get_user_setting(
        db, user_id, KEY_TWO_WAY_CALENDAR_SYNC, default=False
    )
    return bool(val)


__all__ = [
    "KEY_TWO_WAY_CALENDAR_SYNC",
    "get_user_setting",
    "set_user_setting",
    "is_two_way_sync_enabled",
]
