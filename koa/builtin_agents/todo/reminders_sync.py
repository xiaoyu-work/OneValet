"""Reminders sync helpers for TodoAgent.

Two-way bridge between koi's internal todos and the user's local iOS
Reminders (stored in ``local_reminders`` by the eventkit route).

Phase-3 stub.  Real implementation must:
  1. Pull new/changed reminders from local_reminders (since last_sync_at).
  2. Merge with koi todos by external_id ('eventkit:<reminder_id>').
  3. Push koi-originated todos back to the device via EventKitModule save.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


async def pull_reminders(
    db,
    user_id: str,
    *,
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return local reminders modified since ``since`` (or all if None)."""
    q = """SELECT reminder_id, title, notes, due_at, completed_at, list_name,
                  priority, updated_at
           FROM tenant_default.local_reminders
           WHERE user_id = $1"""
    args: List[Any] = [user_id]
    if since is not None:
        q += " AND updated_at >= $2"
        args.append(since)
    q += " ORDER BY updated_at DESC LIMIT 500"
    rows = await db.fetch(q, *args)
    return [dict(r) for r in rows]


async def push_koi_todo_to_reminder(
    *,
    title: str,
    due: Optional[datetime],
    notes: Optional[str],
    list_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Contract for TodoAgent: emit a payload the frontend turns into an
    EventKitModule.saveReminder call.  We can't hit EventKit from Python —
    the iOS app must do it — so we return a job descriptor the worker queues
    onto the per-user event bus.
    """
    return {
        "kind": "eventkit_save_reminder",
        "title": title,
        "due_at": due.astimezone(timezone.utc).isoformat() if due else None,
        "notes": notes,
        "list": list_name,
    }


# TODO(phase-3): wire `pull_reminders` into TodoAgent's context builder and
# expose a merge strategy (source-of-truth rules on conflict).
