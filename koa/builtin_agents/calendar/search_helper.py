"""
Calendar Search Helper - Shared search logic for calendar agents

This module provides shared search functionality used by:
- CalendarAgent (query/view events)
- DeleteEventAgent (search before delete)
- Other calendar operations that need to find events

Time-range contract
-------------------
Callers pass an explicit ``[time_min, time_max)`` half-open window using
ISO-8601 strings (or pure ``YYYY-MM-DD`` dates, which are interpreted as
midnight in the user's timezone). The LLM is the one responsible for
resolving phrases like "last week" or "上周" into concrete dates — the
agent prompt gives it the user's current local datetime and timezone for
that purpose.

There is **no fallback** for missing or unparseable input: it raises
``ValueError`` so the LLM sees the error, self-corrects, and tries
again. Silent defaults to "next 7 days" (the previous behaviour) hid
real bugs in production.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo as ZoneInfo  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _resolve_tz(user_tz: Optional[str] = None):
    """Return a timezone object from a user timezone string (e.g. 'America/Los_Angeles')."""
    if user_tz and user_tz != "UTC":
        try:
            return ZoneInfo(user_tz)
        except Exception:
            logger.warning(f"Unknown timezone '{user_tz}', falling back to UTC")
    return timezone.utc


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_iso_datetime(
    value: Any,
    *,
    field_name: str,
    user_tz: Optional[str] = None,
) -> datetime:
    """Parse an ISO-8601 datetime string supplied by the LLM.

    Accepts:
      * full ISO-8601 datetimes with or without offset (e.g.
        ``"2026-04-27T00:00:00-07:00"``, ``"2026-04-27T07:00:00Z"``,
        ``"2026-04-27T00:00:00"``);
      * pure dates ``YYYY-MM-DD`` (interpreted as midnight in ``user_tz``).

    Naive datetimes are localised to ``user_tz`` (or UTC if absent).

    Raises ``ValueError`` with a clear, field-scoped message if the input is
    missing or unparseable. The ReAct loop surfaces the error back to the
    LLM so it can self-correct on its next turn — there is no silent
    default.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"{field_name} is required (ISO-8601 datetime, e.g. "
            f"'2026-04-27T00:00:00-07:00', or a date 'YYYY-MM-DD')."
        )

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            if _DATE_ONLY_RE.match(text):
                parsed = datetime.fromisoformat(text + "T00:00:00")
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(
                f"{field_name} must be ISO-8601 (e.g. "
                f"'2026-04-27T00:00:00-07:00') or a date 'YYYY-MM-DD'; "
                f"got {value!r} ({e})."
            )
    else:
        raise ValueError(
            f"{field_name} must be a string ISO-8601 datetime; "
            f"got {type(value).__name__}."
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_resolve_tz(user_tz))
    return parsed


def resolve_time_window(
    time_min: Any,
    time_max: Any,
    *,
    user_tz: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """Validate and parse an LLM-supplied ``[time_min, time_max)`` window.

    Both bounds are required; the half-open interval semantics match
    Google Calendar's ``timeMin``/``timeMax``: events overlapping the
    window are returned, but an event whose ``start`` equals ``time_max``
    (or whose ``end`` equals ``time_min``) is excluded.

    Raises ``ValueError`` on missing, unparseable, or non-positive
    windows.
    """
    parsed_min = parse_iso_datetime(time_min, field_name="time_min", user_tz=user_tz)
    parsed_max = parse_iso_datetime(time_max, field_name="time_max", user_tz=user_tz)
    if parsed_max <= parsed_min:
        raise ValueError(
            "time_max must be strictly after time_min "
            f"(got time_min={parsed_min.isoformat()}, time_max={parsed_max.isoformat()})."
        )
    return parsed_min, parsed_max


async def search_calendar_events(
    user_id: str,
    time_min: Any,
    time_max: Any,
    *,
    search_query: str = None,
    max_results: int = 50,
    account_hint: str = "primary",
    user_tz: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search calendar events with given criteria.

    Args:
        user_id: User identifier
        time_min: ISO-8601 inclusive start (string, datetime, or YYYY-MM-DD)
        time_max: ISO-8601 exclusive end (string, datetime, or YYYY-MM-DD)
        search_query: Keywords to search (event title, description)
        max_results: Maximum events to return
        account_hint: Which calendar account to use (default "primary")
        user_tz: IANA timezone string (e.g. "America/New_York"). Used to
            localise naive datetimes provided in ``time_min`` /
            ``time_max``.

    Raises:
        ValueError: If ``time_min`` / ``time_max`` are missing, unparseable
            or form an empty window.

    Returns:
        Dict with:
            - success: bool
            - events: List[Dict] (if successful)
            - account: Dict (calendar account info)
            - error: str (if failed)
    """
    from koa.providers.calendar.factory import CalendarProviderFactory
    from koa.providers.calendar.resolver import CalendarAccountResolver

    parsed_min, parsed_max = resolve_time_window(time_min, time_max, user_tz=user_tz)

    try:
        account = await CalendarAccountResolver.resolve_account(user_id, account_hint)
        if not account:
            return {"success": False, "error": f"No {account_hint} calendar account found"}

        account_email = account.get(
            "account_identifier", account.get("account_name", "your calendar")
        )

        provider = CalendarProviderFactory.create_provider(account)
        if not provider:
            return {
                "success": False,
                "error": f"Sorry, I can't access {account_email} yet - that calendar provider isn't supported.",
            }

        if not await provider.ensure_valid_token():
            return {
                "success": False,
                "error": f"I lost access to your {account_email} calendar. Could you reconnect it in settings?",
            }

        result = await provider.list_events(
            time_min=parsed_min, time_max=parsed_max, query=search_query, max_results=max_results
        )

        if result.get("success"):
            events = result.get("data", [])
            logger.info(
                "Calendar search [%s, %s) returned %d events",
                parsed_min.isoformat(),
                parsed_max.isoformat(),
                len(events),
            )
            return {
                "success": True,
                "events": events,
                "account": account,
                "time_min": parsed_min,
                "time_max": parsed_max,
            }
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}

    except Exception as e:
        logger.error(f"Calendar search failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def find_exact_event(
    events: List[Dict], search_query: str, llm_client=None, user_context: str = None
) -> Dict[str, Any]:
    """
    Find exact event match from search results

    Args:
        events: List of event dicts from search
        search_query: User's original query
        llm_client: LLM client for matching
        user_context: Optional context

    Returns:
        Dict with:
            - success: bool
            - matched_events: List[Dict]
            - confidence: float
    """
    if not events:
        return {
            "success": False,
            "matched_events": [],
            "confidence": 0.0,
            "reason": "No events to match",
        }

    if len(events) == 1:
        return {
            "success": True,
            "matched_events": events,
            "confidence": 1.0,
            "reason": "Only one event found",
        }

    if search_query:
        exact_matches = [e for e in events if search_query.lower() in e.get("summary", "").lower()]
        if len(exact_matches) == 1:
            return {
                "success": True,
                "matched_events": exact_matches,
                "confidence": 0.95,
                "reason": f"Exact title match for '{search_query}'",
            }

    return {
        "success": True,
        "matched_events": events,
        "confidence": 0.7,
        "reason": "Multiple possible matches, showing all for user confirmation",
    }
