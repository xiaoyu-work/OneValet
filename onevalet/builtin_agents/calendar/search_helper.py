"""
Calendar Search Helper - Shared search logic for calendar agents

This module provides shared search functionality used by:
- CalendarAgent (query/view events)
- DeleteEventAgent (search before delete)
- Other calendar operations that need to find events
"""
import logging
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def parse_time_range(time_range_str: str, now: datetime = None) -> Tuple[datetime, datetime]:
    """
    Parse time range string to datetime range

    Args:
        time_range_str: Human-readable time range ("today", "tomorrow", "this week", etc.)
        now: Current datetime (defaults to datetime.now())

    Returns:
        Tuple of (time_min, time_max)
    """
    if now is None:
        now = datetime.now()

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    time_range_lower = time_range_str.lower().strip()

    if time_range_lower == "today":
        time_min = today_start
        time_max = today_start + timedelta(days=1)

    elif time_range_lower == "tomorrow":
        time_min = today_start + timedelta(days=1)
        time_max = today_start + timedelta(days=2)

    elif time_range_lower in ["this week", "week"]:
        days_since_monday = today_start.weekday()
        week_start = today_start - timedelta(days=days_since_monday)
        time_min = week_start
        time_max = week_start + timedelta(days=7)

    elif time_range_lower == "next week":
        days_since_monday = today_start.weekday()
        next_week_start = today_start - timedelta(days=days_since_monday) + timedelta(days=7)
        time_min = next_week_start
        time_max = next_week_start + timedelta(days=7)

    elif time_range_lower in ["this month", "month"]:
        month_start = today_start.replace(day=1)
        if today_start.month == 12:
            next_month = month_start.replace(year=today_start.year + 1, month=1)
        else:
            next_month = month_start.replace(month=today_start.month + 1)
        time_min = month_start
        time_max = next_month

    elif re.match(r"next (\d+) days?", time_range_lower):
        match = re.match(r"next (\d+) days?", time_range_lower)
        days = int(match.group(1))
        time_min = now
        time_max = now + timedelta(days=days)

    else:
        logger.info(f"Unknown time range '{time_range_str}', defaulting to next 7 days")
        time_min = now
        time_max = now + timedelta(days=7)

    logger.info(f"Parsed time range '{time_range_str}' to {time_min} - {time_max}")
    return time_min, time_max


async def search_calendar_events(
    user_id: str,
    search_query: str = None,
    time_range: str = "next 7 days",
    max_results: int = 50,
    account_hint: str = "primary"
) -> Dict[str, Any]:
    """
    Search calendar events with given criteria

    Args:
        user_id: User identifier
        search_query: Keywords to search (event title, description)
        time_range: Time range string (e.g., "today", "this week")
        max_results: Maximum events to return
        account_hint: Which calendar account to use (default "primary")

    Returns:
        Dict with:
            - success: bool
            - events: List[Dict] (if successful)
            - account: Dict (calendar account info)
            - error: str (if failed)
    """
    from onevalet.providers.calendar.resolver import CalendarAccountResolver
    from onevalet.providers.calendar.factory import CalendarProviderFactory

    try:
        account = await CalendarAccountResolver.resolve_account(user_id, account_hint)
        if not account:
            return {
                "success": False,
                "error": f"No {account_hint} calendar account found"
            }

        time_min, time_max = parse_time_range(time_range)

        account_email = account.get("account_identifier", account.get("account_name", "your calendar"))

        provider = CalendarProviderFactory.create_provider(account)
        if not provider:
            return {
                "success": False,
                "error": f"Sorry, I can't access {account_email} yet - that calendar provider isn't supported."
            }

        if not await provider.ensure_valid_token():
            return {
                "success": False,
                "error": f"I lost access to your {account_email} calendar. Could you reconnect it in settings?"
            }

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            query=search_query,
            max_results=max_results
        )

        if result.get("success"):
            events = result.get("data", [])
            logger.info(f"Found {len(events)} events matching criteria")
            return {
                "success": True,
                "events": events,
                "account": account,
                "time_min": time_min,
                "time_max": time_max
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        logger.error(f"Calendar search failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


async def find_exact_event(
    events: List[Dict],
    search_query: str,
    llm_client=None,
    user_context: str = None
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
            "reason": "No events to match"
        }

    if len(events) == 1:
        return {
            "success": True,
            "matched_events": events,
            "confidence": 1.0,
            "reason": "Only one event found"
        }

    if search_query:
        exact_matches = [
            e for e in events
            if search_query.lower() in e.get("summary", "").lower()
        ]
        if len(exact_matches) == 1:
            return {
                "success": True,
                "matched_events": exact_matches,
                "confidence": 0.95,
                "reason": f"Exact title match for '{search_query}'"
            }

    return {
        "success": True,
        "matched_events": events,
        "confidence": 0.7,
        "reason": "Multiple possible matches, showing all for user confirmation"
    }
