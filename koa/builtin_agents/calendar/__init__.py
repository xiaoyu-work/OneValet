"""
Calendar agents for Koa

Provides a unified CalendarAgent for querying, creating, updating, and deleting
calendar events, plus shared search helpers.
"""

from .agent import CalendarAgent
from .search_helper import (
    find_exact_event,
    parse_iso_datetime,
    resolve_time_window,
    search_calendar_events,
)

__all__ = [
    "CalendarAgent",
    "search_calendar_events",
    "parse_iso_datetime",
    "resolve_time_window",
    "find_exact_event",
]
