"""
Calendar Tools — Standalone API functions for CalendarAgent's mini ReAct loop.

Extracted from CalendarAgent, CreateEventAgent, UpdateEventAgent, and DeleteEventAgent.
"""

import html
import logging
import re
from datetime import datetime, timedelta
from typing import Annotated, Dict, Optional

from onevalet.tool_decorator import tool
from onevalet.models import AgentToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

async def _get_provider(tenant_id: str):
    """Resolve the primary calendar account and return (provider, account) or (None, error_msg)."""
    from onevalet.providers.calendar.resolver import CalendarAccountResolver
    from onevalet.providers.calendar.factory import CalendarProviderFactory

    account = await CalendarAccountResolver.resolve_account(tenant_id, "primary")
    if not account:
        return None, None, "No calendar account found. Please connect one first."

    provider = CalendarProviderFactory.create_provider(account)
    if not provider:
        return None, None, "Sorry, I can't access that calendar provider yet."

    if not await provider.ensure_valid_token():
        return None, None, "I lost access to your calendar. Could you reconnect it?"

    return provider, account, None


def _format_event_time(start) -> str:
    """Format an event start time for display."""
    if not start:
        return "Unknown time"
    if isinstance(start, datetime):
        return start.strftime("%a %b %d, %I:%M %p")
    if isinstance(start, dict):
        dt_str = start.get("dateTime", start.get("date", ""))
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.strftime("%a %b %d, %I:%M %p")
        except (ValueError, AttributeError):
            return dt_str
    return str(start)


async def _parse_datetime_with_llm(time_str: str, llm_client) -> Optional[datetime]:
    """Parse a natural language time string to datetime using LLM."""
    if not llm_client:
        return None

    now = datetime.now()
    prompt = (
        f"Parse this time expression into an ISO datetime.\n\n"
        f"Current time: {now.strftime('%Y-%m-%d %H:%M')}\n"
        f'Time expression: "{time_str}"\n\n'
        f"Return ONLY the datetime in ISO format (YYYY-MM-DDTHH:MM:SS), nothing else.\n"
        f"If the expression is relative (like \"3pm\"), assume it means today or the next occurrence.\n\n"
        f"Output:"
    )

    try:
        result = await llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You parse time expressions into ISO datetime format. Return ONLY the datetime string, nothing else."},
                {"role": "user", "content": prompt},
            ],
            enable_thinking=False,
        )
        dt_str = result.content.strip().strip("\"'")
        return datetime.fromisoformat(dt_str)
    except Exception as e:
        logger.error(f"Failed to parse datetime: {e}")
        return None


def _parse_time_to_datetime(time_str: str) -> datetime:
    """Parse natural language time to datetime using dateutil."""
    from dateutil import parser as date_parser

    default_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        return date_parser.parse(time_str, fuzzy=True, default=default_time)
    except Exception:
        raise ValueError(f"Could not parse time: {time_str}")


# =============================================================================
# query_events
# =============================================================================

@tool
async def query_events(
    time_range: Annotated[str, "Time range to search (e.g., 'today', 'tomorrow', 'this week', 'next week', 'this month', 'next 3 days')"],
    query: Annotated[Optional[str], "Optional keywords to search in event titles"] = None,
    max_results: Annotated[int, "Maximum number of events to return (default 10)"] = 10,
    *,
    context: AgentToolContext,
) -> str:
    """Search and list calendar events. Returns events matching the time range and optional keyword query."""
    from .search_helper import parse_time_range

    provider, account, error = await _get_provider(context.tenant_id)
    if error:
        return error

    try:
        time_min, time_max = parse_time_range(time_range)

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            max_results=max_results,
            query=query or None,
        )

        if not result.get("success"):
            return f"Failed to search calendar: {result.get('error', 'Unknown error')}"

        events = result.get("data", [])
        events.sort(key=lambda e: e.get("start") or datetime.min)

        if not events:
            return f"No events found {time_range}."

        parts = [f"Found {len(events)} event(s) {time_range}:"]

        for i, event in enumerate(events[:10], 1):
            summary = html.unescape(event.get("summary", "No title"))
            start_str = _format_event_time(event.get("start"))
            location = event.get("location", "")

            event_text = f"{i}. {start_str} - {summary}"
            if location:
                event_text += f" ({location})"
            parts.append(event_text)

        if len(events) > 10:
            parts.append(f"\n... and {len(events) - 10} more event(s).")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Calendar query failed: {e}", exc_info=True)
        return "Couldn't check the calendar. Try again later?"


# =============================================================================
# create_event
# =============================================================================

async def _preview_create_event(args: dict, context: AgentToolContext) -> str:
    """Generate a preview of the event to be created."""
    summary = args.get("summary", "")
    start_str = args.get("start", "")
    end_str = args.get("end", "")
    description = args.get("description", "")
    location = args.get("location", "")
    attendees = args.get("attendees", "")

    if not end_str and start_str:
        try:
            start_dt = _parse_time_to_datetime(start_str)
            end_dt = start_dt + timedelta(hours=1)
            if end_dt.date() == start_dt.date():
                end_str = end_dt.strftime("%I:%M %p").lstrip("0")
            else:
                end_str = end_dt.strftime("%b %d at %I:%M %p").lstrip("0")
        except Exception:
            end_str = f"{start_str} + 1 hour"

    parts = ["Event Draft:"]
    parts.append(f"Title: {summary}")
    parts.append(f"Start: {start_str}")
    parts.append(f"End: {end_str}")

    if description:
        parts.append(f"Description: {description}")
    if location:
        parts.append(f"Location: {location}")
    if attendees:
        parts.append(f"Attendees: {attendees}")

    parts.append("---")
    parts.append("Looks good?")

    return "\n".join(parts)


@tool(needs_approval=True, get_preview=_preview_create_event)
async def create_event(
    summary: Annotated[str, "Event title/summary"],
    start: Annotated[str, "Event start time (e.g., 'tomorrow at 2pm', '2025-03-15 14:00')"],
    end: Annotated[Optional[str], "Event end time (optional, defaults to 1 hour after start)"] = None,
    description: Annotated[Optional[str], "Event description/details (optional)"] = None,
    location: Annotated[Optional[str], "Event location (optional)"] = None,
    attendees: Annotated[Optional[str], "Comma-separated list of attendee email addresses (optional)"] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Create a new calendar event. Requires at least a title and start time."""
    if not summary or not start:
        return "Error: event title (summary) and start time are required."

    provider, account, error = await _get_provider(context.tenant_id)
    if error:
        return error

    try:
        start_dt = _parse_time_to_datetime(start)
        if end:
            end_dt = _parse_time_to_datetime(end)
        else:
            end_dt = start_dt + timedelta(hours=1)

        attendee_list = []
        if attendees:
            attendee_list = [email.strip() for email in attendees.split(",") if email.strip()]

        result = await provider.create_event(
            summary=summary,
            start=start_dt,
            end=end_dt,
            description=description,
            location=location,
            attendees=attendee_list,
        )

        if result.get("success"):
            event_link = result.get("html_link", "")
            response = f"Done! I've added '{summary}' to your calendar."
            if event_link:
                response += f"\n{event_link}"
            return response
        else:
            return f"I couldn't create that event. {result.get('error', '')}"

    except Exception as e:
        logger.error(f"Failed to create event: {e}", exc_info=True)
        return "Something went wrong creating the event. Want me to try again?"


# =============================================================================
# update_event
# =============================================================================

async def _preview_update_event(args: dict, context: AgentToolContext) -> str:
    """Generate a preview of the changes to be applied."""
    target = args.get("target", "")
    changes = args.get("changes", {})

    parts = [f'I\'ll update the event matching "{target}":']

    if changes.get("new_title"):
        parts.append(f"  Title -> {changes['new_title']}")
    if changes.get("new_time"):
        parts.append(f"  Time -> {changes['new_time']}")
    if changes.get("new_location"):
        parts.append(f"  Location -> {changes['new_location']}")
    if changes.get("new_duration"):
        parts.append(f"  Duration -> {changes['new_duration']}")

    if len(parts) == 1:
        parts.append("  (no changes specified)")

    parts.append("\nMake these changes?")
    return "\n".join(parts)


@tool(needs_approval=True, get_preview=_preview_update_event)
async def update_event(
    target: Annotated[str, "Keywords to identify the event (title, person's name, time reference like 'my 2pm meeting')"],
    changes: Annotated[Dict, "What to change: object with optional keys new_time, new_title, new_location, new_duration"],
    *,
    context: AgentToolContext,
) -> str:
    """Update an existing calendar event. Specify the target event and what to change."""
    if not target:
        return "Error: please specify which event to update (target)."
    if not changes:
        return "Error: please specify what to change (changes)."

    provider, account, error = await _get_provider(context.tenant_id)
    if error:
        return error

    try:
        # Search for the target event
        now = datetime.now()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=30)).isoformat() + "Z"

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            query=target,
            max_results=10,
        )

        target_event = None

        if result.get("success") and result.get("data"):
            events = result["data"]
            target_lower = target.lower()
            for event in events:
                event_title = event.get("summary", "").lower()
                event_desc = event.get("description", "").lower()
                if target_lower in event_title or target_lower in event_desc:
                    target_event = event
                    break
            if not target_event and events:
                if any(word in target_lower for word in ["meeting", "call", "sync", "appointment"]):
                    target_event = events[0]

        if not result.get("success") or not target_event:
            # Broader search without query filter
            result = await provider.list_events(
                time_min=time_min,
                time_max=time_max,
                max_results=20,
            )
            if result.get("success") and result.get("data"):
                target_lower = target.lower()
                for event in result["data"]:
                    event_title = event.get("summary", "").lower()
                    if target_lower in event_title:
                        target_event = event
                        break

        if not target_event:
            return f"I couldn't find an event matching '{target}'. Could you be more specific?"

        event_id = target_event.get("id")
        if not event_id:
            return "I couldn't identify the event to update."

        # Parse changes
        parsed_changes = {}

        if changes.get("new_time"):
            parsed_time = await _parse_datetime_with_llm(changes["new_time"], context.llm_client)
            if parsed_time:
                parsed_changes["start"] = parsed_time
                # Preserve original duration
                old_start = target_event.get("start", {})
                old_end = target_event.get("end", {})
                if old_start.get("dateTime") and old_end.get("dateTime"):
                    try:
                        old_start_dt = datetime.fromisoformat(old_start["dateTime"].replace("Z", "+00:00"))
                        old_end_dt = datetime.fromisoformat(old_end["dateTime"].replace("Z", "+00:00"))
                        duration = old_end_dt - old_start_dt
                        parsed_changes["end"] = parsed_time + duration
                    except Exception:
                        parsed_changes["end"] = parsed_time + timedelta(hours=1)
                else:
                    parsed_changes["end"] = parsed_time + timedelta(hours=1)

        if changes.get("new_title"):
            parsed_changes["summary"] = changes["new_title"]
        if changes.get("new_location"):
            parsed_changes["location"] = changes["new_location"]
        if changes.get("new_duration"):
            duration_str = changes["new_duration"].lower()
            hours = 1
            if "hour" in duration_str:
                match = re.search(r"(\d+)", duration_str)
                if match:
                    hours = int(match.group(1))
            if "start" in parsed_changes:
                parsed_changes["end"] = parsed_changes["start"] + timedelta(hours=hours)

        if not parsed_changes:
            return "I'm not sure what changes to make. Could you be more specific?"

        result = await provider.update_event(
            event_id=event_id,
            summary=parsed_changes.get("summary"),
            start=parsed_changes.get("start"),
            end=parsed_changes.get("end"),
            location=parsed_changes.get("location"),
            description=parsed_changes.get("description"),
        )

        if result.get("success"):
            event_title = parsed_changes.get("summary") or target_event.get("summary", "event")
            if "start" in parsed_changes:
                new_time = parsed_changes["start"].strftime("%Y-%m-%d %H:%M") if isinstance(parsed_changes["start"], datetime) else str(parsed_changes["start"])
                return f'Done! I\'ve moved "{event_title}" to {new_time}.'
            elif "summary" in parsed_changes:
                return f'Done! I\'ve renamed the event to "{event_title}".'
            else:
                return f'Done! I\'ve updated "{event_title}".'
        else:
            return f"I couldn't update that event. {result.get('error', '')}"

    except Exception as e:
        logger.error(f"Failed to update event: {e}", exc_info=True)
        return "Something went wrong updating the event. Want me to try again?"


# =============================================================================
# delete_event
# =============================================================================

async def _preview_delete_event(args: dict, context: AgentToolContext) -> str:
    """Search for events matching criteria and show what would be deleted."""
    from .search_helper import search_calendar_events

    search_query = args.get("search_query", "")
    time_range = args.get("time_range", "next 7 days")

    result = await search_calendar_events(
        user_id=context.tenant_id,
        search_query=search_query,
        time_range=time_range,
        max_results=50,
        account_hint="primary",
    )

    if not result.get("success") or not result.get("events"):
        return "Couldn't find any events matching that criteria."

    events = result["events"]

    if len(events) == 1:
        parts = ["Found 1 event:"]
    else:
        parts = [f"Found {len(events)} events:"]

    for event in events[:5]:
        summary = event.get("summary", "No title")
        start_str = _format_event_time(event.get("start"))
        location = event.get("location", "")

        event_line = f"- {summary} - {start_str}"
        if location:
            event_line += f" @ {location}"
        parts.append(event_line)

    if len(events) > 5:
        parts.append(f"...and {len(events) - 5} more")

    parts.append("")
    if len(events) == 1:
        parts.append("Delete it?")
    else:
        parts.append("Delete these?")

    return "\n".join(parts)


@tool(needs_approval=True, get_preview=_preview_delete_event)
async def delete_event(
    search_query: Annotated[str, "Keywords to search for events to delete (event title, keywords)"],
    time_range: Annotated[str, "Time range to search (e.g., 'today', 'tomorrow', 'this week'). Defaults to 'next 7 days'."] = "next 7 days",
    *,
    context: AgentToolContext,
) -> str:
    """Delete calendar events matching the search criteria."""
    from .search_helper import search_calendar_events
    from onevalet.providers.calendar.factory import CalendarProviderFactory

    result = await search_calendar_events(
        user_id=context.tenant_id,
        search_query=search_query,
        time_range=time_range,
        max_results=50,
        account_hint="primary",
    )

    if not result.get("success") or not result.get("events"):
        return "I couldn't find any events matching that criteria."

    events = result["events"]
    account = result.get("account")
    event_ids = [event.get("event_id") for event in events if event.get("event_id")]

    if not event_ids:
        return "I couldn't find any events to delete."

    if not account:
        return "I'm not sure which calendar to use."

    provider = CalendarProviderFactory.create_provider(account)
    if not provider:
        return "Sorry, I can't access that calendar provider yet."

    if not await provider.ensure_valid_token():
        return "I lost access to your calendar. Could you reconnect it?"

    deleted_count = 0
    failed_count = 0

    for event_id in event_ids:
        del_result = await provider.delete_event(event_id)
        if del_result.get("success"):
            deleted_count += 1
        else:
            failed_count += 1
            logger.warning(f"Failed to delete event {event_id}: {del_result.get('error')}")

    if deleted_count == 0:
        return "I couldn't delete those events. They might have already been removed."
    elif failed_count > 0:
        return f"Done! I deleted {deleted_count} event(s), but {failed_count} couldn't be removed."
    else:
        return f"Done! I've removed {deleted_count} event(s) from your calendar."
