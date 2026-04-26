"""
Calendar Tools — Standalone API functions for CalendarAgent's mini ReAct loop.

Extracted from CalendarAgent, CreateEventAgent, UpdateEventAgent, and DeleteEventAgent.
"""

import html
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated, Dict, Optional

from koa.builtin_agents.shared.routing_preferences import (
    resolve_surface_target,
    wrap_routing_error,
)
from koa.models import AgentToolContext, ToolOutput
from koa.providers.calendar.local import LocalCalendarProvider
from koa.providers.local_backend import LocalBackendClient
from koa.tool_decorator import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================


async def _resolve_calendar_provider(
    context: AgentToolContext,
    target_provider: str | None = None,
    target_account: str | None = None,
    *,
    operation: str = "write",
):
    from koa.providers.calendar.factory import CalendarProviderFactory
    from koa.providers.calendar.resolver import CalendarAccountResolver

    backend_client = LocalBackendClient.from_context(context)
    db = (context.context_hints or {}).get("db")
    try:
        target = await resolve_surface_target(
            tenant_id=context.tenant_id,
            surface="calendar",
            db=db,
            explicit_provider=target_provider,
            explicit_account=target_account,
        )
    except Exception as e:
        logger.error(f"Failed to resolve calendar routing target: {e}", exc_info=True)
        error_reason = "read_failed" if operation == "read" else "write_failed"
        return None, None, wrap_routing_error("calendar", target_provider or "local", error_reason)

    if target.provider == "local":
        if db is None:
            logger.error("LocalCalendarProvider requested but no db in context_hints")
            return None, None, wrap_routing_error(
                "calendar", "local", "read_failed" if operation == "read" else "write_failed"
            )
        return (
            LocalCalendarProvider(context.tenant_id, db),
            {"provider": "local", "account_name": "local"},
            None,
        )

    if target.provider not in CalendarProviderFactory.get_supported_providers():
        return None, None, wrap_routing_error("calendar", target.provider, "unsupported_provider")

    account = await CalendarAccountResolver.resolve_account_for_provider(
        context.tenant_id,
        target.provider,
        target.account or "primary",
    )
    if not account:
        return None, None, wrap_routing_error("calendar", target.provider, "not_connected")

    provider = CalendarProviderFactory.create_provider(account)
    if not provider:
        return None, None, wrap_routing_error("calendar", target.provider, "unsupported_provider")

    if not await provider.ensure_valid_token():
        return None, None, wrap_routing_error("calendar", target.provider, "auth_expired")

    return provider, account, None


def _coerce_event_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, dict):
        value = value.get("dateTime", value.get("date"))
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _event_sort_key(event) -> datetime:
    return _coerce_event_datetime(event.get("start")) or datetime.min.replace(tzinfo=timezone.utc)


def _event_id(event) -> str:
    return event.get("id") or event.get("event_id") or ""


def _event_link(event) -> str:
    return event.get("htmlLink") or event.get("html_link") or ""


async def _search_resolved_calendar_events(
    context: AgentToolContext,
    provider,
    time_range: str,
    search_query: str | None = None,
    max_results: int = 50,
):
    from .search_helper import parse_time_range

    user_tz = context.metadata.get("timezone") if context.metadata else None
    time_min, time_max = parse_time_range(time_range, user_tz=user_tz)
    result = await provider.list_events(
        time_min=time_min,
        time_max=time_max,
        query=search_query,
        max_results=max_results,
    )

    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Unknown error")}

    return {
        "success": True,
        "events": result.get("data", []),
        "time_min": time_min,
        "time_max": time_max,
    }


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

    now = datetime.now(timezone.utc)  # UTC is fine here — LLM just needs a reference point
    prompt = (
        f"Parse this time expression into an ISO datetime.\n\n"
        f"Current time: {now.strftime('%Y-%m-%d %H:%M')}\n"
        f'Time expression: "{time_str}"\n\n'
        f"Return ONLY the datetime in ISO format (YYYY-MM-DDTHH:MM:SS), nothing else.\n"
        f'If the expression is relative (like "3pm"), assume it means today or the next occurrence.\n\n'
        f"Output:"
    )

    try:
        result = await llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You parse time expressions into ISO datetime format. Return ONLY the datetime string, nothing else.",
                },
                {"role": "user", "content": prompt},
            ],
            enable_thinking=False,
        )
        dt_str = result.content.strip().strip("\"'")
        return datetime.fromisoformat(dt_str)
    except Exception as e:
        logger.error(f"Failed to parse datetime: {e}")
        return None


def _parse_time_to_datetime(time_str: str, user_tz: str | None = None) -> datetime:
    """Parse natural language time to datetime using dateutil.

    When user_tz is provided, the default date/time anchor is midnight
    in the user's local timezone. This prevents off-by-one-day errors
    when the user says "April 27" in PST but the server is in UTC.
    """
    from dateutil import parser as date_parser

    if user_tz:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(user_tz)
            default_time = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception:
            default_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        default_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        parsed = date_parser.parse(time_str, fuzzy=True, default=default_time)
        # If the parsed result has no tzinfo, attach the user's timezone
        if parsed.tzinfo is None and user_tz:
            try:
                from zoneinfo import ZoneInfo
                parsed = parsed.replace(tzinfo=ZoneInfo(user_tz))
            except Exception:
                pass
        return parsed
    except Exception:
        raise ValueError(f"Could not parse time: {time_str}")


# =============================================================================
# query_events
# =============================================================================


@tool
async def query_events(
    time_range: Annotated[
        str,
        "Time range to search (e.g., 'today', 'tomorrow', 'this week', 'next week', 'this month', 'next 3 days')",
    ],
    query: Annotated[Optional[str], "Optional keywords to search in event titles"] = None,
    max_results: Annotated[int, "Maximum number of events to return (default 10)"] = 10,
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local or google"
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work"
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Search and list calendar events. Returns events matching the time range and optional keyword query."""
    from .search_helper import parse_time_range

    provider, account, error = await _resolve_calendar_provider(
        context,
        target_provider,
        target_account,
        operation="read",
    )
    if error:
        return error

    try:
        user_tz = context.metadata.get("timezone") if context.metadata else None
        time_min, time_max = parse_time_range(time_range, user_tz=user_tz)

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            max_results=max_results,
            query=query or None,
        )

        if not result.get("success"):
            return wrap_routing_error("calendar", account.get("provider", "calendar"), "read_failed")

        events = result.get("data", [])
        events.sort(key=_event_sort_key)

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

        text_result = "\n".join(parts)

        # Build inline cards for frontend rendering
        event_cards = []
        for event in events[:10]:
            summary = html.unescape(event.get("summary", "No title"))
            start_str = _format_event_time(event.get("start"))
            card = {
                "card_type": "event",
                "name": summary,
                "date": start_str,
            }
            location = event.get("location", "")
            if location:
                card["location"] = location
            event_id = _event_id(event)
            html_link = _event_link(event)
            if html_link:
                card["eventUrl"] = html_link
            elif event_id:
                card["eventId"] = event_id
            event_cards.append(card)

        media = []
        if event_cards:
            media.append(
                {
                    "type": "inline_cards",
                    "data": json.dumps(event_cards),
                    "media_type": "application/json",
                    "metadata": {"for_storage": False},
                }
            )

        return ToolOutput(text=text_result, media=media)

    except Exception as e:
        logger.error(f"Calendar query failed: {e}", exc_info=True)
        return "Couldn't check the calendar. Try again later?"


# =============================================================================
# create_event
# =============================================================================


async def _preview_create_event(args: dict, context: AgentToolContext) -> str:
    """Generate a preview of the event to be created.
    Returns JSON with inline_cards for rich frontend rendering.
    """
    import json as _json

    summary = args.get("summary", "")
    start_str = args.get("start", "")
    end_str = args.get("end", "")
    location = args.get("location", "")

    user_tz = context.metadata.get("timezone") if context.metadata else None

    if not end_str and start_str:
        try:
            start_dt = _parse_time_to_datetime(start_str, user_tz=user_tz)
            end_dt = start_dt + timedelta(hours=1)
            if end_dt.date() == start_dt.date():
                end_str = end_dt.strftime("%I:%M %p").lstrip("0")
            else:
                end_str = end_dt.strftime("%b %d at %I:%M %p").lstrip("0")
        except Exception:
            end_str = f"{start_str} + 1 hour"

    # Parse start date into ISO format for frontend card
    start_date = ""
    start_time = ""
    try:
        start_dt = _parse_time_to_datetime(start_str, user_tz=user_tz)
        start_date = start_dt.strftime("%Y-%m-%d")
        start_time = start_dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        start_date = start_str

    # Build inline card JSON for rich rendering
    card = {
        "card_type": "event_draft",
        "title": summary,
        "startDate": start_date,
        "startTime": start_time,
        "endTime": end_str,
        "location": location or None,
        "options": ["approve", "edit", "decline"],
    }

    # Return text with embedded card marker for the frontend
    text_parts = [f"📅 **{summary}**"]
    text_parts.append(f"🕐 {start_str} — {end_str}")
    if location:
        text_parts.append(f"📍 {location}")

    # Embed card data as a JSON block the frontend can parse
    return (
        "\n".join(text_parts)
        + "\n\n<!-- inline_card:"
        + _json.dumps(card, ensure_ascii=False)
        + " -->"
    )


async def _schedule_event_reminders(
    context: AgentToolContext,
    summary: str,
    start_dt: datetime,
    location: str = None,
    attendees: list = None,
):
    """Auto-schedule proactive reminders when a calendar event is created/updated."""
    cron_service = context.context_hints.get("cron_service") if context.context_hints else None
    if not cron_service:
        return

    from koa.triggers.cron.models import (
        AgentTurnPayload,
        AtSchedule,
        CronJobCreate,
        DeliveryConfig,
        DeliveryMode,
        SessionTarget,
        WakeMode,
    )

    tenant_id = context.tenant_id
    jobs = []

    # 30-min-before reminder
    remind_at = start_dt - timedelta(minutes=30)
    if remind_at > datetime.now(timezone.utc):
        meeting_link_hint = "Include the meeting link if available."
        jobs.append(
            CronJobCreate(
                name=f"Reminder: {summary}",
                description=f"30-min reminder for '{summary}'",
                user_id=tenant_id,
                schedule=AtSchedule(at=remind_at.isoformat()),
                session_target=SessionTarget.ISOLATED,
                wake_mode=WakeMode.NEXT_HEARTBEAT,
                payload=AgentTurnPayload(
                    message=f"Remind the user: '{summary}' starts in 30 minutes. {meeting_link_hint}"
                ),
                delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
            )
        )

    # Departure reminder (45 min before, only if location)
    if location:
        depart_at = start_dt - timedelta(minutes=45)
        if depart_at > datetime.now(timezone.utc):
            jobs.append(
                CronJobCreate(
                    name=f"Depart for: {summary}",
                    description=f"Departure reminder for '{summary}' at {location}",
                    user_id=tenant_id,
                    schedule=AtSchedule(at=depart_at.isoformat()),
                    session_target=SessionTarget.ISOLATED,
                    wake_mode=WakeMode.NEXT_HEARTBEAT,
                    payload=AgentTurnPayload(
                        message=f"The user has '{summary}' at '{location}' starting soon. "
                        f"Get their current location, calculate travel time, and tell them when to leave."
                    ),
                    delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                )
            )

    # Meeting prep (10 min before, only if attendees)
    if attendees:
        prep_at = start_dt - timedelta(minutes=10)
        if prep_at > datetime.now(timezone.utc):
            attendee_str = ", ".join(attendees[:5])
            jobs.append(
                CronJobCreate(
                    name=f"Prep: {summary}",
                    description=f"Meeting prep for '{summary}'",
                    user_id=tenant_id,
                    schedule=AtSchedule(at=prep_at.isoformat()),
                    session_target=SessionTarget.ISOLATED,
                    wake_mode=WakeMode.NEXT_HEARTBEAT,
                    payload=AgentTurnPayload(
                        message=f"'{summary}' starts in 10 minutes with: {attendee_str}. "
                        f"Search recent emails from these people and give the user a brief context summary."
                    ),
                    delivery=DeliveryConfig(mode=DeliveryMode.ANNOUNCE, channel="callback"),
                )
            )

    for job in jobs:
        try:
            await cron_service.add(job)
            logger.info(f"Auto-scheduled reminder '{job.name}' for tenant {tenant_id}")
        except Exception as e:
            logger.warning(f"Failed to schedule reminder '{job.name}': {e}")


@tool(needs_approval=True, get_preview=_preview_create_event)
async def create_event(
    summary: Annotated[str, "Event title/summary"],
    start: Annotated[str, "Event start time (e.g., 'tomorrow at 2pm', '2025-03-15 14:00')"],
    end: Annotated[
        Optional[str], "Event end time (optional, defaults to 1 hour after start)"
    ] = None,
    description: Annotated[Optional[str], "Event description/details (optional)"] = None,
    location: Annotated[Optional[str], "Event location (optional)"] = None,
    attendees: Annotated[
        Optional[str], "Comma-separated list of attendee email addresses (optional)"
    ] = None,
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local or google"
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work"
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Create a new calendar event. Requires at least a title and start time."""
    if not summary or not start:
        return "Error: event title (summary) and start time are required."

    provider, account, error = await _resolve_calendar_provider(
        context,
        target_provider,
        target_account,
    )
    if error:
        return error

    try:
        user_tz = context.metadata.get("timezone") if context.metadata else None
        start_dt = _parse_time_to_datetime(start, user_tz=user_tz)
        if end:
            end_dt = _parse_time_to_datetime(end, user_tz=user_tz)
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
            # Auto-schedule proactive reminders
            try:
                await _schedule_event_reminders(
                    context,
                    summary,
                    start_dt,
                    location=location,
                    attendees=attendee_list or None,
                )
            except Exception as e:
                logger.debug(f"Auto-reminder scheduling failed: {e}")

            event_link = result.get("html_link", "")
            response = f"Done! I've added '{summary}' to your calendar."
            if event_link:
                response += f"\n{event_link}"
            return response
        else:
            return wrap_routing_error("calendar", account.get("provider", "calendar"), "write_failed")

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
    target: Annotated[
        str,
        "Keywords to identify the event (title, person's name, time reference like 'my 2pm meeting')",
    ],
    changes: Annotated[
        Dict,
        "What to change: object with optional keys new_time, new_title, new_location, new_duration",
    ],
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local or google"
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work"
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Update an existing calendar event. Specify the target event and what to change."""
    if not target:
        return "Error: please specify which event to update (target)."
    if not changes:
        return "Error: please specify what to change (changes)."

    provider, account, error = await _resolve_calendar_provider(
        context,
        target_provider,
        target_account,
    )
    if error:
        return error

    try:
        # Search for the target event
        user_tz_str = context.metadata.get("timezone") if context.metadata else None
        from .search_helper import _resolve_tz

        user_tz_obj = _resolve_tz(user_tz_str)
        now = datetime.now(user_tz_obj)
        time_min = now
        time_max = now + timedelta(days=30)

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            query=target,
            max_results=10,
        )
        if not result.get("success"):
            return wrap_routing_error("calendar", account.get("provider", "calendar"), "write_failed")

        target_event = None

        if result.get("data"):
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

        if not target_event:
            # Broader search without query filter
            result = await provider.list_events(
                time_min=time_min,
                time_max=time_max,
                max_results=20,
            )
            if not result.get("success"):
                return wrap_routing_error(
                    "calendar",
                    account.get("provider", "calendar"),
                    "write_failed",
                )
            if result.get("data"):
                target_lower = target.lower()
                for event in result["data"]:
                    event_title = event.get("summary", "").lower()
                    if target_lower in event_title:
                        target_event = event
                        break

        if not target_event:
            return f"I couldn't find an event matching '{target}'. Could you be more specific?"

        event_id = _event_id(target_event)
        if not event_id:
            return "I couldn't identify the event to update."

        # Parse changes
        parsed_changes = {}

        if changes.get("new_time"):
            parsed_time = await _parse_datetime_with_llm(changes["new_time"], context.llm_client)
            if parsed_time:
                parsed_changes["start"] = parsed_time
                # Preserve original duration
                old_start_dt = _coerce_event_datetime(target_event.get("start"))
                old_end_dt = _coerce_event_datetime(target_event.get("end"))
                if old_start_dt and old_end_dt:
                    duration = old_end_dt - old_start_dt
                    parsed_changes["end"] = parsed_time + duration
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
            # Auto-schedule reminders if the event time was changed
            if "start" in parsed_changes and isinstance(parsed_changes["start"], datetime):
                try:
                    event_title = parsed_changes.get("summary") or target_event.get(
                        "summary", "event"
                    )
                    updated_location = parsed_changes.get("location") or target_event.get(
                        "location"
                    )
                    raw_attendees = target_event.get("attendees") or []
                    attendee_emails = (
                        [a.get("email") for a in raw_attendees if a.get("email")]
                        if raw_attendees
                        else None
                    )
                    await _schedule_event_reminders(
                        context,
                        event_title,
                        parsed_changes["start"],
                        location=updated_location,
                        attendees=attendee_emails or None,
                    )
                except Exception as e:
                    logger.debug(f"Auto-reminder scheduling on update failed: {e}")

            event_title = parsed_changes.get("summary") or target_event.get("summary", "event")
            if "start" in parsed_changes:
                new_time = (
                    parsed_changes["start"].strftime("%Y-%m-%d %H:%M")
                    if isinstance(parsed_changes["start"], datetime)
                    else str(parsed_changes["start"])
                )
                return f'Done! I\'ve moved "{event_title}" to {new_time}.'
            elif "summary" in parsed_changes:
                return f'Done! I\'ve renamed the event to "{event_title}".'
            else:
                return f'Done! I\'ve updated "{event_title}".'
        else:
            return wrap_routing_error("calendar", account.get("provider", "calendar"), "write_failed")

    except Exception as e:
        logger.error(f"Failed to update event: {e}", exc_info=True)
        return "Something went wrong updating the event. Want me to try again?"


# =============================================================================
# delete_event
# =============================================================================


async def _preview_delete_event(args: dict, context: AgentToolContext) -> str:
    """Search for events matching criteria and show what would be deleted."""
    search_query = args.get("search_query", "")
    time_range = args.get("time_range", "next 7 days")
    target_provider = args.get("target_provider")
    target_account = args.get("target_account")

    provider, account, error = await _resolve_calendar_provider(
        context,
        target_provider,
        target_account,
        operation="read",
    )
    if error:
        return error

    result = await _search_resolved_calendar_events(
        context=context,
        provider=provider,
        time_range=time_range,
        search_query=search_query,
        max_results=50,
    )
    if not result.get("success"):
        return wrap_routing_error("calendar", account.get("provider", "calendar"), "read_failed")

    if not result.get("events"):
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
    time_range: Annotated[
        str,
        "Time range to search (e.g., 'today', 'tomorrow', 'this week'). Defaults to 'next 7 days'.",
    ] = "next 7 days",
    target_provider: Annotated[
        Optional[str], "Optional explicit provider like local or google"
    ] = None,
    target_account: Annotated[
        Optional[str], "Optional explicit account label like primary or work"
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Delete calendar events matching the search criteria."""
    provider, account, error = await _resolve_calendar_provider(
        context,
        target_provider,
        target_account,
    )
    if error:
        return error

    result = await _search_resolved_calendar_events(
        context=context,
        provider=provider,
        time_range=time_range,
        search_query=search_query,
        max_results=50,
    )
    if not result.get("success"):
        return wrap_routing_error("calendar", account.get("provider", "calendar"), "write_failed")

    if not result.get("events"):
        return "I couldn't find any events matching that criteria."

    events = result["events"]
    event_ids = [_event_id(event) for event in events if _event_id(event)]

    if not event_ids:
        return "I couldn't find any events to delete."

    deleted_count = 0
    failed_count = 0

    for event_id in event_ids:
        del_result = await provider.delete_event(event_id)
        if del_result.get("success"):
            deleted_count += 1
        else:
            failed_count += 1
            logger.warning(f"Failed to delete event {event_id}: {del_result.get('error')}")

    if deleted_count == 0 and failed_count > 0:
        return wrap_routing_error("calendar", account.get("provider", "calendar"), "write_failed")
    if deleted_count == 0:
        return "I couldn't delete those events. They might have already been removed."
    elif failed_count > 0:
        return f"Done! I deleted {deleted_count} event(s), but {failed_count} couldn't be removed."
    else:
        return f"Done! I've removed {deleted_count} event(s) from your calendar."


# =============================================================================
# check_upcoming_events
# =============================================================================


@tool
async def check_upcoming_events(
    minutes_ahead: Annotated[
        int, "Check for events starting within this many minutes. Default 30."
    ] = 30,
    *,
    context: AgentToolContext,
) -> str:
    """Check for calendar events starting soon. Used by proactive reminders."""
    provider, account, error = await _resolve_calendar_provider(context, operation="read")
    if error:
        return error

    try:
        now = datetime.now(timezone.utc)
        time_min = now
        time_max = now + timedelta(minutes=minutes_ahead)

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            max_results=10,
        )

        if not result.get("success"):
            return wrap_routing_error("calendar", account.get("provider", "calendar"), "read_failed")

        events = result.get("data", [])
        if not events:
            return f"No upcoming events in the next {minutes_ahead} minutes."

        lines = []
        for event in events:
            summary = html.unescape(event.get("summary", "Untitled"))
            start = event.get("start")
            start_dt = _coerce_event_datetime(start)

            if start_dt:
                mins_until = max(0, int((start_dt - now).total_seconds() / 60))
                line = f"📅 {summary} starts in {mins_until} minutes"
            else:
                line = f"📅 {summary} starts soon"

            location = event.get("location", "")
            if location:
                line += f"\n📍 {location}"
            html_link = _event_link(event)
            if html_link:
                line += f"\n🔗 {html_link}"
            lines.append(line)

        return "\n\n".join(lines)

    except Exception as e:
        logger.error(f"check_upcoming_events failed: {e}", exc_info=True)
        return "Sorry, I couldn't check your upcoming events right now."


@tool
async def query_local_events(
    hours_ahead: Annotated[
        int, "Window size in hours starting from now. Default 48."
    ] = 48,
    calendar_name: Annotated[
        Optional[str], "Optional filter to a single local calendar name."
    ] = None,
    *,
    context: AgentToolContext,
) -> str:
    """Read events from the user's **local iOS Calendar** (EventKit), synced
    via the Koi app.  Complements the cloud providers (Google/iCloud) by
    covering calendars that only exist on-device.  Returns a compact list."""
    db = context.context_hints.get("db") if context.context_hints else None
    if db is None:
        return "Local calendar is not available right now."

    from .eventkit_adapter import fetch_upcoming_local

    try:
        calendars = [calendar_name] if calendar_name else None
        events = await fetch_upcoming_local(
            db,
            context.tenant_id,
            hours_ahead=max(1, int(hours_ahead)),
            calendars=calendars,
        )
    except Exception as e:
        logger.warning("query_local_events failed: %s", e)
        return "Couldn't read local calendar right now."

    if not events:
        return f"No local calendar events in the next {hours_ahead} hours."

    lines = []
    for e in events[:15]:
        title = e.get("title") or "(Untitled)"
        start = e.get("start") or ""
        cal = e.get("calendar") or ""
        loc = e.get("location") or ""
        tag = f" [{cal}]" if cal else ""
        line = f"📅 {title}{tag} — {start}"
        if loc:
            line += f"\n   📍 {loc}"
        lines.append(line)
    return "\n".join(lines)

