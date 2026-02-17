"""
Important Dates Tools - CRUD for birthdays, anniversaries, and other important dates

These tools use CredentialStore via the AgentToolContext.credentials
attribute. The application is responsible for providing a data-access layer
(e.g. a database client) that implements the actual storage.

The tools expect a callable ``context.metadata["important_dates_store"]``
object with these async methods:
    - get_important_dates(user_id, days_ahead) -> list[dict]
    - search_important_dates(user_id, search_term, limit) -> list[dict]
    - create_important_date(user_id, data) -> dict | None
    - update_important_date(user_id, date_id, updates) -> dict | None
    - delete_important_date(user_id, date_id) -> bool
    - get_today_reminders(user_id) -> list[dict]

If the store is not provided, the tools return an error message.
"""

import re
import logging
from datetime import datetime
from typing import Optional

from onevalet.standard_agent import AgentToolContext

logger = logging.getLogger(__name__)


def _get_store(context: Optional[AgentToolContext]):
    """Extract the important_dates_store from context metadata."""
    if not context:
        return None
    return context.metadata.get("important_dates_store")


def _format_timing(d: dict) -> str:
    """Format a date entry's timing string."""
    days_until = d.get("days_until")
    if days_until == 0:
        return "TODAY"
    if days_until == 1:
        return "tomorrow"
    if days_until is not None and days_until <= 7:
        return f"in {days_until} days"

    upcoming = d.get("upcoming_date") or d.get("date")
    if upcoming:
        try:
            dt = datetime.strptime(str(upcoming)[:10], "%Y-%m-%d") if isinstance(upcoming, str) else upcoming
            return dt.strftime("%b %d")
        except Exception:
            return str(upcoming)[:10]
    return ""


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse a date string in various formats."""
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d", "%b %d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Natural language fallback
    date_lower = date_str.lower()
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    for month_name, month_num in month_map.items():
        if month_name in date_lower:
            day_match = re.search(r"\d+", date_str)
            if day_match:
                return datetime(datetime.now().year, month_num, int(day_match.group()))
    return None


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

async def get_important_dates_executor(args: dict, context: AgentToolContext = None) -> str:
    """Get upcoming important dates."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    store = _get_store(context)
    if not store:
        return "Error: Important dates storage not configured"

    days_ahead = args.get("days_ahead", 60)

    try:
        dates = await store.get_important_dates(user_id=context.tenant_id, days_ahead=days_ahead)
        if not dates:
            return "No upcoming important dates found."

        output = []
        for d in dates[:15]:
            title = d.get("title", "Event")
            timing = _format_timing(d)
            person_name = d.get("person_name", "")

            line = f"- {title}: {timing}"
            if person_name and person_name.lower() not in title.lower():
                line += f" ({person_name})"
            output.append(line)

        result = f"Upcoming important dates ({len(dates)} total):\n" + "\n".join(output)
        if len(dates) > 15:
            result += f"\n... and {len(dates) - 15} more"
        return result

    except Exception as e:
        logger.error(f"Error getting important dates: {e}", exc_info=True)
        return f"Error retrieving important dates: {e}"


async def search_important_dates_executor(args: dict, context: AgentToolContext = None) -> str:
    """Search for important dates by name or title."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    store = _get_store(context)
    if not store:
        return "Error: Important dates storage not configured"

    search_term = args.get("search_term", "")
    if not search_term:
        return "Error: Please provide a search term (name or event)"

    try:
        dates = await store.search_important_dates(
            user_id=context.tenant_id, search_term=search_term, limit=10,
        )
        if not dates:
            return f"No important dates found matching '{search_term}'."

        output = []
        for d in dates:
            title = d.get("title", "Event")
            date_str = d.get("date", "")
            recurring = d.get("recurring", True)

            if date_str:
                try:
                    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d") if isinstance(date_str, str) else date_str
                    date_display = dt.strftime("%B %d") if recurring else dt.strftime("%B %d, %Y")
                except Exception:
                    date_display = str(date_str)[:10]
            else:
                date_display = "Unknown date"
            output.append(f"- {title}: {date_display}")

        return f"Found {len(dates)} result(s):\n" + "\n".join(output)

    except Exception as e:
        logger.error(f"Error searching important dates: {e}", exc_info=True)
        return f"Error searching: {e}"


async def add_important_date_executor(args: dict, context: AgentToolContext = None) -> str:
    """Add a new important date."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    store = _get_store(context)
    if not store:
        return "Error: Important dates storage not configured"

    title = args.get("title", "")
    date = args.get("date", "")
    if not title or not date:
        return "Error: Both title and date are required"

    date_type = args.get("date_type", "custom")
    person_name = args.get("person_name")
    relationship = args.get("relationship")
    recurring = args.get("recurring", True)

    date_obj = _parse_date(date)
    if not date_obj:
        return f"Error: Could not parse date '{date}'. Please use format like 'March 15' or '2024-03-15'"

    date_data = {
        "title": title,
        "date": date_obj.strftime("%Y-%m-%d"),
        "date_type": date_type if date_type in ("birthday", "anniversary", "holiday", "custom") else "custom",
        "recurring": recurring,
        "remind_days_before": [0, 1, 7],
    }
    if person_name:
        date_data["person_name"] = person_name
    if relationship:
        date_data["relationship"] = relationship

    try:
        result = await store.create_important_date(context.tenant_id, date_data)
        if result:
            recurring_text = " (yearly)" if recurring else ""
            return f"Added: {title} on {date_obj.strftime('%B %d')}{recurring_text}"
        return "Error: Could not save the important date"
    except Exception as e:
        logger.error(f"Error adding important date: {e}", exc_info=True)
        return f"Error adding date: {e}"


async def update_important_date_executor(args: dict, context: AgentToolContext = None) -> str:
    """Update an existing important date."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    store = _get_store(context)
    if not store:
        return "Error: Important dates storage not configured"

    search_term = args.get("search_term", "")
    if not search_term:
        return "Error: Please specify which date to update"

    try:
        dates = await store.search_important_dates(context.tenant_id, search_term, limit=5)
        if not dates:
            return f"No important date found matching '{search_term}'"
        if len(dates) > 1:
            lines = ["Found multiple dates. Please be more specific:"]
            for d in dates:
                lines.append(f"- {d.get('title')}")
            return "\n".join(lines)

        date_to_update = dates[0]
        updates = {}

        new_date = args.get("new_date")
        if new_date:
            date_obj = _parse_date(new_date)
            if not date_obj:
                return f"Error: Could not parse new date '{new_date}'"
            updates["date"] = date_obj.strftime("%Y-%m-%d")

        new_title = args.get("new_title")
        if new_title:
            updates["title"] = new_title

        if not updates:
            return "No changes specified. Provide new_date or new_title."

        result = await store.update_important_date(context.tenant_id, date_to_update["id"], updates)
        if result:
            return f"Updated: {date_to_update.get('title')}"
        return "Error: Could not update the date"

    except Exception as e:
        logger.error(f"Error updating important date: {e}", exc_info=True)
        return f"Error updating: {e}"


async def delete_important_date_executor(args: dict, context: AgentToolContext = None) -> str:
    """Delete an important date."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    store = _get_store(context)
    if not store:
        return "Error: Important dates storage not configured"

    search_term = args.get("search_term", "")
    if not search_term:
        return "Error: Please specify which date to delete"

    try:
        dates = await store.search_important_dates(context.tenant_id, search_term, limit=5)
        if not dates:
            return f"No important date found matching '{search_term}'"
        if len(dates) > 1:
            lines = ["Found multiple dates. Please be more specific:"]
            for d in dates:
                lines.append(f"- {d.get('title')}")
            return "\n".join(lines)

        date_to_delete = dates[0]
        success = await store.delete_important_date(context.tenant_id, date_to_delete["id"])
        if success:
            return f"Deleted: {date_to_delete.get('title')}"
        return "Error: Could not delete the date"

    except Exception as e:
        logger.error(f"Error deleting important date: {e}", exc_info=True)
        return f"Error deleting: {e}"


_DATE_TYPE_EMOJI = {
    "birthday": "\U0001f382",    # 🎂
    "anniversary": "\U0001f48d", # 💍
    "holiday": "\U0001f389",     # 🎉
}
_DEFAULT_EMOJI = "\U0001f4c5"    # 📅


async def get_today_reminders_executor(args: dict, context: AgentToolContext = None) -> str:
    """Get today's important-date reminders for the daily digest."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    store = _get_store(context)
    if not store:
        return "Error: Important dates storage not configured"

    try:
        reminders = await store.get_today_reminders(user_id=context.tenant_id)
        if not reminders:
            return "No important-date reminders for today."

        output = []
        for d in reminders:
            emoji = _DATE_TYPE_EMOJI.get(d.get("date_type", ""), _DEFAULT_EMOJI)
            title = d.get("title", "Event")
            days_until = d.get("days_until", 0)
            person = d.get("person_name") or ""

            if days_until == 0:
                timing = "TODAY"
            elif days_until == 1:
                timing = "tomorrow"
            else:
                timing = f"in {days_until} days"

            line = f"{emoji} {title}: {timing}"
            if person and person.lower() not in title.lower():
                line += f" ({person})"
            output.append(line)

        return f"Important-date reminders ({len(reminders)}):\n" + "\n".join(output)

    except Exception as e:
        logger.error(f"Error getting today reminders: {e}", exc_info=True)
        return f"Error retrieving reminders: {e}"


# ---------------------------------------------------------------------------
# Schemas (used by orchestrator to build AgentTool instances)
# ---------------------------------------------------------------------------

IMPORTANT_DATES_TOOL_DEFS = [
    {
        "name": "get_important_dates",
        "description": "Get upcoming important dates like birthdays and anniversaries.",
        "parameters": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "How many days ahead to look (default: 60)",
                    "default": 60,
                },
            },
            "required": [],
        },
        "executor": get_important_dates_executor,
    },
    {
        "name": "search_important_dates",
        "description": "Search for a specific important date by person name or event title.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Name or event to search for (e.g., 'mom', 'anniversary')",
                },
            },
            "required": ["search_term"],
        },
        "executor": search_important_dates_executor,
    },
    {
        "name": "add_important_date",
        "description": "Add a new important date like a birthday or anniversary.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": 'Title of the event (e.g., "Mom\'s birthday")',
                },
                "date": {
                    "type": "string",
                    "description": "The date (e.g., 'March 15', '2024-03-15')",
                },
                "date_type": {
                    "type": "string",
                    "enum": ["birthday", "anniversary", "holiday", "custom"],
                    "description": "Type of date (default: custom)",
                },
                "person_name": {
                    "type": "string",
                    "description": "Name of the person (optional)",
                },
                "relationship": {
                    "type": "string",
                    "description": "Relationship to user (e.g., 'mother', 'spouse')",
                },
                "recurring": {
                    "type": "boolean",
                    "description": "Whether it repeats yearly (default: true)",
                },
            },
            "required": ["title", "date"],
        },
        "executor": add_important_date_executor,
    },
    {
        "name": "update_important_date",
        "description": "Update an existing important date's date or title.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Name or title to find the date to update",
                },
                "new_date": {
                    "type": "string",
                    "description": "New date (optional)",
                },
                "new_title": {
                    "type": "string",
                    "description": "New title (optional)",
                },
            },
            "required": ["search_term"],
        },
        "executor": update_important_date_executor,
    },
    {
        "name": "delete_important_date",
        "description": "Delete an important date.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Name or title of the date to delete",
                },
            },
            "required": ["search_term"],
        },
        "executor": delete_important_date_executor,
    },
    {
        "name": "get_today_reminders",
        "description": "Get today's important-date reminders for the daily digest. Returns dates whose remind_days_before window includes today.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "executor": get_today_reminders_executor,
    },
]
