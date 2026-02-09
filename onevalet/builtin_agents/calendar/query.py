"""
Calendar Agent - Query and search calendar events

This agent handles calendar queries:
- Check upcoming events
- Search events by time range
- Search by keywords

This is a read-only agent (no approval needed).
"""
import logging
import json
import html
from typing import Dict, Any, List
from datetime import datetime, timedelta

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["check calendar", "my calendar", "what's on my calendar", "calendar", "upcoming events"])
class CalendarAgent(StandardAgent):
    """Calendar query agent - simple single-step agent"""

    time_range = InputField(
        prompt="What time range would you like to check?",
        description="Time range for events (e.g., 'today', 'tomorrow', 'this week')",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self.mentioned_other_account = False

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract calendar search criteria from user input"""
        calendar_keywords = [
            'work calendar', 'personal calendar', 'school calendar',
            'from my work', 'from my personal', 'in my work', 'on my work'
        ]

        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in calendar_keywords):
            self.mentioned_other_account = True
            return {}

        if not self.llm_client:
            return {"time_range": "today"}

        try:
            prompt = f"""Extract calendar search criteria from the user's message.

User message: "{user_input}"

Extract:
1. time_range: "today", "tomorrow", "this week", "next week", "this month", or specific dates
2. query: Keywords to search in event titles (optional)

Default to "today" for generic queries like "check my calendar".

Return JSON:
{{"time_range": "today", "query": ""}}

Return ONLY the JSON object:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract calendar search criteria. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            extracted = json.loads(content)

            if not extracted:
                extracted = {"time_range": "today"}

            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"time_range": "today"}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search calendar events"""
        from onevalet.providers.calendar.resolver import CalendarAccountResolver
        from onevalet.providers.calendar.factory import CalendarProviderFactory
        from .search_helper import parse_time_range

        fields = self.collected_fields

        if self.mentioned_other_account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I can only check your primary calendar right now."
            )

        try:
            account = CalendarAccountResolver.resolve_account(self.tenant_id, "primary")

            if not account:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No calendar account found. Please connect one first."
                )

            time_range_str = fields.get("time_range", "today")
            time_min, time_max = parse_time_range(time_range_str)
            query = fields.get("query")
            max_results = fields.get("max_results", 10)

            provider = CalendarProviderFactory.create_provider(account)
            if not provider:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Sorry, I can't access that calendar provider yet."
                )

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I lost access to your calendar. Could you reconnect it?"
                )

            result = await provider.list_events(
                time_min=time_min,
                time_max=time_max,
                max_results=max_results,
                query=query
            )

            if not result.get("success"):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Failed to search calendar: {result.get('error', 'Unknown error')}"
                )

            events = result.get("data", [])
            events.sort(key=lambda e: e.get("start") or datetime.min)

            formatted = self._format_calendar_results(events, time_range_str)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=formatted
            )

        except Exception as e:
            logger.error(f"Calendar search failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't check your calendar. Try again later?"
            )

    def _format_calendar_results(self, events: List[Dict], time_range: str) -> str:
        """Format calendar search results"""
        if not events:
            return f"No events found {time_range}."

        response_parts = [f"Found {len(events)} event(s) {time_range}:"]

        for i, event in enumerate(events[:5], 1):
            summary = html.unescape(event.get("summary", "No title"))
            start = event.get("start")
            location = event.get("location", "")

            if start:
                if isinstance(start, datetime):
                    start_str = start.strftime("%a %b %d, %I:%M %p")
                elif isinstance(start, dict):
                    dt_str = start.get("dateTime", start.get("date", ""))
                    try:
                        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                        start_str = dt.strftime("%a %b %d, %I:%M %p")
                    except (ValueError, AttributeError):
                        start_str = dt_str
                else:
                    start_str = str(start)
            else:
                start_str = "Unknown time"

            event_text = f"{i}. {start_str} - {summary}"
            if location:
                event_text += f" ({location})"

            response_parts.append(event_text)

        if len(events) > 5:
            response_parts.append(f"\n... and {len(events) - 5} more event(s).")

        return "\n".join(response_parts)
