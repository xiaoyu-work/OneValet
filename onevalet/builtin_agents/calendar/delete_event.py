"""
Delete Event Agent - Delete calendar events

Multi-step agent for deleting calendar events with search and approval.
"""
import logging
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class DeleteEventAgent(StandardAgent):
    """Delete a calendar event. Use when the user wants to cancel or remove an event."""

    search_query = InputField(
        prompt="What events would you like to delete?",
        description="Search query for events to delete (title keywords, time range)",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.found_events = []
        self.event_ids = []
        self.account = None
        self.time_range = "next 7 days"
        self.mentioned_other_account = False
        self._search_completed = False

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        """Generate confirmation prompt with list of events to delete"""
        events = self.found_events

        if not events:
            return "Couldn't find any events matching that."

        if len(events) == 1:
            response_parts = ["Found 1 event:"]
        else:
            response_parts = [f"Found {len(events)} events:"]

        for event in events[:5]:
            summary = event.get("summary", "No title")
            start = event.get("start")
            location = event.get("location", "")

            if start:
                if isinstance(start, datetime):
                    start_str = start.strftime("%a %b %d, %I:%M%p").lstrip('0').lower()
                elif isinstance(start, dict):
                    dt_str = start.get("dateTime", start.get("date", ""))
                    try:
                        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                        start_str = dt.strftime("%a %b %d, %I:%M%p").lstrip('0').lower()
                    except (ValueError, AttributeError):
                        start_str = dt_str
                else:
                    start_str = str(start)
            else:
                start_str = "Unknown time"

            event_line = f"- {summary} - {start_str}"
            if location:
                event_line += f" @ {location}"

            response_parts.append(event_line)

        if len(events) > 5:
            response_parts.append(f"...and {len(events) - 5} more")

        response_parts.append("")
        if len(events) == 1:
            response_parts.append("Delete it?")
        else:
            response_parts.append("Delete these?")

        return "\n".join(response_parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract search criteria from user input"""
        calendar_keywords = [
            'work calendar', 'personal calendar', 'school calendar',
            'from my work', 'from my personal', 'in my work', 'on my work'
        ]

        user_input_lower = user_input.lower()
        if any(keyword in user_input_lower for keyword in calendar_keywords):
            self.mentioned_other_account = True
            return {}

        if not self.llm_client:
            return {"search_query": user_input}

        current_time = datetime.now()

        extraction_prompt = f"""Extract calendar event deletion criteria from the user's request:

Current date and time: {current_time.strftime('%A, %B %d, %Y at %I:%M %p')}

User request: {user_input}

Return JSON:
{{
  "search_query": "",
  "time_range": ""
}}

Rules:
- search_query: extract search keywords (event title, keywords in title)
- time_range: time range to search (e.g., "today", "tomorrow", "this week", "next 7 days")
  - ONLY extract if user explicitly mentions a time range
  - Leave EMPTY if not mentioned"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an information extraction assistant. Return only JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())
            result_dict = {}

            search_query = extracted.get("search_query", "").strip()
            if search_query:
                result_dict["search_query"] = search_query

            time_range = extracted.get("time_range", "").strip()
            if time_range:
                result_dict["time_range"] = time_range
                self.time_range = time_range

            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def _search_events(self) -> None:
        """Search for events matching the criteria"""
        from .search_helper import search_calendar_events

        search_query = self.collected_fields.get("search_query", "")
        time_range = self.collected_fields.get("time_range", self.time_range)

        try:
            result = await search_calendar_events(
                user_id=self.tenant_id,
                search_query=search_query,
                time_range=time_range,
                max_results=50,
                account_hint="primary"
            )

            if result.get("success"):
                self.found_events = result.get("events", [])
                self.account = result.get("account")
                self.event_ids = [event.get("event_id") for event in self.found_events if event.get("event_id")]

        except Exception as e:
            logger.error(f"Failed to search events: {e}", exc_info=True)

    async def on_initializing(self, msg: Message) -> AgentResult:
        if self.mentioned_other_account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I can only delete events from your primary calendar right now."
            )

        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        if self.mentioned_other_account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I can only delete events from your primary calendar right now."
            )

        if not self._search_completed:
            await self._search_events()
            self._search_completed = True

        if not self.found_events and not self.collected_fields.get("search_query"):
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="What events would you like to delete? Please specify the event title or time."
            )

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        if self.mentioned_other_account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I can only delete events from your primary calendar right now."
            )

        await self._search_events()
        self._search_completed = True

        if not self.found_events:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="I couldn't find any events matching that. Try different keywords?"
            )

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        user_input = msg.get_text() if msg else ""
        approval = await self._parse_approval_response(user_input)

        if approval == "approved":
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == "rejected":
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Got it, I won't delete those events."
            )

        else:
            self._search_completed = False
            self.found_events = []
            self.event_ids = []

            await self._extract_and_collect_fields(user_input)

            if self.collected_fields.get("search_query"):
                await self._search_events()
                self._search_completed = True

                if self.found_events:
                    return self.make_result(
                        status=AgentStatus.WAITING_FOR_APPROVAL,
                        raw_message=self.get_approval_prompt()
                    )

            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="What events would you like to delete instead?"
            )

    async def _parse_approval_response(self, user_response: str) -> str:
        response_lower = user_response.lower().strip()
        if response_lower in ["yes", "y", "ok", "sure", "go", "confirm", "do it", "delete"]:
            return "approved"
        elif response_lower in ["no", "n", "cancel", "stop", "nevermind"]:
            return "rejected"
        else:
            return "modify"

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute event deletion"""
        from onevalet.providers.calendar.factory import CalendarProviderFactory

        if not self.event_ids:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I couldn't find any events to delete."
            )

        if not self.account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I'm not sure which calendar to use."
            )

        try:
            provider = CalendarProviderFactory.create_provider(self.account)
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

            deleted_count = 0
            failed_count = 0

            for event_id in self.event_ids:
                result = await provider.delete_event(event_id)

                if result.get("success"):
                    deleted_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to delete event {event_id}: {result.get('error')}")

            if deleted_count == 0:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I couldn't delete those events. They might have already been removed."
                )
            elif failed_count > 0:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Done! I deleted {deleted_count} event(s), but {failed_count} couldn't be removed."
                )
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Done! I've removed {deleted_count} event(s) from your calendar."
                )

        except Exception as e:
            logger.error(f"Failed to delete events: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong. Want me to try again?"
            )
