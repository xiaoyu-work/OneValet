"""
Create Event Agent - Create calendar events

Multi-step agent for creating calendar events with field collection and approval.
"""
import logging
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["create event", "add event", "schedule event", "new event", "calendar event"])
class CreateEventAgent(StandardAgent):
    """Create calendar event agent with field collection and approval"""

    summary = InputField(
        prompt="What's the event title?",
        description="Event title/summary",
    )
    start = InputField(
        prompt="When does the event start?",
        description="Event start time (date and time)",
    )
    end = InputField(
        prompt="When does the event end?",
        description="Event end time",
        required=False,
    )
    description = InputField(
        prompt="Any description for the event?",
        description="Event description/details",
        required=False,
    )
    location = InputField(
        prompt="Where is the event?",
        description="Event location",
        required=False,
    )
    attendees = InputField(
        prompt="Who should be invited? (emails, optional)",
        description="Comma-separated list of attendee emails",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self.calendar_email = None
        self.calendar_account_name = None
        self.attendee_names = None
        self.mentioned_other_account = False

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        """Generate event draft for user approval"""
        summary = self.collected_fields.get("summary", "")
        start = self.collected_fields.get("start", "")
        end = self.collected_fields.get("end", "")
        description = self.collected_fields.get("description", "")
        location = self.collected_fields.get("location", "")
        attendees = self.collected_fields.get("attendees", "")

        calendar_email = self.calendar_email or "your-calendar"

        if not end and start:
            try:
                start_dt = self._parse_time_to_datetime(start)
                end_dt = start_dt + timedelta(hours=1)
                if end_dt.date() == start_dt.date():
                    end = end_dt.strftime("%I:%M %p").lstrip('0')
                else:
                    end = end_dt.strftime("%b %d at %I:%M %p").lstrip('0')
            except Exception:
                end = f"{start} + 1 hour"

        parts = ["Event Draft:"]
        parts.append(f"Calendar: {calendar_email}")
        parts.append(f"Title: {summary}")
        parts.append(f"Start: {start}")
        parts.append(f"End: {end}")

        if description:
            parts.append(f"Description: {description}")
        if location:
            parts.append(f"Location: {location}")
        if attendees:
            parts.append(f"Attendees: {attendees}")

        parts.append("---")
        parts.append("Looks good?")

        return "\n".join(parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract event information from user input"""
        calendar_keywords = [
            'work calendar', 'personal calendar', 'school calendar',
            'from my work', 'in my work', 'on my work'
        ]

        if any(kw in user_input.lower() for kw in calendar_keywords):
            self.mentioned_other_account = True
            return {}

        current_time = datetime.now()

        extraction_prompt = f"""Extract calendar event information from the user's message.

Current time: {current_time.strftime('%A, %B %d, %Y at %I:%M %p')}

User message: {user_input}

Return JSON:
{{
  "summary": "",
  "start": "",
  "end": "",
  "duration": null,
  "description": "",
  "location": "",
  "attendees": "",
  "attendee_names": ""
}}

Rules:
- summary: Event title
- start/end: Natural language times
- duration: Hours as number if mentioned (e.g., 1.0, 2.0)
- attendees: Comma-separated emails
- attendee_names: Names without emails (for follow-up)

Only extract explicitly stated information."""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract event information. Return only JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())
            result_dict = {}

            for field in ["summary", "start", "end", "description", "location", "attendees"]:
                value = extracted.get(field, "").strip()
                if value:
                    result_dict[field] = value

            attendee_names = extracted.get("attendee_names", "").strip()
            if attendee_names:
                self.attendee_names = attendee_names
                if not extracted.get("attendees") and "attendees" not in self.collected_fields:
                    for f in self.required_fields:
                        if f.name == "attendees":
                            f.required = True
                            f.prompt = f"What's {attendee_names}'s email address?"
                            break

            duration = extracted.get("duration")
            if duration and duration > 0:
                start_str = self.collected_fields.get("start", result_dict.get("start", ""))
                if start_str:
                    try:
                        start_dt = self._parse_time_to_datetime(start_str)
                        end_dt = start_dt + timedelta(hours=duration)
                        if end_dt.date() == start_dt.date():
                            result_dict["end"] = end_dt.strftime("%I:%M %p").lstrip('0')
                        else:
                            result_dict["end"] = end_dt.strftime("%b %d at %I:%M %p").lstrip('0')
                    except Exception:
                        pass

            elif "start" in result_dict and "end" not in result_dict and "end" not in self.collected_fields:
                try:
                    start_dt = self._parse_time_to_datetime(result_dict["start"])
                    end_dt = start_dt + timedelta(hours=1)
                    if end_dt.date() == start_dt.date():
                        result_dict["end"] = end_dt.strftime("%I:%M %p").lstrip('0')
                    else:
                        result_dict["end"] = end_dt.strftime("%b %d at %I:%M %p").lstrip('0')
                except Exception:
                    pass

            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    def _parse_time_to_datetime(self, time_str: str) -> datetime:
        """Parse natural language time to datetime"""
        from dateutil import parser as date_parser

        current_time = datetime.now()
        default_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

        try:
            return date_parser.parse(time_str, fuzzy=True, default=default_time)
        except Exception:
            raise ValueError(f"Could not parse time: {time_str}")

    async def on_initializing(self, msg: Message) -> AgentResult:
        if self.mentioned_other_account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I can only create events in your primary calendar right now."
            )

        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_required_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=missing[0].prompt
            )

        await self._resolve_calendar_account()

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_required_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=missing[0].prompt
            )

        await self._resolve_calendar_account()

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
                raw_message="Got it, I won't create the event."
            )

        else:
            await self._extract_and_collect_fields(user_input)

            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def _parse_approval_response(self, user_response: str) -> str:
        response_lower = user_response.lower().strip()
        if response_lower in ["yes", "y", "ok", "sure", "go", "confirm", "looks good", "perfect"]:
            return "approved"
        elif response_lower in ["no", "n", "cancel", "stop", "nevermind"]:
            return "rejected"
        else:
            return "modify"

    async def _resolve_calendar_account(self):
        try:
            from onevalet.providers.calendar.resolver import CalendarAccountResolver

            account = await CalendarAccountResolver.resolve_account(self.tenant_id, "primary")
            if account:
                self.calendar_account_name = account.get("account_name", "Unknown")
                self.calendar_email = account.get("account_identifier", "your-calendar")
        except Exception as e:
            logger.error(f"Failed to resolve calendar account: {e}")

    def _get_missing_required_fields(self):
        return [f for f in self.required_fields if f.required and f.name not in self.collected_fields]

    async def on_running(self, msg: Message) -> AgentResult:
        """Create the calendar event"""
        from onevalet.providers.calendar.resolver import CalendarAccountResolver
        from onevalet.providers.calendar.factory import CalendarProviderFactory

        fields = self.collected_fields
        summary = fields["summary"]
        start_str = fields["start"]

        start_dt = self._parse_time_to_datetime(start_str)
        end_str = fields.get("end")
        if end_str:
            end_dt = self._parse_time_to_datetime(end_str)
        else:
            end_dt = start_dt + timedelta(hours=1)

        description = fields.get("description")
        location = fields.get("location")
        attendees_str = fields.get("attendees")

        attendees = []
        if attendees_str:
            attendees = [email.strip() for email in attendees_str.split(",") if email.strip()]

        try:
            account = await CalendarAccountResolver.resolve_account(self.tenant_id, "primary")

            if not account:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No calendar connected. Please connect one first."
                )

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

            result = await provider.create_event(
                summary=summary,
                start=start_dt,
                end=end_dt,
                description=description,
                location=location,
                attendees=attendees
            )

            if result.get("success"):
                event_link = result.get("html_link", "")
                response = f"Done! I've added '{summary}' to your calendar."
                if event_link:
                    response += f"\n{event_link}"
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=response
                )
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I couldn't create that event. {result.get('error', '')}"
                )

        except Exception as e:
            logger.error(f"Failed to create event: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong. Want me to try again?"
            )
