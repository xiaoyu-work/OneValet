"""
Update Event Agent - Modify/reschedule calendar events

Multi-step agent for updating calendar events based on user's instructions.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["update event", "change event", "reschedule event", "move event", "modify event"])
class UpdateEventAgent(StandardAgent):
    """Update calendar event agent with field collection and approval"""

    target = InputField(
        prompt="Which event would you like to update?",
        description="Which event to update: event title, time, or keywords to identify it",
    )
    changes = InputField(
        prompt="What would you like to change?",
        description="What to change: new time, new title, new location, etc.",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.target_event = None
        self.account = None
        self.changes = {}
        self._event_found = False

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        """Generate update preview for approval"""
        if not self.target_event or not self.changes:
            return "I couldn't find the event or determine what to change."

        event_title = self.target_event.get("summary", "Untitled event")

        response_parts = [f"I'll update \"{event_title}\":"]

        if "summary" in self.changes:
            response_parts.append(f"  Title: {self.target_event.get('summary')} -> {self.changes['summary']}")

        if "start" in self.changes or "end" in self.changes:
            old_start = self.target_event.get("start", {})
            old_time = old_start.get("dateTime", old_start.get("date", "Unknown"))
            if "start" in self.changes:
                new_time = self.changes["start"].strftime("%Y-%m-%d %H:%M") if isinstance(self.changes["start"], datetime) else str(self.changes["start"])
                response_parts.append(f"  Time: {old_time[:16]} -> {new_time}")

        if "location" in self.changes:
            old_loc = self.target_event.get("location", "None")
            response_parts.append(f"  Location: {old_loc} -> {self.changes['location']}")

        if "description" in self.changes:
            response_parts.append(f"  Description: (updated)")

        response_parts.append("\nMake these changes?")

        return "\n".join(response_parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract target event and changes from user input"""
        if not self.llm_client:
            return {}

        now = datetime.now()

        prompt = f"""Extract the target event and requested changes from this user message about updating a calendar event.

Current time: {now.strftime('%Y-%m-%d %H:%M')}

User message: "{user_input}"

Return JSON with:
- "target": keywords to identify the event (title, person's name, time reference like "my 2pm meeting")
- "changes": object with what to change:
  - "new_time": new start time if rescheduling (e.g., "3pm", "tomorrow at 2pm")
  - "new_title": new event title if renaming
  - "new_location": new location if changing location
  - "new_duration": new duration if changing length (e.g., "2 hours")

Return only valid JSON:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an information extraction assistant. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            return json.loads(result.content.strip())
        except Exception as e:
            logger.error(f"Failed to extract update fields: {e}")
            return {}

    async def _find_target_event(self, target: str) -> bool:
        """Find the event to update"""
        from onevalet.providers.calendar.resolver import CalendarAccountResolver
        from onevalet.providers.calendar.factory import CalendarProviderFactory

        self.account = CalendarAccountResolver.resolve_account(self.tenant_id, "primary")
        if not self.account:
            return False

        provider = CalendarProviderFactory.create_provider(self.account)
        if not provider:
            return False

        if not await provider.ensure_valid_token():
            return False

        now = datetime.now()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=30)).isoformat() + "Z"

        result = await provider.list_events(
            time_min=time_min,
            time_max=time_max,
            query=target,
            max_results=10
        )

        if not result.get("success") or not result.get("data"):
            result = await provider.list_events(
                time_min=time_min,
                time_max=time_max,
                max_results=20
            )

        if result.get("success") and result.get("data"):
            events = result["data"]
            target_lower = target.lower()

            for event in events:
                event_title = event.get("summary", "").lower()
                event_desc = event.get("description", "").lower()
                if target_lower in event_title or target_lower in event_desc:
                    self.target_event = event
                    return True

            if events and any(word in target_lower for word in ["meeting", "call", "sync", "appointment"]):
                self.target_event = events[0]
                return True

        return False

    async def _parse_changes(self, changes_input: Dict) -> Dict:
        """Parse user's requested changes into actual values"""
        parsed = {}

        if "new_time" in changes_input and changes_input["new_time"]:
            new_time_str = changes_input["new_time"]
            parsed_time = await self._parse_datetime(new_time_str)
            if parsed_time:
                parsed["start"] = parsed_time
                if self.target_event:
                    old_start = self.target_event.get("start", {})
                    old_end = self.target_event.get("end", {})
                    if old_start.get("dateTime") and old_end.get("dateTime"):
                        try:
                            old_start_dt = datetime.fromisoformat(old_start["dateTime"].replace("Z", "+00:00"))
                            old_end_dt = datetime.fromisoformat(old_end["dateTime"].replace("Z", "+00:00"))
                            duration = old_end_dt - old_start_dt
                            parsed["end"] = parsed_time + duration
                        except Exception:
                            parsed["end"] = parsed_time + timedelta(hours=1)
                    else:
                        parsed["end"] = parsed_time + timedelta(hours=1)
                else:
                    parsed["end"] = parsed_time + timedelta(hours=1)

        if "new_title" in changes_input and changes_input["new_title"]:
            parsed["summary"] = changes_input["new_title"]

        if "new_location" in changes_input and changes_input["new_location"]:
            parsed["location"] = changes_input["new_location"]

        if "new_duration" in changes_input and changes_input["new_duration"]:
            duration_str = changes_input["new_duration"].lower()
            hours = 1
            if "hour" in duration_str:
                match = re.search(r'(\d+)', duration_str)
                if match:
                    hours = int(match.group(1))
            if "start" in parsed:
                parsed["end"] = parsed["start"] + timedelta(hours=hours)

        return parsed

    async def _parse_datetime(self, time_str: str) -> Optional[datetime]:
        """Parse natural language time string to datetime"""
        if not self.llm_client:
            return None

        now = datetime.now()

        prompt = f"""Parse this time expression into an ISO datetime.

Current time: {now.strftime('%Y-%m-%d %H:%M')}
Time expression: "{time_str}"

Return ONLY the datetime in ISO format (YYYY-MM-DDTHH:MM:SS), nothing else.
If the expression is relative (like "3pm"), assume it means today or the next occurrence.

Output:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You parse time expressions into ISO datetime format. Return ONLY the datetime string, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                enable_thinking=False
            )

            dt_str = result.content.strip()
            dt_str = dt_str.strip('"\'')
            return datetime.fromisoformat(dt_str)
        except Exception as e:
            logger.error(f"Failed to parse datetime: {e}")
            return None

    async def on_initializing(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_required_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=missing[0].prompt
            )

        target = self.collected_fields.get("target", "")
        changes_input = self.collected_fields.get("changes", {})

        if not await self._find_target_event(target):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I couldn't find an event matching '{target}'. Could you be more specific?"
            )

        self.changes = await self._parse_changes(changes_input)
        if not self.changes:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I'm not sure what changes you want to make. Could you be more specific?"
            )

        self._event_found = True

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

        target = self.collected_fields.get("target", "")
        changes_input = self.collected_fields.get("changes", {})

        if not self._event_found:
            if not await self._find_target_event(target):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I couldn't find an event matching '{target}'. Could you be more specific?"
                )
            self._event_found = True

        if not self.changes:
            self.changes = await self._parse_changes(changes_input)

        if not self.changes:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I'm not sure what changes you want to make. Could you be more specific?"
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
                raw_message="Got it, I won't make those changes."
            )

        else:
            await self._extract_and_collect_fields(user_input)
            changes_input = self.collected_fields.get("changes", {})
            self.changes = await self._parse_changes(changes_input)

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

    def _get_missing_required_fields(self):
        return [f for f in self.required_fields if f.required and f.name not in self.collected_fields]

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute the event update"""
        from onevalet.providers.calendar.factory import CalendarProviderFactory

        if not self.target_event:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I couldn't find the event to update."
            )

        if not self.changes:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I'm not sure what changes to make."
            )

        if not self.account:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I don't see a calendar connected. Could you connect one in settings?"
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

            event_id = self.target_event.get("id")
            if not event_id:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I couldn't identify the event to update."
                )

            result = await provider.update_event(
                event_id=event_id,
                summary=self.changes.get("summary"),
                start=self.changes.get("start"),
                end=self.changes.get("end"),
                location=self.changes.get("location"),
                description=self.changes.get("description")
            )

            if result.get("success"):
                event_title = self.changes.get("summary") or self.target_event.get("summary", "event")

                if "start" in self.changes:
                    new_time = self.changes["start"].strftime("%Y-%m-%d %H:%M") if isinstance(self.changes["start"], datetime) else str(self.changes["start"])
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"Done! I've moved \"{event_title}\" to {new_time}."
                    )
                elif "summary" in self.changes:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"Done! I've renamed the event to \"{event_title}\"."
                    )
                else:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"Done! I've updated \"{event_title}\"."
                    )
            else:
                error_msg = result.get("error", "Unknown error")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I couldn't update that event. {error_msg}"
                )

        except Exception as e:
            logger.error(f"Failed to update event: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong. Want me to try again?"
            )
