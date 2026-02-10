"""
Reminder Agent - Create reminders and scheduled notifications

Only handles CREATING new reminders:
- One-time: "Remind me in 5 minutes to call John"
- Recurring: "Every day at 8am remind me to take medicine"

State Flow:
1. INITIALIZING -> extract reminder details
2. RUNNING -> create reminder via TriggerEngine
3. COMPLETED
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from onevalet import valet, StandardAgent, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class ReminderAgent(StandardAgent):
    """Creates new reminders using TriggerEngine - simple single-step agent (no approval needed)"""

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract reminder details from user input."""
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now.strftime("%A")

        prompt = f"""Extract reminder details and calculate the exact datetime.

User message: "{user_input}"

Current time: {current_time_str}
Today is: {weekday}

Calculate the EXACT datetime when the reminder should fire.
For recurring reminders, generate a cron expression.

Return JSON:
{{
  "schedule_datetime": "<ISO 8601 format, e.g., 2024-01-15T14:30:00>",
  "schedule_type": "one_time" | "recurring",
  "cron_expression": "<cron if recurring, e.g., '0 8 * * *' for daily 8am>",
  "reminder_message": "<what to remind about>",
  "human_readable_time": "<friendly description like 'in 5 minutes' or 'tomorrow at 3pm'>"
}}

Examples:
- "Remind me in 5 minutes to drink water"
  {{"schedule_datetime": "<now + 5 min>", "schedule_type": "one_time", "reminder_message": "drink water", "human_readable_time": "in 5 minutes"}}

- "Every day at 8am remind me to take medicine"
  {{"schedule_datetime": "<next 8am>", "schedule_type": "recurring", "cron_expression": "0 8 * * *", "reminder_message": "take medicine", "human_readable_time": "every day at 8am"}}"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a time calculation assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())

            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def on_running(self, msg: Message) -> AgentResult:
        """Create the reminder"""
        if not self.trigger_engine:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, I can't create reminders right now. Please try again later."
            )

        fields = self.collected_fields
        schedule_datetime = fields.get("schedule_datetime")
        schedule_type = fields.get("schedule_type", "one_time")
        cron_expression = fields.get("cron_expression")
        reminder_message = fields.get("reminder_message", "")
        human_readable_time = fields.get("human_readable_time", "")

        if not schedule_datetime:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="When would you like me to remind you?"
            )

        if not reminder_message:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="What would you like me to remind you about?"
            )

        trigger_config = self._build_trigger_config(
            schedule_datetime, schedule_type, cron_expression
        )

        if not trigger_config:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I couldn't process that time. Could you try again?"
            )

        try:
            task = await self.trigger_engine.create_task(
                user_id=self.tenant_id,
                task_def={
                    "name": reminder_message[:50],
                    "description": f"Reminder: {reminder_message}",
                    "trigger": {
                        "type": "schedule",
                        "config": trigger_config
                    },
                    "action": {
                        "type": "notify",
                        "config": {
                            "message": f"Reminder: {reminder_message}"
                        }
                    },
                    "output": {
                        "channel": "sms"
                    },
                    "metadata": {
                        "created_by": "ReminderAgent",
                        "human_readable_time": human_readable_time
                    }
                }
            )

            logger.info(f"Created reminder: {task.id}")

            time_desc = human_readable_time or schedule_datetime
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Got it! I'll remind you {time_desc}: {reminder_message}"
            )

        except Exception as e:
            logger.error(f"Failed to create reminder: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, I couldn't set up that reminder. Please try again."
            )

    def _build_trigger_config(
        self,
        schedule_datetime: str,
        schedule_type: str,
        cron_expression: Optional[str],
    ) -> Optional[Dict]:
        """Build trigger config from extracted fields"""
        try:
            if 'T' in schedule_datetime:
                local_dt = datetime.fromisoformat(schedule_datetime.replace('Z', '+00:00'))
            else:
                local_dt = datetime.fromisoformat(schedule_datetime)

            if schedule_type == "recurring" and cron_expression:
                return {
                    "cron": cron_expression,
                }
            else:
                return {
                    "at": local_dt.isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to build trigger config: {e}")
            return None
