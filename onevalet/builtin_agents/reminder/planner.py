"""
Planner Agent - Creates structured automation tasks from user requests

Handles complex automation requests that need multiple steps:
- "Every day at 8am give me bitcoin price and White House news"
- "When I receive an Amazon email, notify me"
- "Every Monday morning give me this week's calendar summary"

Outputs structured Task definitions for TriggerEngine.

State Flow:
1. INITIALIZING -> extract fields
2. RUNNING -> create automation task
3. COMPLETED
"""
import logging
import json
from typing import Dict, Any, List
from datetime import datetime

from onevalet import valet, StandardAgent, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class PlannerAgent(StandardAgent):
    """Create automated workflows and scheduled tasks. Use when the user wants to set up recurring actions or complex automations."""

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract automation task structure from user input."""
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now.strftime("%A")

        available_agents = self._get_available_agents_for_prompt()

        prompt = f"""Analyze this automation request and create a structured task definition.

User message: "{user_input}"

Current time information:
- Current time: {current_time_str}
- Today is: {weekday}

Available agents for pipeline steps:
{available_agents}

Your task:
1. Understand what the user wants automated
2. Determine the trigger type (schedule or event)
3. Determine the action type (pipeline for multi-step, notify for simple)
4. Create a structured task definition

Return JSON with the following structure:
{{
    "task_name": "<descriptive name>",
    "task_description": "<what this automation does>",

    "trigger": {{
        "type": "schedule" | "event",
        "config": {{
            // For schedule trigger:
            "cron": "<cron expression>",  // For recurring (e.g., "0 8 * * *" for daily 8am)
            "at": "<ISO datetime>",  // For one-time (e.g., "2024-01-15T14:30:00")

            // For event trigger:
            "source": "email" | "webhook",
            "event": "received",
            "filter": {{"sender_contains": "amazon"}}  // Optional filter
        }}
    }},

    "action": {{
        "type": "pipeline" | "notify",
        "config": {{
            // For pipeline action:
            "steps": [
                {{"agent": "<AgentName>", "input": "<what to do>", "output_key": "<key>"}},
                ...
            ],
            "output_template": "<Jinja2 template using step outputs>",
            "condition": "<optional: only notify if this condition is met>"

            // For notify action:
            "message": "<notification message>"
        }}
    }},

    "output": {{
        "channel": "sms"
    }},

    "human_readable_summary": "<friendly description of what was set up>"
}}

IMPORTANT:
- Cron expression uses 5 fields: minute hour day month weekday
- For schedule, calculate the correct cron based on user's request
- Choose the right agents for each step based on the available agents list
- Use meaningful output_key names
- Create clear output_template
- Use "condition" when user wants to be notified ONLY IF something happens (price drop, delays, etc.)
"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an automation planning assistant. Create structured task definitions from user requests. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())
            extracted["_original_input"] = user_input

            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}", exc_info=True)
            return {"error": str(e)}

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute: create the task in TriggerEngine"""
        fields = self.collected_fields

        if "error" in fields:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, I couldn't understand that automation request. Could you try again?"
            )

        if not self.trigger_engine:
            logger.error("TriggerEngine not available")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, I can't create automations right now. Please try again later."
            )

        try:
            task_def = {
                "name": fields.get("task_name", "Automation Task"),
                "description": fields.get("task_description", ""),
                "trigger": fields.get("trigger", {}),
                "action": fields.get("action", {}),
                "output": fields.get("output", {"channel": "sms"}),
                "metadata": {
                    "original_request": fields.get("_original_input", ""),
                    "created_by": "PlannerAgent",
                    "human_readable_summary": fields.get("human_readable_summary", "")
                }
            }

            task = await self.trigger_engine.create_task(
                user_id=self.tenant_id,
                task_def=task_def
            )

            logger.info(f"Created automation task: {task.id}")

            summary = fields.get("human_readable_summary", f"Created automation: {task_def['name']}")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Got it! {summary}"
            )

        except Exception as e:
            logger.error(f"Failed to create automation task: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, I couldn't set up that automation. Could you try again?"
            )

    def _get_available_agents_for_prompt(self) -> str:
        """Get list of available agents for the LLM prompt"""
        pipeline_capable = [
            "WebSearchAgent: Search the web for information",
            "WeatherAgent: Get weather forecasts",
            "CalendarAgent: Query calendar events",
            "ReadEmailAgent: Read and search emails",
            "MapSearchAgent: Search for places on the map",
            "ShipmentAgent: Track package deliveries",
            "FlightSearchAgent: Search for flights"
        ]

        return "\n".join(f"- {agent}" for agent in pipeline_capable)
