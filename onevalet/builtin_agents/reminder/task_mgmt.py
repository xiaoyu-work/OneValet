"""
Task Management Agent - View, update, and delete scheduled reminders and automations

Allows users to manage their scheduled items through natural language:
- "What reminders do I have?"
- "Change my medicine reminder to 9am"
- "Don't remind me about the meeting anymore"
- "Pause my morning digest"

State Flow:
1. INITIALIZING -> extract action and parameters
2. WAITING_FOR_APPROVAL -> only for delete action
3. RUNNING -> execute the action
4. COMPLETED
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from onevalet import valet, StandardAgent, AgentStatus, AgentResult, Message
from .task_repo import TaskRepository

logger = logging.getLogger(__name__)


@valet()
class TaskManagementAgent(StandardAgent):
    """List, update, pause, or delete the user's scheduled reminders and automations."""

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self.action = None
        self.matched_task = None

    def needs_approval(self) -> bool:
        return self.action == "delete"

    def get_approval_prompt(self) -> str:
        """Generate confirmation prompt for delete action"""
        if self.matched_task:
            task_name = self.matched_task.get("name", "this item")
            return f"Delete '{task_name}'? (yes/no)"
        return "Are you sure you want to delete this? (yes/no)"

    def _get_repo(self):
        """Get TaskRepository from context_hints"""
        db = self.context_hints.get("db")
        if not db:
            return None
        if not hasattr(self, '_task_repo'):
            self._task_repo = TaskRepository(db)
        return self._task_repo

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract action and details from user input."""
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        weekday = now.strftime("%A")

        prompt = f"""Analyze this request about managing reminders or automations.

User message: "{user_input}"

Current time: {current_time_str}
Today is: {weekday}

The user wants to manage their scheduled reminders or automations (we call them "tasks" internally).
Users may say "reminder", "notification", "alert", "automation", "digest", etc. - these are all tasks.

Determine:
1. action: What the user wants to do
2. task_hint: Keywords to find which task (from name or description)
3. For updates: What to change

Return JSON:
{{
  "action": "list" | "show" | "update" | "pause" | "resume" | "delete",
  "task_hint": "<keywords to identify the task>",
  "status_filter": "active" | "paused" | "all",

  // For update action - include what's being changed:
  "new_schedule_datetime": "<ISO 8601 if changing time>",
  "new_schedule_type": "one_time" | "recurring",
  "new_cron_expression": "<if changing to recurring>",
  "new_message": "<if changing the reminder message>",
  "human_readable_time": "<friendly time description>",

  // What kind of update
  "update_type": "time" | "message" | "both" | null
}}

Examples:

1. "What reminders do I have?" / "Show my automations" / "List my notifications"
   {{"action": "list", "status_filter": "all"}}

2. "Change my medicine reminder to 9am"
   {{"action": "update", "task_hint": "medicine", "update_type": "time", "new_schedule_datetime": "<calculated 9am today or tomorrow>", "new_schedule_type": "one_time", "human_readable_time": "9am"}}

3. "Don't remind me about the meeting anymore" / "Cancel my meeting reminder"
   {{"action": "delete", "task_hint": "meeting"}}

4. "Stop my weather notifications" / "Pause the bitcoin updates"
   {{"action": "pause", "task_hint": "weather"}} or {{"action": "pause", "task_hint": "bitcoin"}}

5. "Turn my digest back on" / "Resume my morning notifications"
   {{"action": "resume", "task_hint": "digest"}}
"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an assistant that helps manage reminders and automations. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())
            self.action = extracted.get("action", "list")

            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"action": "list", "status_filter": "all"}

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Extract fields and determine next state"""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        if self.action == "delete":
            task_hint = self.collected_fields.get("task_hint", "")
            if task_hint:
                repo = self._get_repo()
                if not repo:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message="Task storage is not available right now."
                    )

                tasks = await repo.get_user_tasks(self.tenant_id)
                matched = self._match_tasks(tasks, task_hint)

                if not matched:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"I couldn't find anything matching '{task_hint}'."
                    )

                if len(matched) > 1:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=self._format_disambiguation(matched, "Which one should I delete?")
                    )

                self.matched_task = matched[0]
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_APPROVAL,
                    raw_message=self.get_approval_prompt()
                )

        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """Handle user's approval response for delete action"""
        user_input = msg.get_text() if msg else ""
        response_lower = user_input.lower().strip()

        if response_lower in ["yes", "y", "ok", "sure", "confirm", "do it"]:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif response_lower in ["no", "n", "cancel", "stop", "nevermind"]:
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Got it, I won't delete that."
            )

        else:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message="Please say 'yes' to delete or 'no' to cancel."
            )

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute the management action"""
        fields = self.collected_fields
        action = fields.get("action", "list")

        logger.info(f"TaskManagementAgent executing: {action}")

        if action == "list":
            response = await self._list_tasks(fields)
        elif action == "show":
            response = await self._show_task(fields)
        elif action == "update":
            response = await self._update_task(fields)
        elif action == "pause":
            response = await self._pause_task(fields)
        elif action == "resume":
            response = await self._resume_task(fields)
        elif action == "delete":
            response = await self._delete_task(fields)
        else:
            response = "I'm not sure what you want to do. Try 'show my reminders' or 'change my reminder to 9am'."

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=response
        )

    async def _list_tasks(self, fields: Dict[str, Any]) -> str:
        """List all user's reminders/automations"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        status_filter = fields.get("status_filter", "all")
        status_param = None if status_filter == "all" else status_filter
        tasks = await repo.get_user_tasks(self.tenant_id, status=status_param)

        if not tasks:
            if status_filter != "all":
                return f"You don't have any {status_filter} reminders or automations."
            return "You don't have any scheduled reminders or automations yet."

        lines = []
        for i, task in enumerate(tasks, 1):
            name = task.get("name", "Unnamed")
            status = task.get("status", "unknown")

            trigger_config = task.get("trigger_config", {})
            if trigger_config.get("cron"):
                schedule = self._format_cron(trigger_config["cron"])
            elif trigger_config.get("at"):
                schedule = self._format_datetime(trigger_config["at"])
            else:
                schedule = "scheduled"

            status_icon = "" if status == "active" else " (paused)" if status == "paused" else " (disabled)"
            lines.append(f"{i}. {name} - {schedule}{status_icon}")

        count = len(tasks)
        header = f"Your {count} reminder(s)/automation(s):"
        return header + "\n" + "\n".join(lines)

    async def _show_task(self, fields: Dict[str, Any]) -> str:
        """Show details of a specific task"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        task_hint = fields.get("task_hint", "")

        if not task_hint:
            return "Which reminder or automation would you like to see?"

        tasks = await repo.get_user_tasks(self.tenant_id)
        matched = self._match_tasks(tasks, task_hint)

        if not matched:
            return f"I couldn't find anything matching '{task_hint}'."

        if len(matched) > 1:
            return self._format_disambiguation(matched, "Which one do you mean?")

        task = matched[0]
        return self._format_task_details(task)

    async def _update_task(self, fields: Dict[str, Any]) -> str:
        """Update a task's schedule or message"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        task_hint = fields.get("task_hint", "")
        update_type = fields.get("update_type")

        if not task_hint:
            return "Which reminder or automation would you like to update?"

        tasks = await repo.get_user_tasks(self.tenant_id)
        matched = self._match_tasks(tasks, task_hint)

        if not matched:
            return f"I couldn't find anything matching '{task_hint}'."

        if len(matched) > 1:
            return self._format_disambiguation(matched, "Which one should I update?")

        task = matched[0]
        task_id = task.get("id")
        task_name = task.get("name", "item")

        try:
            update_data = {}
            response_parts = []

            if update_type in ("time", "both"):
                new_trigger_config = self._build_trigger_config(fields)
                if new_trigger_config:
                    update_data["trigger_config"] = new_trigger_config

                    if fields.get("new_cron_expression"):
                        update_data["trigger_type"] = "schedule"

                    time_desc = fields.get("human_readable_time", "new time")
                    response_parts.append(f"changed to {time_desc}")

            if update_type in ("message", "both"):
                new_message = fields.get("new_message")
                if new_message:
                    action_config = task.get("action_config", {})
                    action_config["message"] = new_message
                    update_data["action_config"] = action_config
                    update_data["name"] = new_message[:50]
                    update_data["description"] = f"Reminder: {new_message}"
                    response_parts.append("message updated")

            if not update_data:
                return "What would you like to change? (time, message, or both)"

            updated_task = await repo.update_task(task_id, update_data)

            if "trigger_config" in update_data and self.trigger_engine:
                await self.trigger_engine._teardown_task_trigger_by_id(task_id)

                if updated_task and updated_task.get("status") == "active":
                    from onevalet.triggers.models import Task
                    task_obj = Task.from_dict(updated_task)
                    await self.trigger_engine._setup_task_trigger(task_obj)

            response = f"Updated '{task_name}'"
            if response_parts:
                response += f" - {', '.join(response_parts)}"
            return response + "."

        except Exception as e:
            logger.error(f"Failed to update task: {e}", exc_info=True)
            return "Sorry, I couldn't update that. Please try again."

    async def _pause_task(self, fields: Dict[str, Any]) -> str:
        """Pause a task"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        task_hint = fields.get("task_hint", "")

        if task_hint == "all":
            return await self._pause_all_tasks()

        if not task_hint:
            return "Which reminder or automation would you like to pause?"

        tasks = await repo.get_user_tasks(self.tenant_id, status="active")
        matched = self._match_tasks(tasks, task_hint)

        if not matched:
            return f"I couldn't find an active item matching '{task_hint}'."

        if len(matched) > 1:
            return self._format_disambiguation(matched, "Which one should I pause?")

        task = matched[0]
        task_id = task.get("id")
        task_name = task.get("name", "item")

        try:
            await repo.update_task(task_id, {"status": "paused"})

            if self.trigger_engine:
                await self.trigger_engine._teardown_task_trigger_by_id(task_id)

            return f"Paused '{task_name}'. Say 'resume' when you want it back."

        except Exception as e:
            logger.error(f"Failed to pause task: {e}")
            return "Sorry, I couldn't pause that."

    async def _pause_all_tasks(self) -> str:
        """Pause all active tasks"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        tasks = await repo.get_user_tasks(self.tenant_id, status="active")

        if not tasks:
            return "You don't have any active reminders or automations to pause."

        paused_count = 0
        for task in tasks:
            task_id = task.get("id")
            try:
                await repo.update_task(task_id, {"status": "paused"})
                if self.trigger_engine:
                    await self.trigger_engine._teardown_task_trigger_by_id(task_id)
                paused_count += 1
            except Exception as e:
                logger.error(f"Failed to pause task {task_id}: {e}")

        return f"Paused {paused_count} item(s)."

    async def _resume_task(self, fields: Dict[str, Any]) -> str:
        """Resume a paused task"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        task_hint = fields.get("task_hint", "")

        if task_hint == "all":
            return await self._resume_all_tasks()

        if not task_hint:
            return "Which reminder or automation would you like to resume?"

        tasks = await repo.get_user_tasks(self.tenant_id, status="paused")
        matched = self._match_tasks(tasks, task_hint)

        if not matched:
            return f"I couldn't find a paused item matching '{task_hint}'."

        if len(matched) > 1:
            return self._format_disambiguation(matched, "Which one should I resume?")

        task = matched[0]
        task_id = task.get("id")
        task_name = task.get("name", "item")

        try:
            updated_task = await repo.update_task(task_id, {"status": "active"})

            if self.trigger_engine and updated_task:
                from onevalet.triggers.models import Task
                task_obj = Task.from_dict(updated_task)
                await self.trigger_engine._setup_task_trigger(task_obj)

            return f"Resumed '{task_name}'."

        except Exception as e:
            logger.error(f"Failed to resume task: {e}")
            return "Sorry, I couldn't resume that."

    async def _resume_all_tasks(self) -> str:
        """Resume all paused tasks"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        tasks = await repo.get_user_tasks(self.tenant_id, status="paused")

        if not tasks:
            return "You don't have any paused items to resume."

        resumed_count = 0
        for task in tasks:
            task_id = task.get("id")
            try:
                updated_task = await repo.update_task(task_id, {"status": "active"})
                if self.trigger_engine and updated_task:
                    from onevalet.triggers.models import Task
                    task_obj = Task.from_dict(updated_task)
                    await self.trigger_engine._setup_task_trigger(task_obj)
                resumed_count += 1
            except Exception as e:
                logger.error(f"Failed to resume task {task_id}: {e}")

        return f"Resumed {resumed_count} item(s)."

    async def _delete_task(self, fields: Dict[str, Any]) -> str:
        """Delete a task"""
        repo = self._get_repo()
        if not repo:
            return "Task storage is not available right now."

        task = self.matched_task

        if not task:
            task_hint = fields.get("task_hint", "")
            if not task_hint:
                return "Which reminder or automation would you like to delete?"

            tasks = await repo.get_user_tasks(self.tenant_id)
            matched = self._match_tasks(tasks, task_hint)

            if not matched:
                return f"I couldn't find anything matching '{task_hint}'."

            if len(matched) > 1:
                return self._format_disambiguation(matched, "Which one should I delete?")

            task = matched[0]

        task_id = task.get("id")
        task_name = task.get("name", "item")

        try:
            if self.trigger_engine:
                try:
                    await self.trigger_engine.delete_task(task_id)
                except Exception as e:
                    logger.warning(f"Failed to delete from trigger engine: {e}")

            await repo.delete_task(task_id)

            return f"Deleted '{task_name}'."

        except Exception as e:
            logger.error(f"Failed to delete task: {e}")
            return "Sorry, I couldn't delete that."

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_trigger_config(self, fields: Dict[str, Any]) -> Optional[Dict]:
        """Build trigger config from extracted fields"""
        try:
            if fields.get("new_cron_expression"):
                return {
                    "cron": fields["new_cron_expression"],
                }

            schedule_datetime = fields.get("new_schedule_datetime")
            if schedule_datetime:
                if 'T' in schedule_datetime:
                    local_dt = datetime.fromisoformat(schedule_datetime.replace('Z', '+00:00'))
                else:
                    local_dt = datetime.fromisoformat(schedule_datetime)

                return {
                    "at": local_dt.isoformat(),
                }

            return None

        except Exception as e:
            logger.error(f"Failed to build trigger config: {e}")
            return None

    def _match_tasks(self, tasks: List[Dict], hint: str) -> List[Dict]:
        """Match tasks by hint"""
        if not hint or not tasks:
            return []

        hint_lower = hint.lower()
        matched = []

        for task in tasks:
            name = (task.get("name") or "").lower()
            description = (task.get("description") or "").lower()

            action_config = task.get("action_config", {})
            message = (action_config.get("message") or "").lower()

            searchable = f"{name} {description} {message}"
            if hint_lower in searchable:
                matched.append(task)
            elif any(word in searchable for word in hint_lower.split()):
                matched.append(task)

        return matched

    def _format_disambiguation(self, tasks: List[Dict], prompt: str) -> str:
        """Format task list for disambiguation"""
        lines = [f"Found {len(tasks)} matches:"]
        for i, task in enumerate(tasks, 1):
            name = task.get("name", "Unnamed")
            trigger_config = task.get("trigger_config", {})

            if trigger_config.get("cron"):
                schedule = self._format_cron(trigger_config["cron"])
            elif trigger_config.get("at"):
                schedule = self._format_datetime(trigger_config["at"])
            else:
                schedule = ""

            lines.append(f"{i}. {name}" + (f" ({schedule})" if schedule else ""))

        lines.append(f"\n{prompt}")
        return "\n".join(lines)

    def _format_task_details(self, task: Dict) -> str:
        """Format detailed task view"""
        name = task.get("name", "Unnamed")
        status = task.get("status", "unknown")
        action_type = task.get("action_type", "")
        run_count = task.get("run_count", 0)

        type_label = "Reminder" if action_type == "notify" else "Automation"

        trigger_config = task.get("trigger_config", {})
        if trigger_config.get("cron"):
            schedule = self._format_cron(trigger_config["cron"])
        elif trigger_config.get("at"):
            schedule = self._format_datetime(trigger_config["at"])
        else:
            schedule = "scheduled"

        action_config = task.get("action_config", {})
        message = action_config.get("message", "")

        lines = [
            f"{type_label}: {name}",
            f"Schedule: {schedule}",
            f"Status: {status}",
            f"Runs: {run_count}",
        ]

        if message:
            lines.append(f"Message: {message}")

        return "\n".join(lines)

    def _format_cron(self, cron_expr: str) -> str:
        """Convert cron to human-readable"""
        parts = cron_expr.split()
        if len(parts) != 5:
            return cron_expr

        minute, hour, day, month, weekday = parts

        if day == "*" and month == "*" and weekday == "*":
            time_str = f"{hour}:{minute.zfill(2)}"
            return f"daily at {time_str}"

        if day == "*" and month == "*" and weekday != "*":
            days = {"0": "Sun", "1": "Mon", "2": "Tue", "3": "Wed", "4": "Thu", "5": "Fri", "6": "Sat"}
            day_name = days.get(weekday, weekday)
            return f"every {day_name} at {hour}:{minute.zfill(2)}"

        return cron_expr

    def _format_datetime(self, dt_str: str) -> str:
        """Format datetime for display"""
        try:
            if isinstance(dt_str, str):
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            else:
                dt = dt_str

            now = datetime.now(timezone.utc)

            if dt.date() == now.date():
                return f"today at {dt.strftime('%H:%M')}"
            elif (dt.date() - now.date()).days == 1:
                return f"tomorrow at {dt.strftime('%H:%M')}"
            else:
                return dt.strftime("%b %d at %H:%M")

        except Exception:
            return str(dt_str)[:16]
