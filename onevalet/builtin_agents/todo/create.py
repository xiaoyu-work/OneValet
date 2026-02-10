"""
Send Todo Agent - Creates new todo tasks

This agent handles task creation:
- Create tasks with title, due date, priority
- Auto-resolve target todo account
- Requires user approval before creating
"""
import logging
import json
from typing import Dict, Any

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult

logger = logging.getLogger(__name__)


@valet()
class CreateTodoAgent(StandardAgent):
    """Create todo task agent with field collection and approval"""

    title = InputField(
        prompt="What's the task?",
        description="Task title/description",
    )
    due = InputField(
        prompt="When is it due?",
        description="Due date for the task",
        required=False,
    )
    priority = InputField(
        prompt="What priority?",
        description="Task priority (low, medium, high, urgent)",
        required=False,
    )
    account = InputField(
        prompt="Which todo account?",
        description="Todo account to add the task to",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self._resolved_account = None

    def needs_approval(self) -> bool:
        return True

    async def parse_approval_async(self, user_input: str):
        """Parse user's approval response using LLM."""
        prompt = f"""The user was asked to approve creating a todo task. Their response was:
"{user_input}"

What is the user's intent?
- APPROVED: User wants to create the task (yes, ok, create it, go ahead, etc.)
- REJECTED: User wants to cancel (no, cancel, don't create, nevermind, etc.)
- MODIFY: User wants to change something (change the title, different date, etc.)

Return ONLY one word: APPROVED, REJECTED, or MODIFY"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            response = result.content.strip().upper()

            if "APPROVED" in response:
                return ApprovalResult.APPROVED
            elif "REJECTED" in response:
                return ApprovalResult.REJECTED
            else:
                return ApprovalResult.MODIFY
        except Exception as e:
            logger.error(f"Failed to parse approval: {e}")
            return ApprovalResult.MODIFY

    # ===== State Handlers =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Extract fields and resolve account."""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        # Resolve target account early
        await self._resolve_todo_account()

        # Check missing fields
        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - go to approval
        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """Continue collecting fields from user."""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing
            )

        # All fields collected - go to approval
        await self._resolve_todo_account()
        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """Handle yes/no/modify responses."""
        user_input = msg.get_text() if msg else ""
        approval = await self.parse_approval_async(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == ApprovalResult.REJECTED:
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="OK, cancelled."
            )

        else:  # MODIFY
            await self._extract_and_collect_fields(user_input)

            missing = self._get_missing_fields()
            if missing:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=self._get_next_prompt(),
                    missing_fields=missing
                )

            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute task creation."""
        from onevalet.providers.todo.resolver import TodoAccountResolver
        from onevalet.providers.todo.factory import TodoProviderFactory

        fields = self.collected_fields
        title = fields.get("title", "")
        due = fields.get("due")
        priority = fields.get("priority")

        logger.info(f"Creating task: {title}")

        try:
            # Resolve account if not already done
            if not self._resolved_account:
                await self._resolve_todo_account()

            if not self._resolved_account:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I couldn't find your todo account. Please connect one in settings."
                )

            account = self._resolved_account
            provider = TodoProviderFactory.create_provider(account)
            if not provider:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Sorry, I can't create tasks with that provider yet."
                )

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I lost access to your todo account. Please reconnect it in settings."
                )

            result = await provider.create_task(
                title=title,
                due=due,
                priority=priority
            )

            if result.get("success"):
                account_name = account.get("account_name", account.get("provider", ""))
                due_str = f" (due {due})" if due else ""
                logger.info(f"Task created on {account_name}: {title}")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Added to {account_name}: {title}{due_str}"
                )
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Task creation failed: {error_msg}")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't create the task: {error_msg}"
                )

        except Exception as e:
            logger.error(f"Failed to create task: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong creating your task. Want to try again?"
            )

    # ===== Helper Methods =====

    async def _resolve_todo_account(self):
        """Resolve the target todo account."""
        from onevalet.providers.todo.resolver import TodoAccountResolver

        account_spec = self.collected_fields.get("account", "primary")
        account = await TodoAccountResolver.resolve_account(self.tenant_id, account_spec)

        if account:
            self._resolved_account = account
            logger.info(f"Resolved todo account: {account.get('account_name')} ({account.get('email')})")
        else:
            logger.warning(f"No todo account found for tenant {self.tenant_id}")
            self._resolved_account = None

    def get_approval_prompt(self) -> str:
        """Generate task draft for user approval."""
        title = self.collected_fields.get("title", "")
        due = self.collected_fields.get("due", "")
        priority = self.collected_fields.get("priority", "")

        account_name = ""
        if self._resolved_account:
            account_name = self._resolved_account.get("account_name", self._resolved_account.get("provider", ""))

        parts = ["New task:"]
        parts.append(f"Title: {title}")
        if due:
            parts.append(f"Due: {due}")
        if priority:
            parts.append(f"Priority: {priority}")
        if account_name:
            parts.append(f"To: {account_name}")

        parts.append("\nCreate this task?")

        return "\n".join(parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract task information from user input using LLM."""
        extraction_prompt = f"""Extract todo task information from the user's message. The user may speak in any language.

User message: "{user_input}"

Return JSON with these fields (leave empty string if not mentioned):
{{
  "title": "",
  "due": "",
  "priority": "",
  "account": ""
}}

Rules:
- title: The task title or what needs to be done. Extract the core action.
- due: Due date if mentioned (e.g., "tomorrow", "Feb 10", "next week")
- priority: Priority level if mentioned (low, medium, high, urgent)
- account: Todo account name if specified (e.g., "todoist", "work", "personal")
- Do NOT add extra interpretation. Just extract what the user said."""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract task info. Return valid JSON only."},
                    {"role": "user", "content": extraction_prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            response_text = result.content.strip()
            if not response_text:
                return {}

            extracted = json.loads(response_text)
            result_dict = {}

            for field in ["title", "due", "priority", "account"]:
                value = extracted.get(field, "").strip()
                if value:
                    result_dict[field] = value

            logger.info(f"Extracted fields: {list(result_dict.keys())}")
            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}
