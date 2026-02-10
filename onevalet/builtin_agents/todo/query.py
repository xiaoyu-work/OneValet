"""
Todo Query Agent - Query and search todo tasks

This agent handles todo/task queries:
- List all pending tasks
- Search tasks by keyword
- Show tasks across multiple providers (Todoist, Google Tasks, Microsoft To Do)

This is a read-only agent (no approval needed).
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class TodoQueryAgent(StandardAgent):
    """Todo search and query agent"""

    search_query = InputField(
        prompt="What tasks are you looking for?",
        description="Search query for tasks",
        required=False,
    )

    def __init__(
        self,
        tenant_id: str = "",
        llm_client=None,
        **kwargs
    ):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract search criteria from user input"""
        if not self.llm_client:
            return {"search_query": user_input}

        try:
            prompt = f"""Extract todo/task search criteria from the user's message.

User message: "{user_input}"

Extract the following information if present:
1. search_query: Keywords or description of tasks to find
2. show_completed: Whether to include completed tasks (true/false, default false)

Return ONLY the JSON object, no explanations.

JSON Output:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract task search criteria from text and return JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            extracted = json.loads(content)

            if not extracted:
                extracted = {"search_query": user_input}
            else:
                if "search_query" not in extracted:
                    extracted["search_query"] = None

            logger.info(f"Extracted todo search criteria: {extracted}")
            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}", exc_info=True)
            return {"search_query": user_input}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search tasks based on extracted criteria"""
        from onevalet.providers.todo.resolver import TodoAccountResolver
        from onevalet.providers.todo.factory import TodoProviderFactory

        fields = self.collected_fields
        logger.info(f"Searching tasks with criteria: {fields}")

        try:
            accounts = await TodoAccountResolver.resolve_accounts(self.tenant_id, ["all"])

            if not accounts:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No todo accounts found. Please connect one first."
                )

            all_tasks = []
            failed_accounts = []

            for account in accounts:
                provider = TodoProviderFactory.create_provider(account)
                if not provider:
                    failed_accounts.append({
                        "account_name": account.get("account_name", ""),
                        "email": account.get("email", ""),
                        "reason": "unsupported_provider"
                    })
                    continue

                if not await provider.ensure_valid_token():
                    failed_accounts.append({
                        "account_name": account.get("account_name", ""),
                        "email": account.get("email", ""),
                        "reason": "token_expired"
                    })
                    continue

                try:
                    search_query = fields.get("search_query")
                    show_completed = fields.get("show_completed", False)

                    # Skip meta keywords
                    meta_keywords = {"todo", "todos", "tasks", "task", "my tasks", "all", "list", "pending"}
                    if search_query and search_query.lower() in meta_keywords:
                        search_query = None

                    if search_query:
                        result = await provider.search_tasks(query=search_query)
                    else:
                        result = await provider.list_tasks(completed=show_completed)

                    if result.get("success"):
                        tasks = result.get("data", [])
                        for task in tasks:
                            task["_provider"] = account.get("provider", "")
                            task["_account_name"] = account.get("account_name", "")
                            task["_account_email"] = account.get("email", "")
                        all_tasks.extend(tasks)
                    else:
                        failed_accounts.append({
                            "account_name": account.get("account_name", ""),
                            "email": account.get("email", ""),
                            "reason": "search_failed",
                            "error": result.get("error", "Unknown error")
                        })

                except Exception as e:
                    logger.error(f"Failed to query {account.get('account_name')}: {e}", exc_info=True)
                    failed_accounts.append({
                        "account_name": account.get("account_name", ""),
                        "email": account.get("email", ""),
                        "reason": "query_failed",
                        "error": str(e)
                    })

            # Sort by due date (None dates last)
            all_tasks.sort(key=lambda t: t.get("due") or "9999-12-31")

            # Format output
            formatted = self._format_task_results(all_tasks, accounts, failed_accounts)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=formatted
            )

        except Exception as e:
            logger.error(f"Task search failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't search your tasks. Mind trying again later?"
            )

    def _format_task_results(
        self,
        tasks: List[Dict],
        searched_accounts: List[Dict],
        failed_accounts: List[Dict]
    ) -> str:
        """Format task search results"""
        if not tasks and not failed_accounts:
            return "You're all caught up - no tasks found!"

        response_parts = []
        multi_provider = len(searched_accounts) > 1

        if not tasks:
            response_parts.append("No tasks found.")
        else:
            response_parts.append(f"Found {len(tasks)} task(s):\n")

            for i, task in enumerate(tasks, 1):
                title = task.get("title", "Untitled")
                due = task.get("due")
                priority = task.get("priority")
                completed = task.get("completed", False)

                # Format due date
                due_str = ""
                if due:
                    due_str = self._format_due_date(due)

                # Format priority
                priority_str = ""
                if priority and priority.lower() not in ("none", "normal", "medium"):
                    priority_str = f" [{priority}]"

                # Completed marker
                check = "[x]" if completed else "[ ]"

                if multi_provider:
                    provider_name = task.get("_account_name", task.get("_provider", ""))
                    line = f"{i}. {check} [{provider_name}] {title}"
                else:
                    line = f"{i}. {check} {title}"

                if due_str:
                    line += f" - due {due_str}"
                if priority_str:
                    line += priority_str

                response_parts.append(line)

        # Handle failed accounts
        if failed_accounts:
            for failed in failed_accounts:
                account_name = failed.get("account_name", "")
                email = failed.get("email", "")
                reason = failed.get("reason", "unknown")

                account_display = email if email else account_name if account_name else "todo"

                if reason == "token_expired":
                    response_parts.append(
                        f"\nI lost access to your {account_display} account. "
                        f"Could you reconnect it in settings?"
                    )
                elif reason == "unsupported_provider":
                    response_parts.append(
                        f"\nSorry, I can't access {account_display} yet - that provider isn't supported."
                    )
                else:
                    response_parts.append(
                        f"\nI had trouble checking {account_display}. Want me to try again later?"
                    )

        return "\n".join(response_parts)

    @staticmethod
    def _format_due_date(due_str: str) -> str:
        """Format due date string to short display format."""
        if not due_str:
            return ""
        try:
            from dateutil import parser as date_parser
            dt = date_parser.parse(due_str)
            now = datetime.now()
            if dt.year == now.year:
                return dt.strftime("%b %d").lstrip("0")
            else:
                return dt.strftime("%b %d, %Y").lstrip("0")
        except Exception:
            return due_str
