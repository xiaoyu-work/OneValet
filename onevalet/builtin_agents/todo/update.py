"""
Update Todo Agent - Complete or update todo tasks

Multi-step agent for completing/updating tasks with search and approval.

State Flow:
1. INITIALIZING -> extract fields, search tasks
2. WAITING_FOR_APPROVAL -> show found tasks, wait for user selection
3. RUNNING -> execute completion/update
4. COMPLETED
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class UpdateTodoAgent(StandardAgent):
    """Update/complete todo task agent with search and approval"""

    search_query = InputField(
        prompt="Which task would you like to complete or update?",
        description="Task to complete or update",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.found_tasks = []
        self.error_message = None
        self._search_completed = False

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        """Generate confirmation prompt with list of tasks to complete"""
        if self.error_message:
            return self.error_message

        tasks = self.found_tasks

        if not tasks:
            return "I couldn't find any tasks matching your search. Want to try different keywords?"

        if len(tasks) == 1:
            task = tasks[0]
            title = task.get("title", "Untitled")
            due = task.get("due", "")
            due_str = f" (due {due})" if due else ""
            return f"Found this task:\n\"{title}\"{due_str}\n\nMark it as done? (yes/no)"

        response_parts = [f"Found {len(tasks)} tasks:\n"]

        for i, task in enumerate(tasks[:10], 1):
            title = task.get("title", "Untitled")
            due = task.get("due", "")
            if len(title) > 40:
                title = title[:37] + "..."
            due_str = f" - due {due}" if due else ""
            provider = task.get("_account_name", "")
            prefix = f"[{provider}] " if provider else ""
            response_parts.append(f"{i}. {prefix}{title}{due_str}\n")

        if len(tasks) > 10:
            response_parts.append(f"...and {len(tasks) - 10} more\n")

        response_parts.append(f"\nWhich ones to complete? (reply 1, 1-3, all, or no)")

        return "".join(response_parts)

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract field values from user input"""
        if not self.llm_client:
            return {"search_query": user_input}

        extraction_prompt = f"""Extract task completion/update criteria from the user's request:

User request: {user_input}

Return JSON:
{{
  "search_query": ""
}}

Rules:
- search_query: Keywords to identify which task(s) to complete or update.
  Extract the core task description from phrases like "I finished X", "done with X", "completed X"."""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract information. Return only JSON."},
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

            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def on_initializing(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        if not self._search_completed:
            await self._search_tasks()
            self._search_completed = True

        if not self.found_tasks and not self.collected_fields.get("search_query"):
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="Which task would you like to complete? Tell me the task name or keywords."
            )

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        await self._search_tasks()
        self._search_completed = True

        if not self.found_tasks:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="I couldn't find any tasks matching that. Try different keywords?"
            )

        return self.make_result(
            status=AgentStatus.WAITING_FOR_APPROVAL,
            raw_message=self.get_approval_prompt()
        )

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        user_input = msg.get_text() if msg else ""
        approval = await self._parse_approval_with_selection(user_input)

        if approval == "approved":
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == "rejected":
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Got it, no tasks updated."
            )

        else:
            self._search_completed = False
            self.found_tasks = []

            await self._extract_and_collect_fields(user_input)

            if self.collected_fields.get("search_query"):
                await self._search_tasks()
                self._search_completed = True

                if self.found_tasks:
                    return self.make_result(
                        status=AgentStatus.WAITING_FOR_APPROVAL,
                        raw_message=self.get_approval_prompt()
                    )

            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="Which task would you like to complete instead?"
            )

    async def _parse_approval_with_selection(self, user_response: str) -> str:
        """Parse user's approval response with task selection support"""
        if not self.llm_client:
            response_lower = user_response.lower().strip()
            if response_lower in ["yes", "y", "ok", "all", "go", "confirm", "do it"]:
                return "approved"
            elif response_lower in ["no", "n", "cancel", "stop", "nevermind"]:
                return "rejected"
            else:
                return "modify"

        task_count = len(self.found_tasks)
        prompt = f"""Parse the user's response to select which tasks to complete.

Total tasks: {task_count} (numbered 1 to {task_count})
User response: "{user_response}"

Determine intent:
- Complete ALL: {{"action": "all"}}
- Complete SPECIFIC: {{"action": "select", "indices": [1, 2, 3]}}
- CANCEL: {{"action": "cancel"}}
- CHANGE search: {{"action": "modify"}}

Return ONLY valid JSON:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                enable_thinking=False
            )
            parsed = json.loads(result.content.strip())
            action = parsed.get("action", "modify")

            if action == "all":
                return "approved"
            elif action == "select":
                indices = parsed.get("indices", [])
                selected_tasks = []
                for idx in indices:
                    zero_idx = idx - 1
                    if 0 <= zero_idx < len(self.found_tasks):
                        selected_tasks.append(self.found_tasks[zero_idx])

                if selected_tasks:
                    self.found_tasks = selected_tasks
                    return "approved"
                else:
                    return "modify"
            elif action == "cancel":
                return "rejected"
            else:
                return "modify"

        except Exception as e:
            logger.error(f"Failed to parse approval response: {e}")
            return "modify"

    async def _search_tasks(self) -> None:
        """Search for tasks matching the criteria across all providers"""
        from onevalet.providers.todo.resolver import TodoAccountResolver
        from onevalet.providers.todo.factory import TodoProviderFactory

        search_query = self.collected_fields.get("search_query")
        if not search_query:
            return

        try:
            accounts = await TodoAccountResolver.resolve_accounts(self.tenant_id, ["all"])

            if not accounts:
                self.error_message = "No todo accounts found. Please connect one first."
                return

            all_tasks = []

            for account in accounts:
                provider = TodoProviderFactory.create_provider(account)
                if not provider:
                    continue

                if not await provider.ensure_valid_token():
                    continue

                try:
                    result = await provider.search_tasks(query=search_query)

                    if result.get("success"):
                        tasks = result.get("data", [])
                        for task in tasks:
                            task["_provider"] = account.get("provider", "")
                            task["_account_name"] = account.get("account_name", "")
                            task["_account_email"] = account.get("email", "")
                        all_tasks.extend(tasks)
                except Exception as e:
                    logger.error(f"Failed to search {account.get('account_name')}: {e}", exc_info=True)

            # If no results from search, try listing all and filtering with LLM
            if not all_tasks and self.llm_client:
                all_tasks = await self._fallback_search_with_llm(accounts, search_query)

            if all_tasks:
                self.found_tasks = all_tasks

        except Exception as e:
            logger.error(f"Failed to search tasks: {e}", exc_info=True)

    async def _fallback_search_with_llm(self, accounts: List[Dict], search_query: str) -> List[Dict]:
        """Fallback: list all tasks and filter with LLM"""
        from onevalet.providers.todo.factory import TodoProviderFactory

        all_tasks = []
        for account in accounts:
            provider = TodoProviderFactory.create_provider(account)
            if not provider:
                continue
            if not await provider.ensure_valid_token():
                continue

            try:
                result = await provider.list_tasks(max_results=50)
                if result.get("success"):
                    tasks = result.get("data", [])
                    for task in tasks:
                        task["_provider"] = account.get("provider", "")
                        task["_account_name"] = account.get("account_name", "")
                        task["_account_email"] = account.get("email", "")
                    all_tasks.extend(tasks)
            except Exception:
                continue

        if not all_tasks:
            return []

        return await self._filter_tasks_with_llm(all_tasks, search_query)

    async def _filter_tasks_with_llm(self, tasks: List[Dict], search_query: str) -> List[Dict]:
        """Use LLM to filter tasks by relevance"""
        if not tasks or not self.llm_client or not search_query:
            return tasks

        task_list = [
            {"index": i, "title": t.get("title", ""), "due": t.get("due", "")}
            for i, t in enumerate(tasks)
        ]

        prompt = f"""Find tasks matching: "{search_query}"

Tasks: {json.dumps(task_list)}

Return a JSON array of matching indices (0-based), like: [0, 3, 5]"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False
            )
            match = re.search(r'\[[\d,\s]*\]', result.content)
            if match:
                indices = json.loads(match.group())
                return [tasks[i] for i in indices if 0 <= i < len(tasks)]
        except Exception as e:
            logger.error(f"Filter LLM failed: {e}")

        return []

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute task completion"""
        from onevalet.providers.todo.factory import TodoProviderFactory

        if not self.found_tasks:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I couldn't find any tasks to complete."
            )

        try:
            completed_count = 0
            failed_count = 0

            # Group tasks by account for efficient provider creation
            tasks_by_account = {}
            for task in self.found_tasks:
                key = (task.get("_provider", ""), task.get("_account_email", ""))
                if key not in tasks_by_account:
                    tasks_by_account[key] = []
                tasks_by_account[key].append(task)

            from onevalet.providers.todo.resolver import TodoAccountResolver

            for (provider_name, email), tasks in tasks_by_account.items():
                # Resolve account for this group
                account = await TodoAccountResolver.resolve_account(self.tenant_id, email or "primary")
                if not account:
                    failed_count += len(tasks)
                    continue

                provider = TodoProviderFactory.create_provider(account)
                if not provider:
                    failed_count += len(tasks)
                    continue

                if not await provider.ensure_valid_token():
                    failed_count += len(tasks)
                    continue

                for task in tasks:
                    try:
                        result = await provider.complete_task(
                            task_id=task.get("id", ""),
                            list_id=task.get("list_id")
                        )
                        if result.get("success"):
                            completed_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to complete task: {e}")
                        failed_count += 1

            if completed_count > 0 and failed_count == 0:
                if completed_count == 1:
                    title = self.found_tasks[0].get("title", "task")
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"Done! Marked \"{title}\" as complete."
                    )
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Done! Completed {completed_count} task(s)."
                )
            elif completed_count > 0 and failed_count > 0:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Completed {completed_count} task(s), but {failed_count} failed."
                )
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I had trouble completing those tasks. Want me to try again?"
                )

        except Exception as e:
            logger.error(f"Failed to complete tasks: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong. Want me to try again?"
            )
