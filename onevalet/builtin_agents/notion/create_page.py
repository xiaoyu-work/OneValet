"""
NotionCreatePageAgent - Create new Notion pages
"""
import os
import logging
import json
from typing import Dict, Any

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult
from .client import NotionClient

logger = logging.getLogger(__name__)


@valet()
class NotionCreatePageAgent(StandardAgent):
    """Create new Notion pages with content"""

    title = InputField(
        prompt="What's the title for the note?",
        description="Page title",
    )
    content = InputField(
        prompt="What content should be in the note?",
        description="Page content (plain text)",
    )
    parent = InputField(
        prompt="Which page should this be under? (leave blank for default workspace)",
        description="Parent page name (optional, searches Notion to find it)",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self._client = NotionClient()
        self._resolved_parent_id = None
        self._resolved_parent_name = None

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        title = self.collected_fields.get("title", "")
        content = self.collected_fields.get("content", "")
        parent = self._resolved_parent_name or self.collected_fields.get("parent", "workspace")
        preview = content[:100] + "..." if len(content) > 100 else content
        return (
            f"Create this Notion page?\n\n"
            f"Title: {title}\n"
            f"Parent: {parent}\n"
            f"Content: {preview}\n\n"
            f"(yes / no / or describe changes)"
        )

    async def parse_approval_async(self, user_input: str):
        """Parse approval using LLM."""
        if not self.llm_client:
            return self.parse_approval(user_input)

        prompt = f"""The user was asked to approve creating a Notion page. Their response was:
"{user_input}"

Classify as one of:
- APPROVED: if they said yes, ok, sure, go ahead, confirm, etc.
- REJECTED: if they said no, cancel, stop, never mind, etc.
- MODIFY: if they want to change something (title, content, etc.)

Return JSON: {{"decision": "APPROVED|REJECTED|MODIFY"}}"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                enable_thinking=False
            )
            data = json.loads(result.content.strip())
            decision = data.get("decision", "MODIFY")
            if decision == "APPROVED":
                return ApprovalResult.APPROVED
            elif decision == "REJECTED":
                return ApprovalResult.REJECTED
            return ApprovalResult.MODIFY
        except Exception:
            return self.parse_approval(user_input)

    def parse_approval(self, user_input: str) -> ApprovalResult:
        text = user_input.strip().lower()
        if text in ("yes", "y", "ok", "sure", "go", "confirm", "send", "create"):
            return ApprovalResult.APPROVED
        if text in ("no", "n", "cancel", "stop", "nevermind", "never mind"):
            return ApprovalResult.REJECTED
        return ApprovalResult.MODIFY

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        user_input = msg.get_text() if msg else ""
        approval = await self.parse_approval_async(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)
        elif approval == ApprovalResult.REJECTED:
            return self.make_result(status=AgentStatus.CANCELLED)
        else:
            await self._extract_and_collect_fields(user_input)
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Override to resolve parent page before approval."""
        result = await super().on_initializing(msg)

        # Resolve parent if specified
        parent_name = self.collected_fields.get("parent", "")
        if parent_name and result.status == AgentStatus.WAITING_FOR_APPROVAL:
            await self._resolve_parent(parent_name)

        return result

    async def _resolve_parent(self, parent_name: str) -> None:
        """Search for parent page by name."""
        if not parent_name or not os.getenv("NOTION_API_KEY"):
            return

        try:
            data = await self._client.search(query=parent_name, filter_type="page", page_size=1)
            results = data.get("results", [])
            if results:
                self._resolved_parent_id = results[0]["id"]
                self._resolved_parent_name = NotionClient.get_page_title(results[0])
        except Exception as e:
            logger.warning(f"Failed to resolve parent page: {e}")

    async def on_running(self, msg: Message) -> AgentResult:
        """Create the Notion page."""
        title = self.collected_fields.get("title", "")
        content = self.collected_fields.get("content", "")

        if not os.getenv("NOTION_API_KEY"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Notion API key not configured. Please add it in Settings."
            )

        try:
            # Resolve parent if not yet done
            parent_name = self.collected_fields.get("parent", "")
            if parent_name and not self._resolved_parent_id:
                await self._resolve_parent(parent_name)

            if self._resolved_parent_id:
                page = await self._client.create_page(
                    parent_id=self._resolved_parent_id,
                    title=title,
                    content=content,
                    parent_type="page_id",
                )
            else:
                # Use search to find a workspace-level page to use as parent
                # Notion API requires a parent; use the first page found
                data = await self._client.search(filter_type="page", page_size=1)
                results = data.get("results", [])
                if not results:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message="No pages found in your Notion workspace to use as parent. Please specify a parent page."
                    )
                parent_id = results[0]["id"]
                page = await self._client.create_page(
                    parent_id=parent_id,
                    title=title,
                    content=content,
                    parent_type="page_id",
                )

            url = page.get("url", "")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Created Notion page \"{title}\".\n{url}"
            )

        except Exception as e:
            logger.error(f"Failed to create Notion page: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't create the Notion page. Please check your API key and permissions."
            )
