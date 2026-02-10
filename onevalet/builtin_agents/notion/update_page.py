"""
NotionUpdatePageAgent - Update existing Notion pages
"""
import os
import logging
import json
from typing import Dict, Any

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult
from .client import NotionClient

logger = logging.getLogger(__name__)


@valet()
class NotionUpdatePageAgent(StandardAgent):
    """Update an existing Notion page. Use when the user wants to edit or modify content in Notion."""

    page_title = InputField(
        prompt="Which note do you want to update?",
        description="Title or name of the Notion page to update",
    )
    content = InputField(
        prompt="What content do you want to add or change?",
        description="New content to append to the page",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self._client = NotionClient()
        self._resolved_page_id = None
        self._resolved_page_title = None

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        page = self._resolved_page_title or self.collected_fields.get("page_title", "")
        content = self.collected_fields.get("content", "")
        preview = content[:100] + "..." if len(content) > 100 else content
        return (
            f"Update this Notion page?\n\n"
            f"Page: {page}\n"
            f"Add content: {preview}\n\n"
            f"(yes / no / or describe changes)"
        )

    def parse_approval(self, user_input: str) -> ApprovalResult:
        text = user_input.strip().lower()
        if text in ("yes", "y", "ok", "sure", "go", "confirm", "update"):
            return ApprovalResult.APPROVED
        if text in ("no", "n", "cancel", "stop", "nevermind", "never mind"):
            return ApprovalResult.REJECTED
        return ApprovalResult.MODIFY

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Override to resolve page before approval."""
        result = await super().on_initializing(msg)

        page_title = self.collected_fields.get("page_title", "")
        if page_title and result.status in (AgentStatus.WAITING_FOR_APPROVAL, AgentStatus.RUNNING):
            await self._resolve_page(page_title)
            if not self._resolved_page_id:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't find a Notion page matching \"{page_title}\". Try a different name?"
                )

        return result

    async def _resolve_page(self, title: str) -> None:
        """Search for page by title."""
        if not os.getenv("NOTION_API_KEY"):
            return

        try:
            data = await self._client.search(query=title, filter_type="page", page_size=3)
            results = data.get("results", [])
            if results:
                self._resolved_page_id = results[0]["id"]
                self._resolved_page_title = NotionClient.get_page_title(results[0])
        except Exception as e:
            logger.warning(f"Failed to resolve page: {e}")

    async def on_running(self, msg: Message) -> AgentResult:
        """Append content to the Notion page."""
        content = self.collected_fields.get("content", "")

        if not os.getenv("NOTION_API_KEY"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Notion API key not configured. Please add it in Settings."
            )

        if not self._resolved_page_id:
            page_title = self.collected_fields.get("page_title", "")
            await self._resolve_page(page_title)
            if not self._resolved_page_id:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't find a Notion page matching \"{page_title}\"."
                )

        try:
            blocks = NotionClient.text_to_blocks(content)
            await self._client.append_blocks(self._resolved_page_id, blocks)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Updated \"{self._resolved_page_title}\" with new content."
            )

        except Exception as e:
            logger.error(f"Failed to update Notion page: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't update the Notion page. Please check your API key and permissions."
            )
