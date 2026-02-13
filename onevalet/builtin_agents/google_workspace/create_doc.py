"""
GoogleDocsCreateAgent - Create new Google Docs with approval
"""
import logging
import json
from typing import Dict, Any

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult
from .client import GoogleWorkspaceClient

logger = logging.getLogger(__name__)


@valet(expose_as_tool=False)
class GoogleDocsCreateAgent(StandardAgent):
    """Create a new Google Doc. Use when the user wants to write or create a document in Google Docs."""

    title = InputField(
        prompt="What's the title for the document?",
        description="Document title",
    )
    content = InputField(
        prompt="What content should the document have?",
        description="Document content (plain text)",
    )

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        title = self.collected_fields.get("title", "Untitled")
        content = self.collected_fields.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        return (
            f"Create this Google Doc?\n\n"
            f"Title: {title}\n"
            f"Content preview:\n{preview}\n\n"
            f"(yes / no / or describe changes)"
        )

    async def parse_approval_async(self, user_input: str):
        """Parse approval using LLM."""
        if not self.llm_client:
            return self.parse_approval(user_input)

        prompt = f"""The user was asked to approve creating a Google Doc. Their response was:
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
        if text in ("yes", "y", "ok", "sure", "go", "confirm", "create"):
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
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Document creation cancelled."
            )
        else:
            await self._extract_and_collect_fields(user_input)
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_running(self, msg: Message) -> AgentResult:
        title = self.collected_fields.get("title", "Untitled")
        content = self.collected_fields.get("content", "")

        token, error = await self._get_token()
        if error:
            return self.make_result(status=AgentStatus.COMPLETED, raw_message=error)

        try:
            client = GoogleWorkspaceClient(token)
            doc = await client.docs_create(title=title, body_text=content)

            doc_id = doc.get("documentId", "")
            doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Created Google Doc \"{title}\".\nURL: {doc_url}"
            )
        except Exception as e:
            logger.error(f"Failed to create Google Doc: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Failed to create Google Doc: {e}"
            )

    async def _get_token(self):
        """Get Google access token from credential store."""
        from .auth import get_google_token_for_agent
        return await get_google_token_for_agent(self.tenant_id)
