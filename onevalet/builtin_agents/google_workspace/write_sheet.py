"""
GoogleSheetsWriteAgent - Write data to Google Sheets with approval
"""
import logging
import json
from typing import Dict, Any, List, Optional

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult
from .client import GoogleWorkspaceClient

logger = logging.getLogger(__name__)


@valet()
class GoogleSheetsWriteAgent(StandardAgent):
    """Write or update data in a Google Sheets spreadsheet. Use when the user wants to add or modify spreadsheet data."""

    spreadsheet_name = InputField(
        prompt="Which spreadsheet should I write to?",
        description="Name of the Google Sheets spreadsheet",
    )
    range = InputField(
        prompt="What range should I write to? (e.g. Sheet1!A1)",
        description="Cell range in A1 notation (e.g. Sheet1!A1:C10)",
    )
    values = InputField(
        prompt="What data should I write? (JSON array of arrays, e.g. [[\"Name\",\"Age\"],[\"Alice\",30]])",
        description="Data as JSON array of arrays, e.g. [[\"Name\",\"Age\"],[\"Alice\",30]]",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self._resolved_spreadsheet_id = None
        self._resolved_spreadsheet_name = None

    def needs_approval(self) -> bool:
        return True

    def get_approval_prompt(self) -> str:
        name = self._resolved_spreadsheet_name or self.collected_fields.get("spreadsheet_name", "")
        range_ = self.collected_fields.get("range", "")
        values_str = self.collected_fields.get("values", "[]")
        preview = values_str[:300] + "..." if len(values_str) > 300 else values_str
        return (
            f"Write data to this Google Sheet?\n\n"
            f"Spreadsheet: {name}\n"
            f"Range: {range_}\n"
            f"Data:\n{preview}\n\n"
            f"(yes / no / or describe changes)"
        )

    async def parse_approval_async(self, user_input: str):
        """Parse approval using LLM."""
        if not self.llm_client:
            return self.parse_approval(user_input)

        prompt = f"""The user was asked to approve writing data to a Google Sheet. Their response was:
"{user_input}"

Classify as one of:
- APPROVED: if they said yes, ok, sure, go ahead, confirm, etc.
- REJECTED: if they said no, cancel, stop, never mind, etc.
- MODIFY: if they want to change something (data, range, etc.)

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
        if text in ("yes", "y", "ok", "sure", "go", "confirm", "write"):
            return ApprovalResult.APPROVED
        if text in ("no", "n", "cancel", "stop", "nevermind", "never mind"):
            return ApprovalResult.REJECTED
        return ApprovalResult.MODIFY

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Override to resolve spreadsheet ID before approval."""
        result = await super().on_initializing(msg)

        spreadsheet_name = self.collected_fields.get("spreadsheet_name", "")
        if spreadsheet_name and result.status in (AgentStatus.WAITING_FOR_APPROVAL, AgentStatus.RUNNING):
            await self._resolve_spreadsheet(spreadsheet_name)
            if not self._resolved_spreadsheet_id:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't find a spreadsheet matching \"{spreadsheet_name}\". Please check the name and try again."
                )

        return result

    async def _resolve_spreadsheet(self, name: str) -> None:
        """Search Drive for a spreadsheet by name."""
        token, error = await self._get_token()
        if error:
            logger.warning(f"Cannot resolve spreadsheet: {error}")
            return

        try:
            client = GoogleWorkspaceClient(token)
            files = await client.drive_search(query=name, file_type="spreadsheet", page_size=3)
            if files:
                self._resolved_spreadsheet_id = files[0]["id"]
                self._resolved_spreadsheet_name = files[0].get("name", name)
        except Exception as e:
            logger.warning(f"Failed to resolve spreadsheet: {e}")

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        user_input = msg.get_text() if msg else ""
        approval = await self.parse_approval_async(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)
        elif approval == ApprovalResult.REJECTED:
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="Sheet write cancelled."
            )
        else:
            await self._extract_and_collect_fields(user_input)
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt()
            )

    async def on_running(self, msg: Message) -> AgentResult:
        range_ = self.collected_fields.get("range", "")
        values_str = self.collected_fields.get("values", "[]")

        # Parse values JSON
        try:
            values = json.loads(values_str)
            if not isinstance(values, list) or not all(isinstance(row, list) for row in values):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Invalid data format. Values must be a JSON array of arrays, e.g. [[\"Name\",\"Age\"],[\"Alice\",30]]"
                )
        except json.JSONDecodeError as e:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Invalid JSON in values: {e}"
            )

        token, error = await self._get_token()
        if error:
            return self.make_result(status=AgentStatus.COMPLETED, raw_message=error)

        # Resolve spreadsheet if not yet done
        if not self._resolved_spreadsheet_id:
            spreadsheet_name = self.collected_fields.get("spreadsheet_name", "")
            await self._resolve_spreadsheet(spreadsheet_name)
            if not self._resolved_spreadsheet_id:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't find a spreadsheet matching \"{spreadsheet_name}\"."
                )

        try:
            client = GoogleWorkspaceClient(token)
            result = await client.sheets_update_values(
                spreadsheet_id=self._resolved_spreadsheet_id,
                range_=range_,
                values=values,
            )

            updated_range = result.get("updatedRange", range_)
            updated_cells = result.get("updatedCells", len(values))
            sheet_url = f"https://docs.google.com/spreadsheets/d/{self._resolved_spreadsheet_id}/edit"
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=(
                    f"Updated \"{self._resolved_spreadsheet_name}\" range {updated_range} "
                    f"({updated_cells} cells).\nURL: {sheet_url}"
                )
            )
        except Exception as e:
            logger.error(f"Failed to write to Google Sheet: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Failed to write to Google Sheet: {e}"
            )

    async def _get_token(self):
        """Get Google access token from credential store."""
        from .auth import get_google_token_for_agent
        return await get_google_token_for_agent(self.tenant_id)
