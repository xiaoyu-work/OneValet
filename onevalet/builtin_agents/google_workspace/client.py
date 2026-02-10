"""
Google Workspace API Client - Drive, Docs, Sheets.

Shared by ReAct tools and agent-tools.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DRIVE_API = "https://www.googleapis.com/drive/v3"
DOCS_API = "https://docs.googleapis.com/v1"
SHEETS_API = "https://sheets.googleapis.com/v4"


class GoogleWorkspaceClient:
    """Async Google Workspace REST API client."""

    def __init__(self, access_token: str):
        self.access_token = access_token

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    # ── Drive ──

    async def drive_search(
        self, query: str = "", file_type: Optional[str] = None, page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Search files in Google Drive.

        file_type can be: "document", "spreadsheet", "folder", "pdf", or None for all.
        """
        q_parts = []
        if query:
            q_parts.append(f"name contains '{query}'")
        if file_type:
            mime_map = {
                "document": "application/vnd.google-apps.document",
                "spreadsheet": "application/vnd.google-apps.spreadsheet",
                "folder": "application/vnd.google-apps.folder",
                "pdf": "application/pdf",
            }
            mime = mime_map.get(file_type)
            if mime:
                q_parts.append(f"mimeType = '{mime}'")

        q_parts.append("trashed = false")
        q_str = " and ".join(q_parts)

        params = {
            "q": q_str,
            "pageSize": min(page_size, 100),
            "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
            "orderBy": "modifiedTime desc",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{DRIVE_API}/files",
                headers=self._headers,
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json().get("files", [])

    async def drive_get_file(self, file_id: str) -> Dict[str, Any]:
        """Get file metadata."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{DRIVE_API}/files/{file_id}",
                headers=self._headers,
                params={"fields": "id,name,mimeType,modifiedTime,webViewLink"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Docs ──

    async def docs_get(self, document_id: str) -> Dict[str, Any]:
        """Get full document content."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{DOCS_API}/documents/{document_id}",
                headers=self._headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def docs_create(self, title: str, body_text: str = "") -> Dict[str, Any]:
        """Create a new Google Doc with optional body text."""
        # Step 1: Create empty doc
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{DOCS_API}/documents",
                headers=self._headers,
                json={"title": title},
                timeout=15.0,
            )
            resp.raise_for_status()
            doc = resp.json()

        # Step 2: Insert text if provided
        if body_text:
            await self.docs_append(doc["documentId"], body_text)

        return doc

    async def docs_append(self, document_id: str, text: str) -> Dict[str, Any]:
        """Append text to a Google Doc."""
        requests = [
            {
                "insertText": {
                    "location": {"index": 1},
                    "text": text,
                }
            }
        ]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{DOCS_API}/documents/{document_id}:batchUpdate",
                headers=self._headers,
                json={"requests": requests},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Sheets ──

    async def sheets_get_metadata(self, spreadsheet_id: str) -> Dict[str, Any]:
        """Get spreadsheet metadata (sheet names, etc.)."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SHEETS_API}/spreadsheets/{spreadsheet_id}",
                headers=self._headers,
                params={"fields": "spreadsheetId,properties.title,sheets.properties"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def sheets_get_values(
        self, spreadsheet_id: str, range_: str
    ) -> Dict[str, Any]:
        """Get cell values from a spreadsheet range."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SHEETS_API}/spreadsheets/{spreadsheet_id}/values/{range_}",
                headers=self._headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def sheets_update_values(
        self, spreadsheet_id: str, range_: str, values: List[List[Any]]
    ) -> Dict[str, Any]:
        """Write values to a spreadsheet range."""
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{SHEETS_API}/spreadsheets/{spreadsheet_id}/values/{range_}",
                headers=self._headers,
                params={"valueInputOption": "USER_ENTERED"},
                json={"values": values},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Helpers ──

    @staticmethod
    def docs_to_text(doc: Dict[str, Any]) -> str:
        """Extract plain text from a Google Docs API response."""
        body = doc.get("body", {})
        content = body.get("content", [])
        parts = []
        for element in content:
            paragraph = element.get("paragraph")
            if not paragraph:
                continue
            for pe in paragraph.get("elements", []):
                text_run = pe.get("textRun")
                if text_run:
                    parts.append(text_run.get("content", ""))
        return "".join(parts).strip()

    @staticmethod
    def format_mime_type(mime: str) -> str:
        """Human-readable file type."""
        mime_labels = {
            "application/vnd.google-apps.document": "Google Doc",
            "application/vnd.google-apps.spreadsheet": "Google Sheet",
            "application/vnd.google-apps.folder": "Folder",
            "application/vnd.google-apps.presentation": "Google Slides",
            "application/pdf": "PDF",
        }
        return mime_labels.get(mime, mime.split("/")[-1] if "/" in mime else mime)
