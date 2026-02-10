"""
Notion API Client - Shared REST API wrapper for Notion agents.

All Notion agents use this client for API calls.
Requires NOTION_API_KEY environment variable (Internal Integration Token).
"""

import os
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

NOTION_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"


class NotionClient:
    """Async Notion REST API client."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("NOTION_API_KEY", "")

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    # ── Search ──

    async def search(
        self,
        query: str = "",
        filter_type: Optional[str] = None,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Search pages and databases.

        Args:
            query: Search query text.
            filter_type: "page" or "database" to filter results.
            page_size: Max results per page.
        """
        body: Dict[str, Any] = {"page_size": page_size}
        if query:
            body["query"] = query
        if filter_type in ("page", "database"):
            body["filter"] = {"value": filter_type, "property": "object"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/search",
                headers=self._headers,
                json=body,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Pages ──

    async def get_page(self, page_id: str) -> Dict[str, Any]:
        """Get page metadata."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/pages/{page_id}",
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def create_page(
        self,
        parent_id: str,
        title: str,
        content: str = "",
        parent_type: str = "page_id",
    ) -> Dict[str, Any]:
        """
        Create a new page.

        Args:
            parent_id: Parent page or database ID.
            title: Page title.
            content: Plain text content (converted to blocks).
            parent_type: "page_id" or "database_id".
        """
        body: Dict[str, Any] = {
            "parent": {parent_type: parent_id},
        }

        if parent_type == "database_id":
            body["properties"] = {
                "title": {"title": [{"text": {"content": title}}]},
            }
        else:
            body["properties"] = {
                "title": {"title": [{"text": {"content": title}}]},
            }

        if content:
            body["children"] = self.text_to_blocks(content)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/pages",
                headers=self._headers,
                json=body,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def update_page(
        self, page_id: str, properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update page properties."""
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{BASE_URL}/pages/{page_id}",
                headers=self._headers,
                json={"properties": properties},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Blocks ──

    async def get_page_content(self, block_id: str) -> List[Dict[str, Any]]:
        """Get all child blocks of a page/block (handles pagination)."""
        blocks: List[Dict[str, Any]] = []
        start_cursor: Optional[str] = None

        async with httpx.AsyncClient() as client:
            while True:
                params: Dict[str, Any] = {"page_size": 100}
                if start_cursor:
                    params["start_cursor"] = start_cursor

                resp = await client.get(
                    f"{BASE_URL}/blocks/{block_id}/children",
                    headers=self._headers,
                    params=params,
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()

                blocks.extend(data.get("results", []))

                if not data.get("has_more"):
                    break
                start_cursor = data.get("next_cursor")

        return blocks

    async def append_blocks(
        self, block_id: str, children: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Append child blocks to a page/block."""
        async with httpx.AsyncClient() as client:
            resp = await client.patch(
                f"{BASE_URL}/blocks/{block_id}/children",
                headers=self._headers,
                json={"children": children},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Databases ──

    async def query_database(
        self,
        database_id: str,
        filter: Optional[Dict[str, Any]] = None,
        sorts: Optional[List[Dict[str, Any]]] = None,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """Query a Notion database."""
        body: Dict[str, Any] = {"page_size": page_size}
        if filter:
            body["filter"] = filter
        if sorts:
            body["sorts"] = sorts

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/databases/{database_id}/query",
                headers=self._headers,
                json=body,
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Helpers ──

    @staticmethod
    def text_to_blocks(text: str) -> List[Dict[str, Any]]:
        """Convert plain text to Notion paragraph blocks (split by blank lines)."""
        paragraphs = text.split("\n\n") if "\n\n" in text else [text]
        blocks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": para}}]
                    },
                }
            )
        return blocks or [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": ""}}]},
            }
        ]

    @staticmethod
    def blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion blocks."""
        parts = []
        for block in blocks:
            block_type = block.get("type", "")
            type_data = block.get(block_type, {})

            rich_text = type_data.get("rich_text", [])
            if rich_text:
                text = "".join(rt.get("plain_text", "") for rt in rich_text)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def get_page_title(page: Dict[str, Any]) -> str:
        """Extract title from a page object."""
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_parts)
        return "Untitled"
