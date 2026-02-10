"""
OneDrive Provider - Microsoft Graph API implementation for cloud storage

Uses Microsoft Graph API for OneDrive file operations.
Requires OAuth scope: https://graph.microsoft.com/Files.ReadWrite.All
Shares OAuth tokens with outlook/outlook_calendar (service: "onedrive").
"""

import logging
import os
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseCloudStorageProvider

logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0/me/drive"


class OneDriveProvider(BaseCloudStorageProvider):
    """OneDrive provider implementation using Microsoft Graph API."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)

    def _normalize_file(self, item: dict) -> dict:
        """Convert a Graph API drive item to the unified file format."""
        parent_ref = item.get("parentReference", {})
        parent_path = parent_ref.get("path", "")
        # parentReference.path looks like /drive/root:/Documents
        # Strip the /drive/root: prefix for a cleaner display path
        if ":" in parent_path:
            parent_path = parent_path.split(":", 1)[1] or "/"

        file_type = "folder" if "folder" in item else "file"
        if file_type == "file" and item.get("file", {}).get("mimeType"):
            file_type = item["file"]["mimeType"]

        return {
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "type": file_type,
            "modified": item.get("lastModifiedDateTime", ""),
            "size": item.get("size", 0),
            "path": parent_path,
            "url": item.get("webUrl", ""),
        }

    async def _request(
        self,
        method: str,
        url: str,
        client: httpx.AsyncClient,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> httpx.Response:
        """Make an API request with automatic 401 retry."""
        kwargs: Dict[str, Any] = {
            "headers": self._auth_headers(),
            "timeout": 30.0,
        }
        if params:
            kwargs["params"] = params
        if json_body is not None:
            kwargs["headers"]["Content-Type"] = "application/json"
            kwargs["json"] = json_body

        response = await client.request(method, url, **kwargs)

        if response.status_code == 401:
            logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
            if await self.ensure_valid_token(force_refresh=True):
                kwargs["headers"] = self._auth_headers()
                if json_body is not None:
                    kwargs["headers"]["Content-Type"] = "application/json"
                response = await client.request(method, url, **kwargs)

        return response

    async def search_files(
        self,
        query: str,
        file_type: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Search files in OneDrive by keyword."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            params: Dict[str, Any] = {"$top": max_results}

            async with httpx.AsyncClient() as client:
                response = await self._request(
                    "GET",
                    f"{GRAPH_BASE_URL}/root/search(q='{query}')",
                    client,
                    params=params,
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                data = response.json()
                items = data.get("value", [])

                files = [self._normalize_file(item) for item in items]

                # Client-side filter by file type if specified
                if file_type:
                    ft_lower = file_type.lower()
                    files = [
                        f for f in files
                        if ft_lower in f["type"].lower() or ft_lower in f["name"].lower()
                    ]

                logger.info(f"OneDrive search found {len(files)} files for '{query}'")
                return {"success": True, "data": files}

        except Exception as e:
            logger.error(f"OneDrive search error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_recent_files(
        self,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """List recently modified files in OneDrive."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            params: Dict[str, Any] = {"$top": max_results}

            async with httpx.AsyncClient() as client:
                response = await self._request(
                    "GET",
                    f"{GRAPH_BASE_URL}/recent",
                    client,
                    params=params,
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                data = response.json()
                items = data.get("value", [])

                files = [self._normalize_file(item) for item in items]

                logger.info(f"OneDrive listed {len(files)} recent files")
                return {"success": True, "data": files}

        except Exception as e:
            logger.error(f"OneDrive list recent error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get detailed file metadata from OneDrive."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await self._request(
                    "GET",
                    f"{GRAPH_BASE_URL}/items/{file_id}",
                    client,
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                item = response.json()
                file_data = self._normalize_file(item)
                file_data["shared"] = "shared" in item

                logger.info(f"OneDrive got file info: {file_id}")
                return {"success": True, "data": file_data}

        except Exception as e:
            logger.error(f"OneDrive get file info error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_download_link(self, file_id: str) -> Dict[str, Any]:
        """Get a temporary download link for a OneDrive file."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await self._request(
                    "GET",
                    f"{GRAPH_BASE_URL}/items/{file_id}",
                    client,
                    params={"select": "@microsoft.graph.downloadUrl,name"},
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                item = response.json()
                download_url = item.get("@microsoft.graph.downloadUrl", "")

                if not download_url:
                    return {"success": False, "error": "No download URL available for this item"}

                logger.info(f"OneDrive got download link for: {file_id}")
                return {
                    "success": True,
                    "data": {
                        "url": download_url,
                        "expires": "Temporary pre-authenticated URL",
                    },
                }

        except Exception as e:
            logger.error(f"OneDrive get download link error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def share_file(
        self,
        file_id: str,
        email: Optional[str] = None,
        link_type: str = "view",
    ) -> Dict[str, Any]:
        """Share a OneDrive file via link or email invitation."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                if email:
                    # Share with a specific user via invitation
                    roles = ["write"] if link_type == "edit" else ["read"]
                    body = {
                        "recipients": [{"email": email}],
                        "roles": roles,
                        "requireSignIn": True,
                        "sendInvitation": True,
                        "message": "Shared via OneValet",
                    }

                    response = await self._request(
                        "POST",
                        f"{GRAPH_BASE_URL}/items/{file_id}/invite",
                        client,
                        json_body=body,
                    )

                    if response.status_code != 200:
                        return {"success": False, "error": f"Graph API error: {response.status_code}"}

                    data = response.json()
                    permissions = data.get("value", [])
                    link_url = ""
                    if permissions:
                        link_url = permissions[0].get("link", {}).get("webUrl", "")

                    logger.info(f"OneDrive shared file {file_id} with {email}")
                    return {
                        "success": True,
                        "data": {"url": link_url, "type": link_type, "shared_with": email},
                    }
                else:
                    # Create an anonymous sharing link
                    body = {
                        "type": link_type,
                        "scope": "anonymous",
                    }

                    response = await self._request(
                        "POST",
                        f"{GRAPH_BASE_URL}/items/{file_id}/createLink",
                        client,
                        json_body=body,
                    )

                    if response.status_code not in (200, 201):
                        return {"success": False, "error": f"Graph API error: {response.status_code}"}

                    data = response.json()
                    link_url = data.get("link", {}).get("webUrl", "")

                    logger.info(f"OneDrive created {link_type} link for file {file_id}")
                    return {
                        "success": True,
                        "data": {"url": link_url, "type": link_type},
                    }

        except Exception as e:
            logger.error(f"OneDrive share file error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_storage_usage(self) -> Dict[str, Any]:
        """Get OneDrive storage usage information."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await self._request(
                    "GET",
                    GRAPH_BASE_URL,
                    client,
                    params={"$select": "quota"},
                )

                if response.status_code != 200:
                    return {"success": False, "error": f"Graph API error: {response.status_code}"}

                data = response.json()
                quota = data.get("quota", {})
                used = quota.get("used", 0)
                total = quota.get("total", 0)
                percent = (used / total * 100) if total > 0 else 0.0

                logger.info(f"OneDrive storage: {self.format_size(used)} / {self.format_size(total)}")
                return {
                    "success": True,
                    "data": {
                        "used": used,
                        "total": total,
                        "percent": round(percent, 2),
                    },
                }

        except Exception as e:
            logger.error(f"OneDrive get storage usage error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Microsoft OAuth token for OneDrive."""
        try:
            client_id = os.getenv("MICROSOFT_CLIENT_ID")
            client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
            tenant = os.getenv("MICROSOFT_TENANT_ID", "common")

            if not client_id or not client_secret:
                return {"success": False, "error": "Microsoft OAuth credentials not configured"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": self.refresh_token,
                        "grant_type": "refresh_token",
                        "scope": "https://graph.microsoft.com/Files.ReadWrite.All offline_access",
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    expires_in = data.get("expires_in", 3600)
                    token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                    logger.info(f"OneDrive token refreshed for {self.account_name}")
                    return {
                        "success": True,
                        "access_token": data["access_token"],
                        "expires_in": expires_in,
                        "token_expiry": token_expiry,
                    }
                else:
                    logger.error(f"OneDrive token refresh failed: {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"OneDrive token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
