"""
Dropbox Provider - Dropbox API v2 implementation for cloud storage

Uses Dropbox HTTP API v2 (POST-based) for file operations.
Docs: https://www.dropbox.com/developers/documentation/http/documentation
"""

import os
import logging
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseCloudStorageProvider

logger = logging.getLogger(__name__)

API_BASE = "https://api.dropboxapi.com/2"
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"


def _file_type_from_name(name: str) -> str:
    """Derive a human-readable file type from a filename extension."""
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    type_map = {
        "pdf": "pdf",
        "doc": "document",
        "docx": "document",
        "xls": "spreadsheet",
        "xlsx": "spreadsheet",
        "ppt": "presentation",
        "pptx": "presentation",
        "txt": "text",
        "md": "text",
        "csv": "spreadsheet",
        "jpg": "image",
        "jpeg": "image",
        "png": "image",
        "gif": "image",
        "svg": "image",
        "mp4": "video",
        "mov": "video",
        "mp3": "audio",
        "wav": "audio",
        "zip": "archive",
        "gz": "archive",
        "tar": "archive",
        "py": "code",
        "js": "code",
        "ts": "code",
        "html": "code",
        "css": "code",
        "json": "data",
        "xml": "data",
    }
    return type_map.get(ext, ext if ext else "file")


def _normalize_file(entry: dict) -> dict:
    """Convert a Dropbox file metadata entry to the unified format."""
    name = entry.get("name", "")
    return {
        "id": entry.get("path_display") or entry.get("id", ""),
        "name": name,
        "type": _file_type_from_name(name),
        "modified": entry.get("server_modified", ""),
        "size": entry.get("size", 0),
        "path": entry.get("path_display", ""),
        "url": "",
    }


class DropboxProvider(BaseCloudStorageProvider):
    """Dropbox cloud storage provider using API v2."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)

    async def _post(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        body: Optional[dict] = None,
    ) -> httpx.Response:
        """Send a POST request to the Dropbox API."""
        headers = {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }
        return await client.post(
            f"{API_BASE}{endpoint}",
            headers=headers,
            json=body if body is not None else {},
            timeout=30.0,
        )

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        body: Optional[dict] = None,
    ) -> httpx.Response:
        """POST with automatic 401 retry after token refresh."""
        response = await self._post(client, endpoint, body)

        if response.status_code == 401:
            logger.warning("Dropbox 401 - attempting token refresh")
            if await self.ensure_valid_token(force_refresh=True):
                response = await self._post(client, endpoint, body)

        return response

    # ===== Abstract method implementations =====

    async def search_files(
        self,
        query: str,
        file_type: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Search files by keyword using Dropbox search_v2."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                body: Dict[str, Any] = {
                    "query": query,
                    "options": {"max_results": max_results},
                }

                response = await self._post_with_retry(
                    client, "/files/search_v2", body
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Dropbox API error: {response.status_code} - {response.text}",
                    }

                data = response.json()
                matches = data.get("matches", [])

                files = []
                for match in matches:
                    metadata = match.get("metadata", {}).get("metadata", {})
                    tag = metadata.get(".tag", "")
                    if tag == "file":
                        files.append(_normalize_file(metadata))

                logger.info(
                    f"Dropbox search found {len(files)} files for query '{query}'"
                )
                return {"success": True, "data": files}

        except Exception as e:
            logger.error(f"Dropbox search_files error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_recent_files(
        self,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """List recently modified files from Dropbox root folder."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                body = {
                    "path": "",
                    "recursive": True,
                    "limit": min(max_results * 3, 2000),
                }

                response = await self._post_with_retry(
                    client, "/files/list_folder", body
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Dropbox API error: {response.status_code} - {response.text}",
                    }

                data = response.json()
                entries = data.get("entries", [])

                # Filter to files only and sort by server_modified descending
                files = [e for e in entries if e.get(".tag") == "file"]
                files.sort(
                    key=lambda f: f.get("server_modified", ""), reverse=True
                )
                files = files[:max_results]

                result = [_normalize_file(f) for f in files]
                logger.info(f"Dropbox listed {len(result)} recent files")
                return {"success": True, "data": result}

        except Exception as e:
            logger.error(f"Dropbox list_recent_files error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get detailed file metadata from Dropbox."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await self._post_with_retry(
                    client, "/files/get_metadata", {"path": file_id}
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Dropbox API error: {response.status_code} - {response.text}",
                    }

                metadata = response.json()
                file_data = _normalize_file(metadata)
                file_data["shared"] = bool(
                    metadata.get("sharing_info")
                    or metadata.get("has_explicit_shared_members")
                )

                logger.info(f"Dropbox get_file_info: {file_id}")
                return {"success": True, "data": file_data}

        except Exception as e:
            logger.error(f"Dropbox get_file_info error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_download_link(self, file_id: str) -> Dict[str, Any]:
        """Get a temporary download link for a file (valid ~4 hours)."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await self._post_with_retry(
                    client, "/files/get_temporary_link", {"path": file_id}
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Dropbox API error: {response.status_code} - {response.text}",
                    }

                data = response.json()
                link = data.get("link", "")
                expires = (
                    datetime.now(timezone.utc) + timedelta(hours=4)
                ).isoformat()

                logger.info(f"Dropbox get_download_link: {file_id}")
                return {
                    "success": True,
                    "data": {"url": link, "expires": expires},
                }

        except Exception as e:
            logger.error(f"Dropbox get_download_link error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def share_file(
        self,
        file_id: str,
        email: Optional[str] = None,
        link_type: str = "view",
    ) -> Dict[str, Any]:
        """Share a file via link or with a specific email address."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                if email:
                    # Share with a specific user by email
                    access_level = (
                        "editor" if link_type == "edit" else "viewer"
                    )
                    body = {
                        "file": file_id,
                        "members": [
                            {
                                ".tag": "email",
                                "email": email,
                            }
                        ],
                        "access_level": {".tag": access_level},
                    }
                    response = await self._post_with_retry(
                        client, "/sharing/add_file_member", body
                    )

                    if response.status_code != 200:
                        return {
                            "success": False,
                            "error": f"Dropbox API error: {response.status_code} - {response.text}",
                        }

                    logger.info(
                        f"Dropbox shared {file_id} with {email} ({access_level})"
                    )
                    return {
                        "success": True,
                        "data": {
                            "url": "",
                            "type": f"shared_with_{access_level}",
                        },
                    }
                else:
                    # Create a shared link
                    body = {
                        "path": file_id,
                        "settings": {"requested_visibility": "public"},
                    }
                    response = await self._post_with_retry(
                        client,
                        "/sharing/create_shared_link_with_settings",
                        body,
                    )

                    if response.status_code == 409:
                        # Link may already exist - Dropbox returns conflict
                        # Try to retrieve existing shared links
                        list_body = {"path": file_id}
                        list_response = await self._post_with_retry(
                            client,
                            "/sharing/list_shared_links",
                            list_body,
                        )
                        if list_response.status_code == 200:
                            links = list_response.json().get("links", [])
                            if links:
                                url = links[0].get("url", "")
                                logger.info(
                                    f"Dropbox retrieved existing shared link for {file_id}"
                                )
                                return {
                                    "success": True,
                                    "data": {"url": url, "type": "link"},
                                }

                        return {
                            "success": False,
                            "error": "Shared link already exists but could not be retrieved",
                        }

                    if response.status_code != 200:
                        return {
                            "success": False,
                            "error": f"Dropbox API error: {response.status_code} - {response.text}",
                        }

                    url = response.json().get("url", "")
                    logger.info(f"Dropbox created shared link for {file_id}")
                    return {
                        "success": True,
                        "data": {"url": url, "type": "link"},
                    }

        except Exception as e:
            logger.error(f"Dropbox share_file error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_storage_usage(self) -> Dict[str, Any]:
        """Get Dropbox storage usage information."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                # Dropbox /users/get_space_usage takes no body but expects
                # a null JSON body or empty string; send None to avoid issues.
                headers = {
                    **self._auth_headers(),
                    "Content-Type": "application/json",
                }
                response = await client.post(
                    f"{API_BASE}/users/get_space_usage",
                    headers=headers,
                    content="null",
                    timeout=30.0,
                )

                if response.status_code == 401:
                    logger.warning("Dropbox 401 on get_space_usage - refreshing token")
                    if await self.ensure_valid_token(force_refresh=True):
                        headers = {
                            **self._auth_headers(),
                            "Content-Type": "application/json",
                        }
                        response = await client.post(
                            f"{API_BASE}/users/get_space_usage",
                            headers=headers,
                            content="null",
                            timeout=30.0,
                        )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Dropbox API error: {response.status_code} - {response.text}",
                    }

                data = response.json()
                used = data.get("used", 0)
                allocation = data.get("allocation", {})
                total = allocation.get("allocated", 0)
                percent = round((used / total) * 100, 2) if total > 0 else 0.0

                logger.info(
                    f"Dropbox storage: {self.format_size(used)} / {self.format_size(total)} ({percent}%)"
                )
                return {
                    "success": True,
                    "data": {
                        "used": used,
                        "total": total,
                        "percent": percent,
                    },
                }

        except Exception as e:
            logger.error(f"Dropbox get_storage_usage error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def upload_file(
        self,
        file_name: str,
        file_data: bytes,
        mime_type: str = "image/jpeg",
        folder_path: str = "",
    ) -> Dict[str, Any]:
        """Upload a file to Dropbox. Not yet implemented."""
        return {"success": False, "error": "Dropbox upload not implemented yet"}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh the Dropbox OAuth2 access token using a refresh token."""
        try:
            app_key = os.environ.get("DROPBOX_APP_KEY", "")
            app_secret = os.environ.get("DROPBOX_APP_SECRET", "")

            if not self.refresh_token:
                return {"success": False, "error": "No refresh token available"}
            if not app_key or not app_secret:
                return {
                    "success": False,
                    "error": "DROPBOX_APP_KEY and DROPBOX_APP_SECRET must be set in environment",
                }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                        "client_id": app_key,
                        "client_secret": app_secret,
                    },
                    timeout=30.0,
                )

                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"Token refresh failed: {response.status_code} - {response.text}",
                    }

                token_data = response.json()
                access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 14400)
                token_expiry = datetime.now(timezone.utc) + timedelta(
                    seconds=expires_in
                )

                logger.info("Dropbox access token refreshed successfully")
                return {
                    "success": True,
                    "access_token": access_token,
                    "expires_in": expires_in,
                    "token_expiry": token_expiry,
                }

        except Exception as e:
            logger.error(f"Dropbox refresh_access_token error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
