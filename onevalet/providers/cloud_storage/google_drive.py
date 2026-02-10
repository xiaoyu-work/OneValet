"""
Google Drive Provider - Google Drive API v3 implementation

Uses Google Drive API for cloud storage operations.
Shares OAuth tokens with Gmail / Google Calendar (service name: google_drive).
Requires OAuth scope: https://www.googleapis.com/auth/drive
"""

import logging
import os
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseCloudStorageProvider

logger = logging.getLogger(__name__)

DRIVE_API = "https://www.googleapis.com/drive/v3"

# Map Google MIME types to human-readable labels
MIME_TYPE_LABELS = {
    "application/vnd.google-apps.document": "Google Doc",
    "application/vnd.google-apps.spreadsheet": "Google Sheet",
    "application/vnd.google-apps.presentation": "Google Slides",
    "application/vnd.google-apps.folder": "Folder",
    "application/vnd.google-apps.form": "Google Form",
    "application/vnd.google-apps.drawing": "Google Drawing",
    "application/vnd.google-apps.site": "Google Site",
    "application/vnd.google-apps.shortcut": "Shortcut",
    "application/pdf": "PDF",
    "image/jpeg": "JPEG Image",
    "image/png": "PNG Image",
    "image/gif": "GIF Image",
    "video/mp4": "MP4 Video",
    "audio/mpeg": "MP3 Audio",
    "application/zip": "ZIP Archive",
    "text/plain": "Text File",
    "text/csv": "CSV File",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word Document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel Spreadsheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint Presentation",
}

# MIME type filter mapping for file_type parameter
FILE_TYPE_MIME_MAP = {
    "document": "application/vnd.google-apps.document",
    "spreadsheet": "application/vnd.google-apps.spreadsheet",
    "presentation": "application/vnd.google-apps.presentation",
    "folder": "application/vnd.google-apps.folder",
    "pdf": "application/pdf",
    "image": "image/",
    "video": "video/",
}

# Standard fields requested for file listings
FILE_FIELDS = "id,name,mimeType,modifiedTime,size,webViewLink,parents"
FILES_LIST_FIELDS = f"files({FILE_FIELDS})"


class GoogleDriveProvider(BaseCloudStorageProvider):
    """Google Drive provider implementation using Drive API v3."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)

    @staticmethod
    def _format_mime_type(mime: str) -> str:
        """Convert MIME type to a human-readable label."""
        if mime in MIME_TYPE_LABELS:
            return MIME_TYPE_LABELS[mime]
        # Fallback: use the subtype portion
        if "/" in mime:
            return mime.split("/")[-1].upper()
        return mime

    def _normalize_file(self, f: dict) -> dict:
        """Normalize a Drive API file resource to the unified format."""
        size_raw = f.get("size")
        size = int(size_raw) if size_raw else None

        return {
            "id": f.get("id", ""),
            "name": f.get("name", ""),
            "type": self._format_mime_type(f.get("mimeType", "")),
            "modified": f.get("modifiedTime", ""),
            "size": size,
            "path": "",  # Drive doesn't expose a flat path; parents are available
            "url": f.get("webViewLink", ""),
        }

    async def search_files(
        self,
        query: str,
        file_type: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Search files in Google Drive by keyword."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            q_parts = [f"name contains '{query}'", "trashed = false"]

            if file_type:
                mime = FILE_TYPE_MIME_MAP.get(file_type.lower())
                if mime:
                    if mime.endswith("/"):
                        q_parts.append(f"mimeType contains '{mime}'")
                    else:
                        q_parts.append(f"mimeType = '{mime}'")

            q_str = " and ".join(q_parts)

            params = {
                "q": q_str,
                "pageSize": min(max_results, 100),
                "fields": FILES_LIST_FIELDS,
                "orderBy": "modifiedTime desc",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DRIVE_API}/files",
                    headers=self._auth_headers(),
                    params=params,
                    timeout=15.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{DRIVE_API}/files",
                            headers=self._auth_headers(),
                            params=params,
                            timeout=15.0,
                        )

                if response.status_code != 200:
                    logger.error(f"Drive search failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Drive API error: {response.status_code}"}

                files = response.json().get("files", [])
                results = [self._normalize_file(f) for f in files]

                logger.info(f"Drive search found {len(results)} files for '{query}'")
                return {"success": True, "data": results}

        except Exception as e:
            logger.error(f"Drive search error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_recent_files(
        self,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """List recently modified files in Google Drive."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            params = {
                "q": "trashed = false",
                "pageSize": min(max_results, 100),
                "fields": FILES_LIST_FIELDS,
                "orderBy": "modifiedTime desc",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DRIVE_API}/files",
                    headers=self._auth_headers(),
                    params=params,
                    timeout=15.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{DRIVE_API}/files",
                            headers=self._auth_headers(),
                            params=params,
                            timeout=15.0,
                        )

                if response.status_code != 200:
                    logger.error(f"Drive list recent failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Drive API error: {response.status_code}"}

                files = response.json().get("files", [])
                results = [self._normalize_file(f) for f in files]

                logger.info(f"Drive listed {len(results)} recent files")
                return {"success": True, "data": results}

        except Exception as e:
            logger.error(f"Drive list recent error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get detailed metadata for a single file."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            fields = f"{FILE_FIELDS},shared,owners,sharingUser,capabilities"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=self._auth_headers(),
                    params={"fields": fields},
                    timeout=10.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{DRIVE_API}/files/{file_id}",
                            headers=self._auth_headers(),
                            params={"fields": fields},
                            timeout=10.0,
                        )

                if response.status_code != 200:
                    logger.error(f"Drive get file failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Drive API error: {response.status_code}"}

                f = response.json()
                data = self._normalize_file(f)
                data["shared"] = f.get("shared", False)

                logger.info(f"Drive got file info: {file_id}")
                return {"success": True, "data": data}

        except Exception as e:
            logger.error(f"Drive get file error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_download_link(self, file_id: str) -> Dict[str, Any]:
        """Get a download link for a file.

        For native Google types (Docs, Sheets, etc.) there is no direct
        download link -- the webViewLink is returned instead.
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=self._auth_headers(),
                    params={"fields": "webContentLink,webViewLink,mimeType"},
                    timeout=10.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{DRIVE_API}/files/{file_id}",
                            headers=self._auth_headers(),
                            params={"fields": "webContentLink,webViewLink,mimeType"},
                            timeout=10.0,
                        )

                if response.status_code != 200:
                    logger.error(f"Drive download link failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Drive API error: {response.status_code}"}

                data = response.json()
                url = data.get("webContentLink") or data.get("webViewLink", "")

                logger.info(f"Drive got download link for: {file_id}")
                return {
                    "success": True,
                    "data": {"url": url, "expires": ""},
                }

        except Exception as e:
            logger.error(f"Drive download link error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def share_file(
        self,
        file_id: str,
        email: Optional[str] = None,
        link_type: str = "view",
    ) -> Dict[str, Any]:
        """Share a file with a user or create a shareable link.

        Args:
            file_id: Drive file ID.
            email: If provided, share directly with this email address.
            link_type: "view" (reader) or "edit" (writer).

        Returns:
            {"success": bool, "data": {"url": str, "type": str}, "error": str}
        """
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            role = "writer" if link_type == "edit" else "reader"

            if email:
                permission = {
                    "type": "user",
                    "role": role,
                    "emailAddress": email,
                }
            else:
                permission = {
                    "type": "anyone",
                    "role": role,
                }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{DRIVE_API}/files/{file_id}/permissions",
                    headers={
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    },
                    json=permission,
                    timeout=15.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.post(
                            f"{DRIVE_API}/files/{file_id}/permissions",
                            headers={
                                **self._auth_headers(),
                                "Content-Type": "application/json",
                            },
                            json=permission,
                            timeout=15.0,
                        )

                if response.status_code not in (200, 201):
                    logger.error(f"Drive share failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Drive API error: {response.status_code}"}

                # Get the shareable link
                file_resp = await client.get(
                    f"{DRIVE_API}/files/{file_id}",
                    headers=self._auth_headers(),
                    params={"fields": "webViewLink"},
                    timeout=10.0,
                )

                url = ""
                if file_resp.status_code == 200:
                    url = file_resp.json().get("webViewLink", "")

                share_type = f"{'email' if email else 'link'} ({link_type})"
                logger.info(f"Drive shared file {file_id} via {share_type}")
                return {
                    "success": True,
                    "data": {"url": url, "type": share_type},
                }

        except Exception as e:
            logger.error(f"Drive share error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_storage_usage(self) -> Dict[str, Any]:
        """Get Google Drive storage usage information."""
        try:
            if not await self.ensure_valid_token():
                return {"success": False, "error": "Failed to refresh access token"}

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{DRIVE_API}/about",
                    headers=self._auth_headers(),
                    params={"fields": "storageQuota"},
                    timeout=10.0,
                )

                if response.status_code == 401:
                    logger.warning(f"401 Unauthorized - refreshing token for {self.account_name}")
                    if await self.ensure_valid_token(force_refresh=True):
                        response = await client.get(
                            f"{DRIVE_API}/about",
                            headers=self._auth_headers(),
                            params={"fields": "storageQuota"},
                            timeout=10.0,
                        )

                if response.status_code != 200:
                    logger.error(f"Drive storage usage failed: {response.status_code} - {response.text}")
                    return {"success": False, "error": f"Drive API error: {response.status_code}"}

                quota = response.json().get("storageQuota", {})
                used = int(quota.get("usage", 0))
                total = int(quota.get("limit", 0))
                percent = (used / total * 100) if total > 0 else 0.0

                logger.info(f"Drive storage: {self.format_size(used)} / {self.format_size(total)}")
                return {
                    "success": True,
                    "data": {
                        "used": used,
                        "total": total,
                        "percent": round(percent, 2),
                    },
                }

        except Exception as e:
            logger.error(f"Drive storage usage error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Google OAuth access token (shared with Gmail / Calendar)."""
        try:
            client_id = os.getenv("GOOGLE_CLIENT_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

            if not client_id or not client_secret:
                return {"success": False, "error": "Google OAuth credentials not configured"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "refresh_token": self.refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    expires_in = data.get("expires_in", 3600)
                    token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                    logger.info(f"Google Drive token refreshed for {self.account_name}")
                    return {
                        "success": True,
                        "access_token": data["access_token"],
                        "expires_in": expires_in,
                        "token_expiry": token_expiry,
                    }
                else:
                    logger.error(f"Google Drive token refresh failed: {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"Google Drive token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
