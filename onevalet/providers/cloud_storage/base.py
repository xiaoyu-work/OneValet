"""
Base Cloud Storage Provider - Abstract interface for cloud storage services

Each provider (Google Drive, OneDrive, Dropbox) must implement this interface.
Providers use OAuth2 tokens (same pattern as BaseTodoProvider).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class BaseCloudStorageProvider(ABC):
    """
    Abstract base class for cloud storage providers.

    All providers must implement:
    - search_files()
    - list_recent_files()
    - get_file_info()
    - get_download_link()
    - share_file()
    - get_storage_usage()
    - refresh_access_token()
    """

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        self.credentials = credentials
        self.provider = credentials.get("provider", "")
        self.account_name = credentials.get("account_name", "primary")
        self.email = credentials.get("email", "")
        self.access_token = credentials.get("access_token", "")
        self.refresh_token = credentials.get("refresh_token", "")
        self.token_expiry = credentials.get("token_expiry")
        self._on_token_refreshed = on_token_refreshed

    # ===== Abstract methods =====

    @abstractmethod
    async def search_files(
        self,
        query: str,
        file_type: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Search files by keyword.

        Returns:
            {"success": bool, "data": [
                {"id": str, "name": str, "type": str, "modified": str,
                 "size": int, "path": str, "url": str}
            ], "error": str}
        """
        pass

    @abstractmethod
    async def list_recent_files(
        self,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        List recently modified files.

        Returns same format as search_files().
        """
        pass

    @abstractmethod
    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        Get detailed file metadata.

        Returns:
            {"success": bool, "data": {
                "id": str, "name": str, "type": str, "modified": str,
                "size": int, "path": str, "url": str, "shared": bool
            }, "error": str}
        """
        pass

    @abstractmethod
    async def get_download_link(self, file_id: str) -> Dict[str, Any]:
        """
        Get a temporary download link for a file.

        Returns:
            {"success": bool, "data": {"url": str, "expires": str}, "error": str}
        """
        pass

    @abstractmethod
    async def share_file(
        self,
        file_id: str,
        email: Optional[str] = None,
        link_type: str = "view",
    ) -> Dict[str, Any]:
        """
        Share a file. If email is provided, share with that user.
        Otherwise create a shareable link.

        Args:
            file_id: File identifier
            email: Email to share with (None for link sharing)
            link_type: "view" or "edit"

        Returns:
            {"success": bool, "data": {"url": str, "type": str}, "error": str}
        """
        pass

    @abstractmethod
    async def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage information.

        Returns:
            {"success": bool, "data": {
                "used": int (bytes), "total": int (bytes), "percent": float
            }, "error": str}
        """
        pass

    @abstractmethod
    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh OAuth access token."""
        pass

    # ===== Common helpers =====

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        """Check if access token is valid, refresh if needed."""
        if force_refresh:
            return await self._do_refresh()

        if not self.token_expiry:
            return True

        if isinstance(self.token_expiry, str):
            from dateutil import parser
            self.token_expiry = parser.parse(self.token_expiry)

        now = datetime.now(timezone.utc)
        if self.token_expiry <= now + timedelta(minutes=5):
            logger.info(f"Token expiring for {self.provider}, refreshing...")
            return await self._do_refresh()

        return True

    async def _do_refresh(self) -> bool:
        """Perform token refresh and notify callback."""
        result = await self.refresh_access_token()
        if result.get("success"):
            self.access_token = result["access_token"]
            self.token_expiry = result.get("token_expiry")

            self.credentials["access_token"] = self.access_token
            if self.token_expiry:
                self.credentials["token_expiry"] = (
                    self.token_expiry.isoformat()
                    if isinstance(self.token_expiry, datetime)
                    else self.token_expiry
                )

            if self._on_token_refreshed:
                self._on_token_refreshed(self.credentials)

            logger.info(f"Token refreshed for {self.provider}")
            return True
        else:
            logger.error(f"Token refresh failed for {self.provider}: {result.get('error')}")
            return False

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format bytes to human-readable string."""
        if size_bytes is None:
            return ""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def get_provider_display_name(self) -> str:
        names = {"google": "Google Drive", "onedrive": "OneDrive", "dropbox": "Dropbox"}
        return names.get(self.provider, self.provider.title())

    def __repr__(self):
        return f"<{self.__class__.__name__} provider={self.provider} email={self.email}>"
