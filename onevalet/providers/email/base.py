"""
Base Email Provider - Abstract interface for all email providers

Each email provider (Gmail, Outlook, etc.) must implement this interface.
Providers receive a credentials dict directly - they do not query any database.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class BaseEmailProvider(ABC):
    """
    Abstract base class for email providers.

    All email providers must implement:
    - send_email()
    - search_emails()
    - delete_emails()
    - refresh_access_token()
    """

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize provider with credentials dict.

        Args:
            credentials: Dict containing:
                - provider: str (gmail, outlook, etc.)
                - account_name: str (user-defined name like "work", "personal")
                - email: str (email address)
                - access_token: str
                - refresh_token: str
                - token_expiry: str or datetime
                - scopes: List[str]
                - provider_data: dict (provider-specific metadata)
            on_token_refreshed: Optional callback invoked with the full new
                credentials dict after a successful token refresh. The caller
                can use this to persist via credential_store.save().
        """
        self.credentials = credentials
        self.provider = credentials["provider"]
        self.account_name = credentials.get("account_name", "primary")
        self.email = credentials["email"]
        self.access_token = credentials["access_token"]
        self.refresh_token = credentials.get("refresh_token")
        self.token_expiry = credentials.get("token_expiry")
        self.scopes = credentials.get("scopes", [])
        self.provider_data = credentials.get("provider_data", {})
        self._on_token_refreshed = on_token_refreshed

    # ===== Abstract methods (must be implemented by subclasses) =====

    @abstractmethod
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Send an email."""
        pass

    @abstractmethod
    async def search_emails(
        self,
        query: Optional[str] = None,
        sender: Optional[str] = None,
        date_range: Optional[str] = None,
        unread_only: bool = False,
        max_results: int = 20,
        days_back: Optional[int] = None,
        include_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search emails with filters."""
        pass

    @abstractmethod
    async def delete_emails(
        self,
        message_ids: List[str],
        permanent: bool = False,
    ) -> Dict[str, Any]:
        """Delete or trash emails."""
        pass

    @abstractmethod
    async def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh expired OAuth access token.

        Returns:
            {
                "success": bool,
                "access_token": str (new token),
                "expires_in": int (seconds),
                "token_expiry": datetime (calculated expiry time),
                "error": str (if failed)
            }
        """
        pass

    # ===== Common helper methods =====

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        """
        Check if access token is valid, refresh if needed.

        After a successful refresh the provider updates its own fields and
        invokes the on_token_refreshed callback so the caller can persist.
        """
        if force_refresh:
            logger.info(f"Forcing token refresh for {self.account_name}...")
            return await self._do_refresh()

        if not self.token_expiry:
            logger.debug(f"No token_expiry info for {self.account_name}, assuming token is valid")
            return True

        # Parse token_expiry if it's a string
        if isinstance(self.token_expiry, str):
            from dateutil import parser
            self.token_expiry = parser.parse(self.token_expiry)

        now = datetime.now(timezone.utc)
        if self.token_expiry <= now + timedelta(minutes=5):
            logger.info(f"Token expiring soon for {self.account_name}, refreshing...")
            return await self._do_refresh()

        return True

    async def _do_refresh(self) -> bool:
        """Perform token refresh and notify callback."""
        result = await self.refresh_access_token()
        if result.get("success"):
            self.access_token = result["access_token"]
            self.token_expiry = result.get("token_expiry")

            # Update internal credentials dict
            self.credentials["access_token"] = self.access_token
            if self.token_expiry:
                self.credentials["token_expiry"] = (
                    self.token_expiry.isoformat()
                    if isinstance(self.token_expiry, datetime)
                    else self.token_expiry
                )

            if self._on_token_refreshed:
                self._on_token_refreshed(self.credentials)

            logger.info(f"Token refreshed successfully for {self.account_name}")
            return True
        else:
            logger.error(f"Token refresh failed for {self.account_name}: {result.get('error')}")
            return False

    def __repr__(self):
        return f"<{self.__class__.__name__} account={self.account_name} email={self.email}>"
