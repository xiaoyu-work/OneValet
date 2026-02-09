"""
Base Calendar Provider - Abstract interface for all calendar providers

Each calendar provider (Google Calendar, Outlook Calendar, etc.) must implement this interface.
Providers receive a credentials dict directly - they do not query any database.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class BaseCalendarProvider(ABC):
    """
    Abstract base class for calendar providers.

    All calendar providers must implement:
    - list_events()
    - create_event()
    - update_event()
    - delete_event()
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
                - provider: str (google, microsoft, etc.)
                - account_name: str
                - email: str (email address or calendar ID)
                - access_token: str
                - refresh_token: str
                - token_expiry: str or datetime
                - scopes: List[str]
                - provider_data: dict
            on_token_refreshed: Optional callback invoked after successful token refresh.
        """
        self.credentials = credentials
        self.provider = credentials["provider"]
        self.account_name = credentials.get("account_name", "primary")
        self.calendar_id = credentials.get("email", "primary")
        self.access_token = credentials["access_token"]
        self.refresh_token = credentials.get("refresh_token")
        self.token_expiry = credentials.get("token_expiry")
        self.scopes = credentials.get("scopes", [])
        self.provider_data = credentials.get("provider_data", {})
        self._on_token_refreshed = on_token_refreshed

    # ===== Abstract methods =====

    @abstractmethod
    async def list_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10,
        query: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List calendar events."""
        pass

    @abstractmethod
    async def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event."""
        pass

    @abstractmethod
    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing calendar event."""
        pass

    @abstractmethod
    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a calendar event."""
        pass

    @abstractmethod
    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh expired OAuth access token."""
        pass

    # ===== Common helper methods =====

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        """Check if access token is valid, refresh if needed."""
        if force_refresh:
            logger.info(f"Forcing token refresh for {self.account_name}...")
            return await self._do_refresh()

        if not self.token_expiry:
            logger.debug(f"No token_expiry info for {self.account_name}, assuming token is valid")
            return True

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
        return f"<{self.__class__.__name__} account={self.account_name} calendar_id={self.calendar_id}>"
