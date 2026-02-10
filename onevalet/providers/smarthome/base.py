"""
Base Smart Home Provider - Common OAuth token management for smart home devices

Philips Hue and Sonos both use OAuth2, so this base class provides
shared token refresh logic (same pattern as BaseTodoProvider).
Each subclass defines its own device-specific methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class BaseSmartHomeProvider(ABC):
    """
    Abstract base class for smart home providers.

    Provides common OAuth2 token management. Subclasses implement
    device-specific control methods.
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
                - provider: str (philips_hue, sonos)
                - email: str (account email)
                - access_token: str
                - refresh_token: str
                - token_expiry: str or datetime
                - account_name: str
            on_token_refreshed: Callback after successful token refresh
        """
        self.credentials = credentials
        self.provider = credentials.get("provider", "")
        self.account_name = credentials.get("account_name", "primary")
        self.email = credentials.get("email", "")
        self.access_token = credentials.get("access_token", "")
        self.refresh_token = credentials.get("refresh_token", "")
        self.token_expiry = credentials.get("token_expiry")
        self._on_token_refreshed = on_token_refreshed

    @abstractmethod
    async def refresh_access_token(self) -> Dict[str, Any]:
        """
        Refresh expired OAuth access token.

        Returns:
            {
                "success": bool,
                "access_token": str,
                "token_expiry": datetime,
                "error": str (if failed)
            }
        """
        pass

    async def ensure_valid_token(self, force_refresh: bool = False) -> bool:
        """Check if access token is valid, refresh if needed."""
        if force_refresh:
            logger.info(f"Forcing token refresh for {self.provider} ({self.account_name})")
            return await self._do_refresh()

        if not self.token_expiry:
            logger.debug(f"No token_expiry for {self.provider}, assuming valid")
            return True

        if isinstance(self.token_expiry, str):
            from dateutil import parser
            self.token_expiry = parser.parse(self.token_expiry)

        now = datetime.now(timezone.utc)
        if self.token_expiry <= now + timedelta(minutes=5):
            logger.info(f"Token expiring soon for {self.provider}, refreshing...")
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

            logger.info(f"Token refreshed for {self.provider} ({self.account_name})")
            return True
        else:
            logger.error(f"Token refresh failed for {self.provider}: {result.get('error')}")
            return False

    def _auth_headers(self) -> dict:
        """Common authorization headers."""
        return {"Authorization": f"Bearer {self.access_token}"}

    def __repr__(self):
        return f"<{self.__class__.__name__} provider={self.provider} account={self.account_name}>"
