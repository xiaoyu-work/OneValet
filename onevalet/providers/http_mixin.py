"""Shared HTTP utilities for OAuth-based providers.

Eliminates duplicated token refresh retry logic and error handling
across email, calendar, todo, and cloud storage providers.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class OAuthHTTPMixin:
    """Mixin providing HTTP request methods with automatic token refresh.

    Subclasses must have:
        - ensure_valid_token(force_refresh: bool) -> bool
        - self._get_headers() -> dict  (returns auth headers)

    These are already provided by all Base*Provider classes (ensure_valid_token)
    and can be implemented trivially in the concrete provider (_get_headers).

    Usage example::

        class MyProvider(BaseCalendarProvider, OAuthHTTPMixin):
            def _get_headers(self) -> dict:
                return {"Authorization": f"Bearer {self.access_token}"}

            async def list_events(self, ...):
                response = await self._oauth_request(
                    "GET", f"{self.api_base_url}/events",
                    params={"maxResults": 10},
                )
                ...
    """

    async def _oauth_request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json: Any = None,
        data: Any = None,
        content: Any = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> httpx.Response:
        """Make an HTTP request with automatic 401 token refresh retry.

        Calls the API once. On 401, refreshes the token via
        ``ensure_valid_token(force_refresh=True)`` and retries once.

        Returns the raw ``httpx.Response`` -- callers decide how to handle
        status codes (raise_for_status, check manually, etc.).
        """
        req_headers = self._get_headers()
        if headers:
            req_headers.update(headers)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method, url, headers=req_headers,
                json=json, data=data, content=content, params=params,
            )

            if response.status_code == 401:
                logger.info(
                    f"[{self.__class__.__name__}] 401 received, refreshing token"
                )
                if await self.ensure_valid_token(force_refresh=True):
                    req_headers = self._get_headers()
                    if headers:
                        req_headers.update(headers)
                    response = await client.request(
                        method, url, headers=req_headers,
                        json=json, data=data, content=content, params=params,
                    )

            return response

    def _get_headers(self) -> Dict[str, str]:
        """Return authorization headers. Override in subclass."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_headers()"
        )

    async def refresh_access_token(self) -> Dict[str, Any]:
        """Refresh Google OAuth access token using refresh token."""
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
                    logger.info(f"{self.__class__.__name__} token refreshed for {self.account_name}")
                    return {
                        "success": True,
                        "access_token": data["access_token"],
                        "expires_in": expires_in,
                        "token_expiry": token_expiry,
                    }
                else:
                    logger.error(f"{self.__class__.__name__} token refresh failed: {response.text}")
                    return {"success": False, "error": f"Token refresh failed: {response.status_code}"}

        except Exception as e:
            logger.error(f"{self.__class__.__name__} token refresh error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
