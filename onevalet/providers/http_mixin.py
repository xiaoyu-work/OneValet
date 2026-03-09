"""Shared HTTP utilities for OAuth-based providers.

Eliminates duplicated token refresh retry logic and error handling
across email, calendar, todo, and cloud storage providers.
"""

import logging
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
