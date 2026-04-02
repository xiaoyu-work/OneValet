"""
GitHub Copilot OAuth Authentication

Implements the two-tier token system for GitHub Copilot LLM access:
1. GitHub OAuth token (long-lived) — obtained via Device Flow or provided directly
2. Copilot API token (short-lived ~30min) — exchanged from GitHub token

Usage:
    # One-time device flow (CLI)
    github_token = await device_flow_authenticate()

    # Token management
    manager = CopilotTokenManager(github_token="gho_xxx")
    copilot_token, base_url = await manager.get_token()
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# GitHub OAuth App — same client ID used by VS Code / Copilot CLI
GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_SCOPES = ["read:user"]

# Endpoints
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_COPILOT_BASE_URL = "https://api.individual.githubcopilot.com"

# Required headers for Copilot API calls
COPILOT_EXTRA_HEADERS = {
    "Editor-Version": "vscode/1.96.2",
    "Copilot-Integration-Id": "vscode-chat",
}

# Token refresh safety margin (5 minutes before expiry)
TOKEN_REFRESH_MARGIN_SECONDS = 5 * 60


@dataclass
class DeviceCodeResponse:
    """Response from GitHub device code request."""
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass
class CopilotToken:
    """A short-lived Copilot API token."""
    token: str
    expires_at: float  # Unix timestamp in seconds
    base_url: str

    @property
    def is_expired(self) -> bool:
        return time.time() >= (self.expires_at - TOKEN_REFRESH_MARGIN_SECONDS)


def _derive_base_url(copilot_token: str) -> str:
    """Extract the API base URL from the Copilot token's embedded metadata.

    The token may contain a proxy endpoint hint like:
        ;proxy-ep=proxy.individual.githubcopilot.com;
    Convert proxy.* to api.* for the actual API endpoint.
    """
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", copilot_token, re.IGNORECASE)
    if not match or not match.group(1):
        return DEFAULT_COPILOT_BASE_URL
    proxy_host = match.group(1).strip()
    proxy_host = re.sub(r"^https?://", "", proxy_host)
    api_host = re.sub(r"^proxy\.", "api.", proxy_host, flags=re.IGNORECASE)
    return f"https://{api_host}"


class CopilotTokenManager:
    """Manages the two-tier Copilot token lifecycle.

    Holds a long-lived GitHub OAuth token and exchanges it for short-lived
    Copilot API tokens on demand, with automatic caching and refresh.

    Args:
        github_token: GitHub OAuth access token (from device flow).
        github_refresh_token: Optional refresh token for renewing the GitHub token.
    """

    def __init__(
        self,
        github_token: str,
        github_refresh_token: Optional[str] = None,
    ):
        self._github_token = github_token
        self._github_refresh_token = github_refresh_token
        self._cached_token: Optional[CopilotToken] = None
        self._lock = asyncio.Lock()

    @property
    def github_token(self) -> str:
        return self._github_token

    async def get_token(self) -> Tuple[str, str]:
        """Get a valid Copilot API token, refreshing if needed.

        Returns:
            Tuple of (copilot_token, base_url).

        Raises:
            RuntimeError: If token exchange fails.
        """
        async with self._lock:
            if self._cached_token and not self._cached_token.is_expired:
                return self._cached_token.token, self._cached_token.base_url

            token = await self._exchange_token()
            self._cached_token = token
            return token.token, token.base_url

    async def _exchange_token(self, _retried: bool = False) -> CopilotToken:
        """Exchange GitHub token for a Copilot API token."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                COPILOT_TOKEN_URL,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._github_token}",
                },
            )

        if response.status_code == 401:
            # GitHub token may have expired — try refresh (once)
            if not _retried and self._github_refresh_token:
                logger.info("GitHub token expired, attempting refresh...")
                await self._refresh_github_token()
                return await self._exchange_token(_retried=True)
            raise RuntimeError(
                "GitHub token expired and no refresh token available. "
                "Please re-authenticate with: python -m onevalet copilot-auth"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Copilot token exchange failed ({response.status_code}): "
                f"{response.text}"
            )

        data = response.json()
        copilot_token = data["token"]
        expires_at = data.get("expires_at", 0)
        base_url = _derive_base_url(copilot_token)

        # Validate expires_at; fall back to 30 minutes from now if missing
        if not expires_at or expires_at < time.time():
            expires_at = time.time() + 30 * 60
            logger.warning(
                "Copilot token missing or invalid expires_at, "
                "assuming 30-minute validity"
            )

        logger.info(
            f"Copilot token obtained, expires_at={expires_at}, base_url={base_url}"
        )

        return CopilotToken(
            token=copilot_token,
            expires_at=float(expires_at),
            base_url=base_url,
        )

    async def _refresh_github_token(self) -> None:
        """Refresh the GitHub OAuth token using the refresh token."""
        if not self._github_refresh_token:
            raise RuntimeError("No refresh token available")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GITHUB_TOKEN_URL,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": self._github_refresh_token,
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"GitHub token refresh failed ({response.status_code}): "
                f"{response.text}"
            )

        data = response.json()
        if "error" in data:
            raise RuntimeError(
                f"GitHub token refresh error: {data.get('error_description', data['error'])}"
            )

        self._github_token = data["access_token"]
        if data.get("refresh_token"):
            self._github_refresh_token = data["refresh_token"]

        logger.info("GitHub token refreshed successfully")


# ---------------------------------------------------------------------------
# Device Flow — one-time interactive CLI authentication
# ---------------------------------------------------------------------------


async def request_device_code() -> DeviceCodeResponse:
    """Request a device code from GitHub for the OAuth device flow.

    Returns:
        DeviceCodeResponse with the user code and verification URL.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            GITHUB_DEVICE_CODE_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "client_id": GITHUB_CLIENT_ID,
                "scope": " ".join(GITHUB_SCOPES),
            },
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Device code request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    return DeviceCodeResponse(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data.get("verification_uri") or data.get("verification_url"),
        expires_in=data["expires_in"],
        interval=data.get("interval", 5),
    )


async def poll_for_token(
    device_code: str,
    interval: int,
) -> dict:
    """Poll GitHub for the OAuth token after user authorization.

    Args:
        device_code: The device code from the device code request.
        interval: Polling interval in seconds.

    Returns:
        Dict with access_token, refresh_token (optional), expires_in (optional).

    Raises:
        RuntimeError: If the device code expires or the user denies access.
    """
    poll_interval = interval

    while True:
        await asyncio.sleep(poll_interval)

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GITHUB_TOKEN_URL,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )

        data = response.json()

        if "access_token" in data:
            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "expires_in": data.get("expires_in"),
            }

        error = data.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            poll_interval += 5
            continue
        if error == "expired_token":
            raise RuntimeError("Device code expired. Please try again.")
        if error == "access_denied":
            raise RuntimeError("Authorization denied by user.")

        raise RuntimeError(
            f"Token polling failed: {data.get('error_description', error or str(data))}"
        )


async def device_flow_authenticate() -> dict:
    """Run the full GitHub device flow interactively.

    Prints instructions to stdout for the user to follow.

    Returns:
        Dict with access_token and optional refresh_token.
    """
    print("\n" + "=" * 60)
    print("  GitHub Copilot Authentication (Device Flow)")
    print("=" * 60)

    device = await request_device_code()

    print(f"\n  1. Open: {device.verification_uri}")
    print(f"  2. Enter code: {device.user_code}")
    print(f"\n  Waiting for authorization (expires in {device.expires_in}s)...")

    token_data = await poll_for_token(device.device_code, device.interval)

    print("\n  ✓ Authorization successful!")
    print("=" * 60)

    return token_data
