"""Microsoft OAuth 2.0 Authorization Code Flow."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

MICROSOFT_SCOPES = [
    "offline_access",
    "https://graph.microsoft.com/Mail.ReadWrite",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/Calendars.ReadWrite",
    "https://graph.microsoft.com/Tasks.ReadWrite",
    "https://graph.microsoft.com/User.Read",
]


class MicrosoftOAuth:
    """Microsoft OAuth 2.0 Authorization Code Flow."""

    AUTHORIZE_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
    TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    GRAPH_ME_URL = "https://graph.microsoft.com/v1.0/me"

    @staticmethod
    def get_credentials() -> tuple[str, str, str]:
        """Read client_id, client_secret, tenant_id from env vars."""
        client_id = os.getenv("MICROSOFT_CLIENT_ID")
        client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
        tenant_id = os.getenv("MICROSOFT_TENANT_ID", "common")
        if not client_id or not client_secret:
            raise ValueError(
                "Microsoft OAuth not configured. "
                "Set Microsoft OAuth App credentials in Settings > OAuth Apps."
            )
        return client_id, client_secret, tenant_id

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str) -> str:
        """Build Microsoft authorization URL."""
        client_id, _, tenant_id = MicrosoftOAuth.get_credentials()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(MICROSOFT_SCOPES),
            "response_mode": "query",
            "state": state,
        }
        base_url = MicrosoftOAuth.AUTHORIZE_URL.format(tenant=tenant_id)
        return f"{base_url}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        client_id, client_secret, tenant_id = MicrosoftOAuth.get_credentials()
        token_url = MicrosoftOAuth.TOKEN_URL.format(tenant=tenant_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                    "scope": " ".join(MICROSOFT_SCOPES),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        expires_in = data.get("expires_in", 3600)
        token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "token_expiry": token_expiry.isoformat(),
            "scope": data.get("scope", ""),
        }

    @staticmethod
    async def fetch_user_email(access_token: str) -> str:
        """Fetch user email from Microsoft Graph /me endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                MicrosoftOAuth.GRAPH_ME_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("mail") or data.get("userPrincipalName", "")
