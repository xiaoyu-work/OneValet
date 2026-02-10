"""Google OAuth 2.0 Authorization Code Flow."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
]


class GoogleOAuth:
    """Google OAuth 2.0 Authorization Code Flow."""

    AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

    @staticmethod
    def get_credentials() -> tuple[str, str]:
        """Read client_id and client_secret from env vars."""
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "Google OAuth not configured. "
                "Set Google OAuth App credentials in Settings > OAuth Apps."
            )
        return client_id, client_secret

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str) -> str:
        """Build Google authorization URL."""
        client_id, _ = GoogleOAuth.get_credentials()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{GoogleOAuth.AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        client_id, client_secret = GoogleOAuth.get_credentials()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                GoogleOAuth.TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
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
        """Fetch user email from Google userinfo endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                GoogleOAuth.USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("email", "")
