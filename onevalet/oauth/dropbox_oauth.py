"""Dropbox OAuth 2.0 Authorization Code Flow."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


class DropboxOAuth:
    """Dropbox OAuth 2.0 Authorization Code Flow."""

    AUTHORIZE_URL = "https://www.dropbox.com/oauth2/authorize"
    TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"

    @staticmethod
    def get_credentials() -> tuple[str, str]:
        """Read app_key and app_secret from env vars."""
        app_key = os.getenv("DROPBOX_APP_KEY")
        app_secret = os.getenv("DROPBOX_APP_SECRET")
        if not app_key or not app_secret:
            raise ValueError(
                "Dropbox OAuth not configured. "
                "Set Dropbox OAuth App credentials in Settings > Cloud Storage."
            )
        return app_key, app_secret

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str) -> str:
        """Build Dropbox authorization URL."""
        app_key, _ = DropboxOAuth.get_credentials()
        params = {
            "client_id": app_key,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "token_access_type": "offline",
            "state": state,
        }
        return f"{DropboxOAuth.AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        app_key, app_secret = DropboxOAuth.get_credentials()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                DropboxOAuth.TOKEN_URL,
                data={
                    "code": code,
                    "grant_type": "authorization_code",
                    "client_id": app_key,
                    "client_secret": app_secret,
                    "redirect_uri": redirect_uri,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        expires_in = data.get("expires_in", 14400)
        token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "token_expiry": token_expiry.isoformat(),
        }

    @staticmethod
    async def fetch_user_email(access_token: str) -> str:
        """Fetch user email from Dropbox get_current_account endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.dropboxapi.com/2/users/get_current_account",
                headers={"Authorization": f"Bearer {access_token}"},
                content=b"null",
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("email", "")
