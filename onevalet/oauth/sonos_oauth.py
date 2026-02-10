"""Sonos OAuth 2.0 Authorization Code Flow."""

import base64
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


class SonosOAuth:
    """Sonos OAuth 2.0 Authorization Code Flow."""

    AUTHORIZE_URL = "https://api.sonos.com/login/v3/oauth"
    TOKEN_URL = "https://api.sonos.com/login/v3/oauth/access"

    @staticmethod
    def get_credentials() -> tuple[str, str]:
        """Read client_id and client_secret from env vars."""
        client_id = os.getenv("SONOS_CLIENT_ID")
        client_secret = os.getenv("SONOS_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "Sonos OAuth not configured. "
                "Set Sonos OAuth App credentials in Settings > Smart Home."
            )
        return client_id, client_secret

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str) -> str:
        """Build Sonos authorization URL."""
        client_id, _ = SonosOAuth.get_credentials()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "playback-control-all",
            "state": state,
        }
        return f"{SonosOAuth.AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        client_id, client_secret = SonosOAuth.get_credentials()
        basic_auth = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                SonosOAuth.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Authorization": f"Basic {basic_auth}"},
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
        }
