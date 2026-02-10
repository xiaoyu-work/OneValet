"""Philips Hue OAuth 2.0 Authorization Code Flow."""

import base64
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


class HueOAuth:
    """Philips Hue OAuth 2.0 Authorization Code Flow."""

    AUTHORIZE_URL = "https://api.meethue.com/v2/oauth2/authorize"
    TOKEN_URL = "https://api.meethue.com/v2/oauth2/token"

    @staticmethod
    def get_credentials() -> tuple[str, str, str]:
        """Read client_id, client_secret, and app_id from env vars."""
        client_id = os.getenv("HUE_CLIENT_ID")
        client_secret = os.getenv("HUE_CLIENT_SECRET")
        app_id = os.getenv("HUE_APP_ID")
        if not client_id or not client_secret or not app_id:
            raise ValueError(
                "Philips Hue OAuth not configured. "
                "Set Hue OAuth App credentials in Settings > Smart Home."
            )
        return client_id, client_secret, app_id

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str) -> str:
        """Build Hue authorization URL."""
        client_id, _, app_id = HueOAuth.get_credentials()
        params = {
            "client_id": client_id,
            "appid": app_id,
            "deviceid": "onevalet",
            "devicename": "OneValet",
            "response_type": "code",
            "state": state,
        }
        return f"{HueOAuth.AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        client_id, client_secret, _ = HueOAuth.get_credentials()
        basic_auth = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                HueOAuth.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
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

    @staticmethod
    async def fetch_username(access_token: str) -> str:
        """Link the app via the Hue Remote API (whitelist).

        For the Remote Hue API the access_token is sufficient,
        so this returns an empty string as a placeholder.
        """
        return ""
