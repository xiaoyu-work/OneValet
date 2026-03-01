"""Notion OAuth 2.0 helper."""

import base64
import os

import httpx


class NotionOAuth:
    AUTHORIZE_URL = "https://api.notion.com/v1/oauth/authorize"
    TOKEN_URL = "https://api.notion.com/v1/oauth/token"

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str) -> str:
        client_id = os.getenv("NOTION_CLIENT_ID")
        if not client_id:
            raise ValueError("NOTION_CLIENT_ID not configured.")
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "owner": "user",
            "state": state,
        }
        from urllib.parse import urlencode
        return f"{NotionOAuth.AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> dict:
        client_id = os.getenv("NOTION_CLIENT_ID", "")
        client_secret = os.getenv("NOTION_CLIENT_SECRET", "")
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                NotionOAuth.TOKEN_URL,
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/json",
                },
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "access_token": data["access_token"],
            "workspace_name": data.get("workspace_name", ""),
            "workspace_id": data.get("workspace_id", ""),
            "bot_id": data.get("bot_id", ""),
            "owner": data.get("owner", {}),
        }
