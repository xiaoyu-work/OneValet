"""GitHub OAuth 2.0 helper.

Used to obtain a real GitHub access token for a tenant.  The token authenticates
directly against GitHub's hosted MCP server (https://api.githubcopilot.com/mcp/),
so unlike the Composio proxy path it must be a genuine GitHub credential.
"""

import os
from urllib.parse import urlencode

import httpx

#: Scopes needed by the pinned GitHub MCP tools (repo read/write, issues, PRs,
#: notifications).  `repo` covers private repositories; drop it to `public_repo`
#: if a deployment only ever needs public data.
DEFAULT_SCOPES = "repo read:org read:user notifications"


class GitHubOAuth:
    AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USER_URL = "https://api.github.com/user"

    @staticmethod
    def get_credentials() -> tuple[str, str]:
        client_id = os.getenv("GITHUB_CLIENT_ID", "")
        client_secret = os.getenv("GITHUB_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            raise ValueError("GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET not configured.")
        return client_id, client_secret

    @staticmethod
    def build_authorize_url(redirect_uri: str, state: str, scopes: str = DEFAULT_SCOPES) -> str:
        client_id, _ = GitHubOAuth.get_credentials()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scopes,
            "state": state,
        }
        return f"{GitHubOAuth.AUTHORIZE_URL}?{urlencode(params)}"

    @staticmethod
    async def exchange_code(code: str, redirect_uri: str) -> dict:
        client_id, client_secret = GitHubOAuth.get_credentials()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                GitHubOAuth.TOKEN_URL,
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # GitHub reports failures with HTTP 200 and an `error` field.
        if "error" in data:
            raise ValueError(data.get("error_description") or data["error"])

        return {
            "access_token": data["access_token"],
            "scope": data.get("scope", ""),
            "token_type": data.get("token_type", "bearer"),
        }

    @staticmethod
    async def fetch_user_login(access_token: str) -> str:
        """Return the authenticated user's login, for display after connecting."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                GitHubOAuth.USER_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            resp.raise_for_status()
            return resp.json().get("login", "")
