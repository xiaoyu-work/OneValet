"""
Composio API Client - Shared REST API wrapper for Composio-powered agents.

All Composio agents use this client for API calls.
Requires COMPOSIO_API_KEY environment variable.
"""

import json
import os
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://backend.composio.dev/api"


class ComposioClient:
    """Async Composio REST API client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COMPOSIO_API_KEY", "")

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    # ── Actions ──

    async def list_actions(
        self,
        app_name: str,
        use_case: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        List available actions for a given app.

        Args:
            app_name: App identifier (e.g. "github", "slack", "gmail").
            use_case: Optional natural language filter for relevance.
            limit: Maximum number of actions to return.
        """
        params: Dict[str, Any] = {
            "appNames": app_name,
            "limit": limit,
        }
        if use_case:
            params["useCase"] = use_case

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/v2/actions",
                headers=self._headers,
                params=params,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        entity_id: str = "default",
        connected_account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Composio action.

        Args:
            action_name: The action identifier (e.g. "GITHUB_CREATE_AN_ISSUE").
            params: Input parameters for the action.
            entity_id: Entity ID representing the user.
            connected_account_id: Optional specific connected account UUID.
        """
        body: Dict[str, Any] = {
            "input": params,
            "entityId": entity_id,
        }
        if connected_account_id:
            body["connectedAccountId"] = connected_account_id

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/v2/actions/{action_name}/execute",
                headers=self._headers,
                json=body,
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Connected Accounts ──

    async def initiate_connection(
        self,
        app_name: str,
        entity_id: str = "default",
        redirect_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initiate an OAuth connection to an app.

        Args:
            app_name: The app to connect (e.g. "github", "slack").
            entity_id: Entity ID representing the user.
            redirect_url: Optional redirect URL after OAuth completes.
        """
        body: Dict[str, Any] = {
            "integrationId": app_name,
            "entityId": entity_id,
        }
        if redirect_url:
            body["redirectUrl"] = redirect_url

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/v1/connectedAccounts",
                headers=self._headers,
                json=body,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def list_connections(
        self,
        entity_id: str = "default",
    ) -> Dict[str, Any]:
        """
        List connected accounts for an entity.

        Args:
            entity_id: Entity ID to list connections for.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/v1/connectedAccounts",
                headers=self._headers,
                params={"user_uuid": entity_id},
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()

    # ── Helpers ──

    @staticmethod
    def format_action_result(data: Dict[str, Any]) -> str:
        """Format an execute_action response into a readable string."""
        if data.get("successfull") or data.get("successful"):
            response_data = data.get("data", {})
            if isinstance(response_data, dict):
                parts = []
                for key, value in response_data.items():
                    display = str(value)
                    if len(display) > 300:
                        display = display[:297] + "..."
                    parts.append(f"  {key}: {display}")
                return "\n".join(parts) if parts else json.dumps(response_data)
            return str(response_data)
        else:
            error = data.get("error", data.get("message", "Unknown error"))
            return f"Error: {error}"
