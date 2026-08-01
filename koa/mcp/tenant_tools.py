"""Per-tenant loading of official remote MCP servers.

Resolves a tenant's stored credential for a registered vendor server, opens an
authenticated connection, and returns the pinned tools as Koa ``AgentTool``s.

Token flow: the OAuth dance (and the encrypted-at-rest storage) is the existing
``CredentialStore`` machinery — the only MCP-specific step is putting the
resolved token in the ``Authorization`` header of the MCP transport.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from ..models import AgentTool
from .models import MCPServerConfig, MCPTransportType
from .registry import MCPServerSpec, get_server
from .sdk_client import MCPSDKClient

logger = logging.getLogger(__name__)

#: Credential keys checked, in order, for a usable bearer token.
_TOKEN_KEYS = ("access_token", "token", "api_key", "personal_access_token")


class MCPCredentialError(RuntimeError):
    """Raised when no usable credential exists for a tenant + server."""


def extract_token(credentials: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pull a bearer token out of a stored credential dict."""
    if not credentials:
        return None
    # CredentialStore.get() may return the raw dict or a {"credentials": {...}} envelope.
    inner = credentials.get("credentials")
    if isinstance(inner, dict):
        credentials = inner
    for key in _TOKEN_KEYS:
        value = credentials.get(key)
        if value:
            return str(value)
    return None


async def resolve_token(
    spec: MCPServerSpec,
    tenant_id: str,
    credential_store: Any = None,
    account_name: str = "primary",
) -> Optional[str]:
    """Find a bearer token for *tenant_id* against *spec*.

    Per-tenant credentials win over the shared env fallback, so a connected
    account is never shadowed by a deployment-wide token.
    """
    if credential_store is not None and tenant_id:
        for service in spec.credential_services:
            try:
                cred = await credential_store.get(tenant_id, service, account_name)
            except Exception as e:  # noqa: BLE001 - a broken store must not kill the agent
                logger.debug("Credential lookup failed for %s/%s: %s", tenant_id, service, e)
                continue
            token = extract_token(cred)
            if token:
                return token

    if spec.fallback_env:
        return os.getenv(spec.fallback_env) or None
    return None


def build_client(spec: MCPServerSpec, token: str, tenant_id: str = "") -> MCPSDKClient:
    """Build an authenticated MCP client for one tenant."""
    headers = dict(spec.headers)
    headers["Authorization"] = f"Bearer {token}"
    config = MCPServerConfig(
        name=spec.name,
        transport=MCPTransportType.STREAMABLE_HTTP,
        url=spec.url,
        headers=headers,
    )
    return MCPSDKClient(config)


async def load_tools(
    server_name: str,
    tenant_id: str,
    credential_store: Any = None,
    account_name: str = "primary",
) -> Tuple[List[AgentTool], MCPSDKClient]:
    """Connect to a registered MCP server as *tenant_id* and return its pinned tools.

    Returns the tools plus the live client: the caller owns that connection and
    must ``disconnect()`` it, since the tool executors keep using the session.

    Raises:
        MCPCredentialError: the tenant has not connected this service.
        ValueError: *server_name* is not in the registry.
    """
    spec = get_server(server_name)
    if spec is None:
        raise ValueError(f"unknown MCP server: {server_name}")

    token = await resolve_token(spec, tenant_id, credential_store, account_name)
    if not token:
        raise MCPCredentialError(
            f"{spec.title} is not connected. Connect it in Settings to use these tools."
        )

    client = build_client(spec, token, tenant_id)
    await client.connect()

    from .provider import MCPToolProvider

    provider = MCPToolProvider(client, tool_specs=spec.tool_specs)
    try:
        tools = await provider.discover_tools()
    except Exception:
        # Discovery failed, so nothing will ever call disconnect() for us.
        await client.disconnect()
        raise
    return tools, client
