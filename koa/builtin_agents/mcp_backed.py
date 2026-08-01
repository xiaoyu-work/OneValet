"""Base class for agents whose tools come from a vendor-hosted MCP server.

``StandardAgent.tools`` is a class attribute, but MCP tools are per-tenant: they
depend on that tenant's credential and are only known after ``list_tools()``.
This class keeps the class attribute empty and populates an instance-level
``self.tools`` on first use, so each agent instance carries exactly the tools its
own tenant is authorised for.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from koa.mcp.registry import get_server
from koa.mcp.tenant_tools import MCPCredentialError, load_tools
from koa.message import Message
from koa.standard_agent import AgentResult, AgentStatus, StandardAgent

logger = logging.getLogger(__name__)


class MCPBackedAgent(StandardAgent):
    """A StandardAgent that sources its tools from a registered MCP server."""

    #: Registry key from koa.mcp.registry.SERVERS.
    mcp_server: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Shadow the empty class attribute with a per-instance tuple.
        self.tools: tuple = ()
        self._mcp_client = None
        self._mcp_loaded = False

    def _credential_store(self):
        """Find the CredentialStore for this request.

        Tools receive it through AgentToolContext, but tool *loading* happens
        before any tool runs, so fall back to the process-wide default the
        AccountResolver already holds.
        """
        store = (self.context_hints or {}).get("credential_store")
        if store is not None:
            return store
        try:
            from koa.providers.email.resolver import AccountResolver

            return AccountResolver().credential_store
        except Exception:  # noqa: BLE001
            return None

    async def _ensure_mcp_tools(self) -> Optional[str]:
        """Load this tenant's MCP tools. Returns an error message on failure."""
        if self._mcp_loaded:
            return None

        spec = get_server(self.mcp_server)
        if spec is None:
            return f"Unknown MCP server: {self.mcp_server}"

        try:
            tools, client = await load_tools(
                self.mcp_server,
                tenant_id=self.tenant_id,
                credential_store=self._credential_store(),
            )
        except MCPCredentialError as e:
            return str(e)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to load %s MCP tools for tenant %s: %s",
                self.mcp_server,
                self.tenant_id,
                e,
                exc_info=True,
            )
            return f"Couldn't reach {spec.title} right now. Please try again."

        self.tools = tuple(tools)
        self._mcp_client = client
        self._mcp_loaded = True
        # Schemas changed, so a validator built from the old tuple is stale.
        self._tool_validator = None
        logger.info(
            "Loaded %d %s MCP tools for tenant %s",
            len(tools),
            self.mcp_server,
            self.tenant_id,
        )
        return None

    async def reply(self, msg: Message = None) -> AgentResult:
        # Tools must exist before the state machine inspects self.tools: with an
        # empty tuple, on_running() falls through to the no-tools branch and the
        # agent completes without doing anything.
        error = await self._ensure_mcp_tools()
        if error:
            return self.make_result(status=AgentStatus.COMPLETED, raw_message=error)
        try:
            return await super().reply(msg)
        finally:
            await self._release_mcp_client()

    async def _release_mcp_client(self) -> None:
        """Close the MCP connection once the agent is done with it.

        Only terminal states release: WAITING_FOR_APPROVAL/INPUT resume on a
        later reply() and need their tools to stay live.
        """
        if self.status in (
            AgentStatus.WAITING_FOR_APPROVAL,
            AgentStatus.WAITING_FOR_INPUT,
            AgentStatus.PAUSED,
        ):
            return
        client = self._mcp_client
        if client is None:
            return
        self._mcp_client = None
        self._mcp_loaded = False
        self.tools = ()
        try:
            await client.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.debug("MCP disconnect failed for %s: %s", self.mcp_server, e)
