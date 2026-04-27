"""
MCP SDK Client - Concrete implementation using the official mcp package.

Supports stdio and SSE transports via AsyncExitStack to keep
context-manager-based transports alive across connect/disconnect lifecycle.

Requires: pip install mcp
"""

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from .client import MCPClient
from .models import (
    MCPPrompt,
    MCPResource,
    MCPServerConfig,
    MCPTool,
)

logger = logging.getLogger(__name__)

#: Default per-call MCP timeout (seconds).  Servers that hang after a
#: successful ``initialize`` are a common production failure mode: without
#: a hard timeout a single misbehaving server can wedge the whole agent.
_DEFAULT_MCP_TIMEOUT = float(os.environ.get("KOA_MCP_CALL_TIMEOUT", "30"))

#: Response size cap in characters. Protects against runaway tool output.
_DEFAULT_MCP_RESPONSE_CAP = int(os.environ.get("KOA_MCP_RESPONSE_CAP", "1048576"))


class MCPTimeoutError(TimeoutError):
    """Raised when an MCP call exceeds its timeout."""


def _call_timeout(config: MCPServerConfig) -> float:
    """Resolve the effective timeout for one MCP RPC."""
    t = getattr(config, "timeout", None)
    if t and t > 0:
        return float(t)
    return _DEFAULT_MCP_TIMEOUT


class MCPSDKClient(MCPClient):
    """
    MCP client backed by the official ``mcp`` Python SDK.

    Supported transports:
      - **stdio**: spawns a subprocess and communicates over stdin/stdout
      - **sse**: connects to an HTTP SSE endpoint

    Example::

        from koa.mcp.models import MCPServerConfig, MCPTransportType
        from koa.mcp.sdk_client import MCPSDKClient

        config = MCPServerConfig(
            name="filesystem",
            transport=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        client = MCPSDKClient(config)
        await client.connect()

        tools = await client.list_tools()
        result = await client.call_tool("read_file", {"path": "/tmp/hello.txt"})

        await client.disconnect()
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session = None  # mcp.client.session.ClientSession

    # ── Transport connections ────────────────────────────────────────

    async def _connect_stdio(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env or None,
        )

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(  # type: ignore[func-returns-value]
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        logger.info(f"MCP stdio session initialized: {self.config.name}")

    async def _connect_sse(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(
                url=self.config.url,
                headers=self.config.headers or None,
                timeout=self.config.timeout,
            )
        )
        self._session = await self._exit_stack.enter_async_context(  # type: ignore[func-returns-value]
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        logger.info(f"MCP SSE session initialized: {self.config.name}")

    async def _connect_websocket(self) -> None:
        raise NotImplementedError(
            "WebSocket transport is not supported by the mcp SDK. "
            "Use stdio, sse, or streamable_http transport instead."
        )

    async def _connect_streamable_http(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        read_stream, write_stream = await self._exit_stack.enter_async_context(  # type: ignore[misc]
            streamablehttp_client(
                url=self.config.url,
                headers=self.config.headers or None,
                timeout=self.config.timeout,
            )
        )
        self._session = await self._exit_stack.enter_async_context(  # type: ignore[func-returns-value]
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        logger.info(f"MCP Streamable HTTP session initialized: {self.config.name}")

    # ── Capability discovery ─────────────────────────────────────────

    async def _fetch_tools(self) -> List[MCPTool]:
        try:
            result = await asyncio.wait_for(
                self._session.list_tools(), timeout=_call_timeout(self.config)
            )
        except asyncio.TimeoutError as exc:
            raise MCPTimeoutError(
                f"MCP list_tools timed out for server {self.config.name!r}"
            ) from exc
        return [
            MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                server_name=self.config.name,
            )
            for tool in result.tools
        ]

    async def _fetch_resources(self) -> List[MCPResource]:
        try:
            result = await asyncio.wait_for(
                self._session.list_resources(), timeout=_call_timeout(self.config)
            )
        except asyncio.TimeoutError:
            logger.warning("MCP list_resources timed out for server %s", self.config.name)
            return []
        except Exception:
            # Server may not support resources
            return []
        return [
            MCPResource(
                uri=str(r.uri),
                name=r.name,
                description=getattr(r, "description", None),
                mime_type=getattr(r, "mimeType", None),
                server_name=self.config.name,
            )
            for r in result.resources
        ]

    async def _fetch_prompts(self) -> List[MCPPrompt]:
        try:
            result = await asyncio.wait_for(
                self._session.list_prompts(), timeout=_call_timeout(self.config)
            )
        except asyncio.TimeoutError:
            logger.warning("MCP list_prompts timed out for server %s", self.config.name)
            return []
        except Exception:
            # Server may not support prompts
            return []
        return [
            MCPPrompt(
                name=p.name,
                description=getattr(p, "description", None),
                arguments=[
                    {
                        "name": a.name,
                        "description": getattr(a, "description", None),
                        "required": getattr(a, "required", False),
                    }
                    for a in (p.arguments or [])
                ],
                server_name=self.config.name,
            )
            for p in result.prompts
        ]

    # ── Tool execution ───────────────────────────────────────────────

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=_call_timeout(self.config),
            )
        except asyncio.TimeoutError as exc:
            raise MCPTimeoutError(
                f"MCP tool {name!r} on server {self.config.name!r} exceeded "
                f"{_call_timeout(self.config):.1f}s timeout"
            ) from exc

        # Extract text from content blocks
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[binary: {getattr(block, 'mimeType', 'unknown')}]")

        text = "\n".join(parts) if parts else ""

        # Cap response size to protect downstream prompt budgets.
        if len(text) > _DEFAULT_MCP_RESPONSE_CAP:
            logger.warning(
                "MCP tool %s on %s returned %d chars; truncating to %d",
                name,
                self.config.name,
                len(text),
                _DEFAULT_MCP_RESPONSE_CAP,
            )
            text = text[:_DEFAULT_MCP_RESPONSE_CAP] + "\n...[truncated]"

        if result.isError:
            raise RuntimeError(text or "MCP tool returned an error")

        return text

    # ── Resource / Prompt access ─────────────────────────────────────

    async def read_resource(self, uri: str) -> Any:
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        from pydantic import AnyUrl

        result = await self._session.read_resource(AnyUrl(uri))
        parts = []
        for block in result.contents:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self._connected:
            raise ConnectionError("Not connected to MCP server")
        str_args = {k: str(v) for k, v in arguments.items()} if arguments else None
        result = await self._session.get_prompt(name, str_args)
        parts = []
        for msg in result.messages:
            # PromptMessage.content is a single ContentBlock
            block = msg.content
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    # ── Lifecycle ────────────────────────────────────────────────────

    async def disconnect(self) -> None:
        if self._exit_stack:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing MCP session for {self.config.name}: {e}")
            finally:
                self._exit_stack = None
                self._session = None
        await super().disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"MCPSDKClient(server='{self.server_name}', status={status})"
