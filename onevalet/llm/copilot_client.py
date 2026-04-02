"""
GitHub Copilot LLM Client

Extends LiteLLMClient to handle Copilot's two-tier token system:
- Automatically refreshes the short-lived Copilot API token before each call
- Injects required Copilot headers (Editor-Version, Copilot-Integration-Id)
- Routes through litellm as an OpenAI-compatible provider

Usage:
    from onevalet.llm.copilot_client import CopilotLLMClient
    from onevalet.llm.copilot_auth import CopilotTokenManager

    manager = CopilotTokenManager(github_token="gho_xxx")
    client = CopilotLLMClient(model="claude-sonnet-4.6", token_manager=manager)

    response = await client.chat_completion([
        {"role": "user", "content": "Hello!"}
    ])
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import LLMConfig, LLMResponse, StreamChunk
from .copilot_auth import CopilotTokenManager, COPILOT_EXTRA_HEADERS
from .litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class CopilotLLMClient(LiteLLMClient):
    """LLM client for GitHub Copilot.

    Wraps LiteLLMClient to transparently manage Copilot's two-tier auth:
    1. Before each API call, ensures a valid Copilot token
    2. Updates api_key and api_base with fresh credentials
    3. Injects required Copilot headers

    The Copilot API is OpenAI-compatible, so litellm handles it via the
    "openai/" model prefix with a custom base URL.
    """

    def __init__(
        self,
        token_manager: CopilotTokenManager,
        model: str = "gpt-4o",
        config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Initialize CopilotLLMClient.

        Args:
            token_manager: CopilotTokenManager for handling token lifecycle.
            model: Model name (e.g., "claude-sonnet-4.6", "gpt-4o").
            config: Optional LLMConfig (api_key and base_url will be overridden).
            **kwargs: Additional config overrides.
        """
        if config is None:
            config = LLMConfig(model=model, **kwargs)

        # Initialize as openai-compatible provider (Copilot API is OpenAI-format)
        super().__init__(config=config, provider_name="openai")

        self._token_manager = token_manager
        self._refresh_lock = asyncio.Lock()

        logger.info(f"CopilotLLMClient initialized: model={model}")

    async def _refresh_credentials(self) -> None:
        """Refresh the Copilot token and update internal state."""
        async with self._refresh_lock:
            copilot_token, base_url = await self._token_manager.get_token()

            # Update the base kwargs used by LiteLLMClient._call_api / _stream_api
            self._base_kwargs["api_key"] = copilot_token
            self._base_kwargs["api_base"] = base_url
            self._base_kwargs["extra_headers"] = COPILOT_EXTRA_HEADERS
            self._api_key = copilot_token

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Make a non-streaming call, refreshing credentials first."""
        await self._refresh_credentials()
        return await super()._call_api(messages, tools, **kwargs)

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Make a streaming call, refreshing credentials first."""
        await self._refresh_credentials()
        async for chunk in super()._stream_api(messages, tools, **kwargs):
            yield chunk
