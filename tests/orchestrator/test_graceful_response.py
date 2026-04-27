"""Tests for koa.orchestrator.graceful_response"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from koa.orchestrator.graceful_response import (
    FALLBACK_MESSAGES,
    generate_graceful_error,
    get_fallback_message,
)

# ── FALLBACK_MESSAGES pool ──


class TestFallbackPool:
    def test_fallback_pool_not_empty(self):
        assert len(FALLBACK_MESSAGES) >= 5

    def test_fallback_messages_are_strings(self):
        for msg in FALLBACK_MESSAGES:
            assert isinstance(msg, str)
            assert len(msg) > 0


# ── get_fallback_message ──


class TestGetFallbackMessage:
    def test_get_fallback_message_returns_from_pool(self):
        for _ in range(20):
            assert get_fallback_message() in FALLBACK_MESSAGES


# ── generate_graceful_error ──


class TestGenerateGracefulError:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        result = await generate_graceful_error(RuntimeError("boom"))
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_no_llm_client_returns_fallback(self):
        result = await generate_graceful_error(RuntimeError("boom"), llm_client=None)
        assert result in FALLBACK_MESSAGES

    @pytest.mark.asyncio
    async def test_llm_success_returns_generated(self):
        mock_response = MagicMock()
        mock_response.content = "whoops, brain fart. try again?"

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = mock_response

        result = await generate_graceful_error(RuntimeError("boom"), llm_client=mock_client)
        assert result == "whoops, brain fart. try again?"
        mock_client.chat_completion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self):
        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = Exception("LLM exploded")

        result = await generate_graceful_error(RuntimeError("boom"), llm_client=mock_client)
        assert result in FALLBACK_MESSAGES

    @pytest.mark.asyncio
    async def test_llm_empty_response_returns_fallback(self):
        mock_response = MagicMock()
        mock_response.content = ""

        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = mock_response

        result = await generate_graceful_error(RuntimeError("boom"), llm_client=mock_client)
        assert result in FALLBACK_MESSAGES

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback(self):
        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(10)

        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = slow_llm

        result = await generate_graceful_error(
            RuntimeError("boom"), llm_client=mock_client, timeout=0.1
        )
        assert result in FALLBACK_MESSAGES
