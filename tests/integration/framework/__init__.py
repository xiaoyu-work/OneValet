"""Multi-turn conversation testing framework for LLM agent integration tests.

Provides a :class:`Conversation` wrapper that handles multi-turn flows
(text confirmations, approval gates, etc.) and assertion helpers for
verifying tool selection, argument extraction, and response quality.

Quick start::

    async def test_tool_selection(conversation):
        conv = await conversation()
        await conv.send_until_tool_called("Find Italian restaurants near me")
        conv.assert_tool_called("search_places")
"""

from .assertions import (
    assert_any_tool_called,
    assert_status,
    assert_tool_args,
    assert_tool_called,
)
from .conversation import Conversation, ConversationError
from .types import MessageHandler, ToolRecorder

__all__ = [
    "Conversation",
    "ConversationError",
    "MessageHandler",
    "ToolRecorder",
    "assert_any_tool_called",
    "assert_status",
    "assert_tool_args",
    "assert_tool_called",
]
