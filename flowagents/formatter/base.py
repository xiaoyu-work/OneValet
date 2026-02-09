"""
FlowAgent Formatter Base - Base class for prompt formatters

This module provides:
- FormatterBase: Abstract base class for all formatters
- Common formatting utilities
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator

from ..message import Message
from .models import (
    Provider,
    FormatterConfig,
    ToolSchema,
    FormattedRequest,
    TruncationStrategy,
)

logger = logging.getLogger(__name__)


class FormatterBase(ABC):
    """
    Abstract base class for prompt formatters.

    Each provider has its own formatter that converts:
    - Messages to provider-specific format
    - Tools to provider-specific format
    - Streaming chunks to normalized format

    Usage:
        formatter = OpenAIFormatter(config)
        request = await formatter.format(messages, tools)
        # Send request to provider...

        # Parse streaming response
        async for chunk in response:
            content = await formatter.parse_stream_chunk(chunk)
    """

    def __init__(self, config: Optional[FormatterConfig] = None):
        self.config = config or FormatterConfig()

    @property
    def provider(self) -> Provider:
        """Get the provider this formatter is for"""
        return self.config.provider

    @property
    def context_limit(self) -> int:
        """Get context window limit"""
        return self.config.context_limit

    # ==========================================================================
    # Abstract Methods - Must implement in subclasses
    # ==========================================================================

    @abstractmethod
    def format_messages(
        self,
        messages: List[Message]
    ) -> Any:
        """
        Format messages to provider-specific format.

        Args:
            messages: List of Message objects

        Returns:
            Provider-specific message format
        """
        pass

    @abstractmethod
    def format_tools(
        self,
        tools: List[ToolSchema]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Format tools to provider-specific format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            Provider-specific tools format, or None if no tools
        """
        pass

    @abstractmethod
    def parse_response(
        self,
        response: Any
    ) -> Dict[str, Any]:
        """
        Parse provider response to normalized format.

        Args:
            response: Raw provider response

        Returns:
            Normalized response dict with:
            - content: Text content
            - tool_calls: List of tool calls (if any)
            - finish_reason: Why generation stopped
        """
        pass

    @abstractmethod
    async def parse_stream_chunk(
        self,
        chunk: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a streaming chunk to normalized format.

        Args:
            chunk: Raw streaming chunk

        Returns:
            Normalized chunk dict or None if not content
        """
        pass

    # ==========================================================================
    # Common Methods
    # ==========================================================================

    async def format(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None
    ) -> FormattedRequest:
        """
        Format messages and tools for the provider.

        This is the main entry point. It handles:
        1. Context window management (truncation if needed)
        2. Message formatting
        3. Tool formatting

        Args:
            messages: List of Message objects
            tools: Optional list of ToolSchema objects

        Returns:
            FormattedRequest ready for API call
        """
        # Apply context management if enabled
        ctx_config = self.config.context_management
        if ctx_config.get("enabled", True):
            messages = await self._manage_context(messages)

        # Format messages
        formatted_messages = self.format_messages(messages)

        # Format tools
        formatted_tools = None
        if tools:
            formatted_tools = self.format_tools(tools)

        # Extract system prompt for Anthropic
        system_prompt = None
        if self.provider == Provider.ANTHROPIC:
            system_prompt = self._extract_system_prompt(messages)

        return FormattedRequest(
            provider=self.provider,
            model=self.config.model,
            messages=formatted_messages,
            tools=formatted_tools,
            system_prompt=system_prompt,
            extra_params={
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
        )

    def _extract_system_prompt(self, messages: List[Message]) -> Optional[str]:
        """Extract system prompt from messages"""
        for msg in messages:
            if msg.role == "system":
                return msg.get_text()
        return None

    async def _manage_context(
        self,
        messages: List[Message]
    ) -> List[Message]:
        """
        Manage context window - truncate if needed.

        Uses the configured truncation strategy.
        """
        ctx_config = self.config.context_management
        strategy = ctx_config.get("strategy", TruncationStrategy.SMART_TRUNCATE.value)

        # Estimate current tokens (rough estimate: 4 chars = 1 token)
        total_chars = sum(len(m.get_text()) for m in messages)
        estimated_tokens = total_chars // 4

        # Add reserve for response
        reserve = ctx_config.get("reserve_tokens", 1000)
        available = self.context_limit - reserve

        if estimated_tokens <= available:
            return messages  # No truncation needed

        logger.debug(f"Context overflow: ~{estimated_tokens} tokens, limit {available}")

        if strategy == TruncationStrategy.SMART_TRUNCATE.value:
            return await self._smart_truncate(messages, available)
        elif strategy == TruncationStrategy.SLIDING_WINDOW.value:
            return self._sliding_window(messages, ctx_config.get("keep_last_n", 10))
        elif strategy == TruncationStrategy.DROP_OLDEST.value:
            return self._drop_oldest(messages, available)
        elif strategy == TruncationStrategy.ERROR.value:
            raise ValueError(f"Context overflow: ~{estimated_tokens} tokens exceeds {available}")
        else:
            return messages

    async def _smart_truncate(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> List[Message]:
        """
        Smart truncation: keep system + recent + important messages.

        Priority:
        1. System prompt (always keep)
        2. Last N messages (most recent context)
        3. Tool calls and results (important for continuity)
        4. Summarize middle if needed
        """
        ctx_config = self.config.context_management
        keep_last_n = ctx_config.get("keep_last_n", 10)

        # Separate message types
        system_msgs = [m for m in messages if m.role == "system"]
        recent_msgs = messages[-keep_last_n:] if len(messages) > keep_last_n else messages
        tool_msgs = [m for m in messages
                     if (m.metadata and m.metadata.get("is_tool")) or m.role == "tool"]

        # Combine unique messages
        seen_ids = set()
        result = []

        for msg in system_msgs + recent_msgs + tool_msgs:
            msg_id = id(msg)
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                result.append(msg)

        return result

    def _sliding_window(
        self,
        messages: List[Message],
        window_size: int
    ) -> List[Message]:
        """Keep last N messages plus system prompt"""
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        return system_msgs + other_msgs[-window_size:]

    def _drop_oldest(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> List[Message]:
        """Drop oldest messages until under limit"""
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        # Keep dropping oldest until under limit
        while other_msgs:
            total_chars = sum(len(m.get_text()) for m in system_msgs + other_msgs)
            if total_chars // 4 <= max_tokens:
                break
            other_msgs.pop(0)

        return system_msgs + other_msgs

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars = 1 token)"""
        return len(text) // 4

    def count_message_tokens(self, messages: List[Message]) -> int:
        """Count total tokens in messages"""
        return sum(self.estimate_tokens(m.get_text()) for m in messages)
