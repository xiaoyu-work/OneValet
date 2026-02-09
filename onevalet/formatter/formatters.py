"""
OneValet Formatters - Provider-specific prompt formatters

This module provides formatters for:
- OpenAI (GPT-4, GPT-3.5, o1)
- Anthropic (Claude 3)
- DashScope (Qwen)
- Gemini (Google)
- Deepseek
- Ollama
- vLLM (OpenAI-compatible)
"""

import json
import logging
from typing import Dict, Any, List, Optional

from ..message import Message
from .models import Provider, FormatterConfig, ToolSchema
from .base import FormatterBase

logger = logging.getLogger(__name__)


class OpenAIFormatter(FormatterBase):
    """
    Formatter for OpenAI API (GPT-4, GPT-3.5, o1).

    OpenAI format is the "default" format that many providers
    are compatible with.
    """

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages to OpenAI format"""
        formatted = []
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "content": msg.get_text()
            }
            # Handle name field
            if msg.name and msg.role != "system":
                formatted_msg["name"] = msg.name

            formatted.append(formatted_msg)

        return formatted

    def format_tools(self, tools: List[ToolSchema]) -> Optional[List[Dict[str, Any]]]:
        """Format tools to OpenAI function calling format"""
        if not tools:
            return None

        return [tool.to_openai_format() for tool in tools]

    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse OpenAI response to normalized format"""
        # Handle dict response (from API)
        if isinstance(response, dict):
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})

            return {
                "content": message.get("content", ""),
                "tool_calls": self._parse_tool_calls(message.get("tool_calls")),
                "finish_reason": choice.get("finish_reason"),
            }

        # Handle object response (from SDK)
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message

            return {
                "content": message.content or "",
                "tool_calls": self._parse_tool_calls(
                    message.tool_calls if hasattr(message, "tool_calls") else None
                ),
                "finish_reason": choice.finish_reason,
            }

        return {"content": str(response), "tool_calls": None, "finish_reason": None}

    def _parse_tool_calls(self, tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from OpenAI response"""
        if not tool_calls:
            return None

        parsed = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                parsed.append({
                    "id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
                })
            elif hasattr(tc, "function"):
                parsed.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return parsed if parsed else None

    async def parse_stream_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """Parse OpenAI streaming chunk"""
        # Handle SSE data string
        if isinstance(chunk, str):
            if chunk.strip() == "[DONE]":
                return {"done": True}

            if chunk.startswith("data: "):
                chunk = chunk[6:]

            try:
                data = json.loads(chunk)
                return self._parse_stream_data(data)
            except json.JSONDecodeError:
                return None

        # Handle dict
        if isinstance(chunk, dict):
            return self._parse_stream_data(chunk)

        # Handle object (from SDK)
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            return {
                "content": delta.content if hasattr(delta, "content") else "",
                "tool_calls": self._parse_tool_calls(
                    delta.tool_calls if hasattr(delta, "tool_calls") else None
                ),
            }

        return None

    def _parse_stream_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse streaming data dict"""
        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        return {
            "content": delta.get("content", ""),
            "tool_calls": self._parse_tool_calls(delta.get("tool_calls")),
        }


class AnthropicFormatter(FormatterBase):
    """
    Formatter for Anthropic API (Claude 3).

    Key differences from OpenAI:
    - System prompt is separate from messages
    - Different tool calling format
    - Different streaming event format
    """

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages to Anthropic format (excludes system)"""
        formatted = []
        for msg in messages:
            # Skip system messages (handled separately)
            if msg.role == "system":
                continue

            formatted.append({
                "role": msg.role,
                "content": msg.get_text()
            })

        return formatted

    def format_tools(self, tools: List[ToolSchema]) -> Optional[List[Dict[str, Any]]]:
        """Format tools to Anthropic format"""
        if not tools:
            return None

        return [tool.to_anthropic_format() for tool in tools]

    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse Anthropic response to normalized format"""
        if isinstance(response, dict):
            content = response.get("content", [])
            text_content = ""
            tool_calls = []

            for block in content:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": block.get("input", {}),
                    })

            return {
                "content": text_content,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": response.get("stop_reason"),
            }

        # Handle SDK response object
        if hasattr(response, "content"):
            text_content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    text_content += block.text
                elif hasattr(block, "name") and hasattr(block, "input"):
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })

            return {
                "content": text_content,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": response.stop_reason if hasattr(response, "stop_reason") else None,
            }

        return {"content": str(response), "tool_calls": None, "finish_reason": None}

    async def parse_stream_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """Parse Anthropic streaming chunk"""
        # Handle SSE event
        if isinstance(chunk, str):
            # Parse "event: xxx\ndata: xxx" format
            lines = chunk.strip().split("\n")
            event_type = None
            data = None

            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

            if event_type == "content_block_delta" and data:
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    return {"content": delta.get("text", "")}

            return None

        # Handle dict
        if isinstance(chunk, dict):
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    return {"content": delta.get("text", "")}

        # Handle SDK streaming object
        if hasattr(chunk, "type"):
            if chunk.type == "content_block_delta":
                if hasattr(chunk.delta, "text"):
                    return {"content": chunk.delta.text}

        return None


class DashScopeFormatter(FormatterBase):
    """
    Formatter for DashScope API (Alibaba Qwen).

    Similar to OpenAI format with some differences.
    """

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages to DashScope format"""
        # DashScope uses same format as OpenAI for messages
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": msg.get_text()
            })
        return formatted

    def format_tools(self, tools: List[ToolSchema]) -> Optional[List[Dict[str, Any]]]:
        """Format tools to DashScope format (OpenAI-compatible)"""
        if not tools:
            return None
        return [tool.to_openai_format() for tool in tools]

    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse DashScope response"""
        if isinstance(response, dict):
            output = response.get("output", {})
            choices = output.get("choices", [{}])
            if choices:
                message = choices[0].get("message", {})
                return {
                    "content": message.get("content", ""),
                    "tool_calls": self._parse_tool_calls(message.get("tool_calls")),
                    "finish_reason": output.get("finish_reason"),
                }

        return {"content": str(response), "tool_calls": None, "finish_reason": None}

    def _parse_tool_calls(self, tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from DashScope response"""
        if not tool_calls:
            return None

        parsed = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                parsed.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name"),
                    "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")),
                })

        return parsed if parsed else None

    async def parse_stream_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """Parse DashScope streaming chunk"""
        # DashScope streaming is similar to OpenAI
        if isinstance(chunk, str):
            if chunk.startswith("data:"):
                try:
                    data = json.loads(chunk[5:].strip())
                    output = data.get("output", {})
                    choices = output.get("choices", [])
                    if choices:
                        delta = choices[0].get("message", {})
                        return {"content": delta.get("content", "")}
                except json.JSONDecodeError:
                    pass

        return None


class GeminiFormatter(FormatterBase):
    """
    Formatter for Google Gemini API.

    Uses different structure for messages and tools.
    """

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages to Gemini format"""
        formatted = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            if msg.role == "system":
                # Gemini handles system differently - prepend to first user message
                continue

            formatted.append({
                "role": role,
                "parts": [{"text": msg.get_text()}]
            })

        return formatted

    def format_tools(self, tools: List[ToolSchema]) -> Optional[List[Dict[str, Any]]]:
        """Format tools to Gemini function declarations"""
        if not tools:
            return None

        # Gemini uses functionDeclarations
        return [{
            "functionDeclarations": [tool.to_gemini_format() for tool in tools]
        }]

    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse Gemini response"""
        if isinstance(response, dict):
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                text_content = ""
                tool_calls = []

                for part in parts:
                    if "text" in part:
                        text_content += part["text"]
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append({
                            "id": fc.get("name"),  # Gemini doesn't have separate ID
                            "name": fc.get("name"),
                            "arguments": fc.get("args", {}),
                        })

                return {
                    "content": text_content,
                    "tool_calls": tool_calls if tool_calls else None,
                    "finish_reason": candidates[0].get("finishReason"),
                }

        return {"content": str(response), "tool_calls": None, "finish_reason": None}

    async def parse_stream_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """Parse Gemini streaming chunk"""
        if isinstance(chunk, dict):
            candidates = chunk.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        return {"content": part["text"]}

        return None


class DeepseekFormatter(OpenAIFormatter):
    """
    Formatter for Deepseek API.

    Uses OpenAI-compatible format.
    """
    pass  # Inherits everything from OpenAI


class OllamaFormatter(OpenAIFormatter):
    """
    Formatter for Ollama API.

    Uses OpenAI-compatible format with some adjustments.
    """

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages to Ollama format"""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role,
                "content": msg.get_text()
            })
        return formatted


class VLLMFormatter(OpenAIFormatter):
    """
    Formatter for vLLM (OpenAI-compatible).

    vLLM exposes OpenAI-compatible API endpoint.
    """
    pass  # Inherits everything from OpenAI


# Formatter factory
FORMATTER_MAP: Dict[Provider, type] = {
    Provider.OPENAI: OpenAIFormatter,
    Provider.ANTHROPIC: AnthropicFormatter,
    Provider.DASHSCOPE: DashScopeFormatter,
    Provider.GEMINI: GeminiFormatter,
    Provider.DEEPSEEK: DeepseekFormatter,
    Provider.OLLAMA: OllamaFormatter,
    Provider.VLLM: VLLMFormatter,
    Provider.OPENAI_COMPATIBLE: OpenAIFormatter,
}


def get_formatter(
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    config: Optional[FormatterConfig] = None
) -> FormatterBase:
    """
    Get appropriate formatter for a provider/model.

    Args:
        provider: LLM provider
        model: Model name (used to detect provider if not specified)
        config: Optional FormatterConfig

    Returns:
        Appropriate formatter instance
    """
    if config is None:
        config = FormatterConfig(
            provider=provider,
            model=model or "gpt-4"
        )

    # Use config's provider
    actual_provider = config.provider

    formatter_class = FORMATTER_MAP.get(actual_provider, OpenAIFormatter)
    return formatter_class(config)
