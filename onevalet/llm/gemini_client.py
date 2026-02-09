"""
OneValet Gemini Client - Built-in Google Gemini API client

Supports:
- Gemini 1.5 Pro, Gemini 1.5 Flash
- Gemini 1.0 Pro
- Tool use and function calling
"""

import json
import os
from typing import Dict, Any, List, Optional, AsyncIterator

from .base import (
    BaseLLMClient, LLMConfig, LLMResponse, StreamChunk,
    ToolCall, Usage, StopReason
)


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client.

    Example:
        # Basic usage
        client = GeminiClient(api_key="xxx")
        response = await client.chat_completion([
            {"role": "user", "content": "Hello!"}
        ])

        # With tools
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[weather_tool]
        )

        # Streaming
        async for chunk in client.stream_completion(messages):
            print(chunk.content, end="")
    """

    provider = "gemini"

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-pro-latest": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-flash-latest": {"input": 0.000075, "output": 0.0003},
        "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize Gemini client.

        Args:
            config: LLMConfig instance
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Model name (default: gemini-1.5-pro)
            **kwargs: Additional config options
        """
        # Get API key from env if not provided
        if config is None and "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("GOOGLE_API_KEY")

        if config is None:
            if "model" not in kwargs:
                raise ValueError("model is required")
            model = kwargs.pop("model")
            config = LLMConfig(model=model, **kwargs)

        super().__init__(config, **kwargs)

    def _get_client(self):
        """Get or create the Gemini client"""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )

            genai.configure(api_key=self.config.api_key)
            self._client = genai

        return self._client

    def _format_tool(self, tool) -> Dict[str, Any]:
        """Format tool to Gemini format"""
        if hasattr(tool, "name"):
            # ToolDefinition
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        elif isinstance(tool, dict):
            if "function" in tool:
                # OpenAI format - convert
                func = tool["function"]
                return {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}}),
                }
            return tool
        return tool

    def _convert_messages(self, messages: List[Dict[str, Any]], media: Optional[List[Dict[str, Any]]] = None) -> tuple:
        """Convert messages to Gemini format, extracting system prompt"""
        import base64

        system_instruction = None
        gemini_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                parts = [content] if isinstance(content, str) else []

                # Add media to the last user message
                if media and msg == messages[-1]:
                    for item in media:
                        if item.get("type") == "image":
                            data = item.get("data", "")
                            media_type = item.get("media_type", "image/jpeg")

                            if data.startswith("http://") or data.startswith("https://"):
                                # URL-based image - Gemini needs inline_data
                                # For now, skip URL images as Gemini prefers base64
                                pass
                            else:
                                # Base64 image
                                parts.insert(0, {
                                    "inline_data": {
                                        "mime_type": media_type,
                                        "data": data
                                    }
                                })

                gemini_messages.append({"role": "user", "parts": parts})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        return system_instruction, gemini_messages

    async def _call_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Make Gemini API call"""
        genai = self._get_client()

        # Handle media
        media = kwargs.pop("media", None)

        # Convert messages
        system_instruction, gemini_messages = self._convert_messages(messages, media)

        # Build model config
        model_name = kwargs.get("model", self.config.model)

        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Create model
        model_kwargs = {
            "model_name": model_name,
            "generation_config": generation_config,
        }

        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = genai.GenerativeModel(**model_kwargs)

        # Add tools if provided
        tool_config = None
        if tools:
            formatted_tools = [self._format_tool(t) for t in tools]
            model = genai.GenerativeModel(
                **model_kwargs,
                tools=[{"function_declarations": formatted_tools}],
            )

        # Start chat with history
        chat = model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])

        # Send the last message
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        response = await chat.send_message_async(last_message)

        # Parse response
        content = ""
        tool_calls = []

        for part in response.parts:
            if hasattr(part, "text"):
                content += part.text
            elif hasattr(part, "function_call"):
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=fc.name,  # Gemini doesn't have separate IDs
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))

        # Parse stop reason
        stop_reason = self._parse_stop_reason(
            response.candidates[0].finish_reason if response.candidates else None
        )

        # Parse usage
        usage = None
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            usage = Usage(
                prompt_tokens=um.prompt_token_count,
                completion_tokens=um.candidates_token_count,
                total_tokens=um.total_token_count,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            usage=usage,
            model=model_name,
            raw_response=response,
        )

    async def _stream_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Make streaming Gemini API call"""
        genai = self._get_client()

        # Convert messages
        system_instruction, gemini_messages = self._convert_messages(messages)

        model_name = kwargs.get("model", self.config.model)

        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        model_kwargs = {
            "model_name": model_name,
            "generation_config": generation_config,
        }

        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        model = genai.GenerativeModel(**model_kwargs)

        if tools:
            formatted_tools = [self._format_tool(t) for t in tools]
            model = genai.GenerativeModel(
                **model_kwargs,
                tools=[{"function_declarations": formatted_tools}],
            )

        chat = model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])

        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""

        # Stream response
        response = await chat.send_message_async(last_message, stream=True)

        tool_calls = []
        async for chunk in response:
            content = ""
            for part in chunk.parts:
                if hasattr(part, "text"):
                    content += part.text
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=fc.name,
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

            # Check if this is the final chunk
            is_final = False
            stop_reason = None
            usage = None

            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.finish_reason:
                    is_final = True
                    stop_reason = self._parse_stop_reason(candidate.finish_reason)

            yield StreamChunk(
                content=content,
                tool_calls=tool_calls if is_final and tool_calls else None,
                is_final=is_final,
                stop_reason=stop_reason,
                usage=usage,
            )

    def _parse_stop_reason(self, finish_reason) -> StopReason:
        """Parse Gemini finish_reason to StopReason"""
        if finish_reason is None:
            return StopReason.END_TURN

        # Gemini uses enum
        reason_str = str(finish_reason).upper()

        if "STOP" in reason_str:
            return StopReason.END_TURN
        elif "MAX_TOKENS" in reason_str:
            return StopReason.MAX_TOKENS
        elif "SAFETY" in reason_str:
            return StopReason.END_TURN
        elif "FUNCTION" in reason_str or "TOOL" in reason_str:
            return StopReason.TOOL_USE

        return StopReason.END_TURN

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        genai = self._get_client()
        model = genai.GenerativeModel(self.config.model)
        result = model.count_tokens(text)
        return result.total_tokens
