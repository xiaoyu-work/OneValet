"""
OneValet Tool Executor - Execute tools and handle tool calling loop

Note:
    Approval logic is handled at the Agent/Orchestrator level, not here.
    Use `requires_approval` in agent config (onevalet.yaml) to require
    user confirmation before an agent executes.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from ..protocols import LLMClientProtocol
from .registry import ToolRegistry
from .models import (
    ToolCall,
    ToolResult,
    ToolExecutionContext,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tools and manages the tool calling loop with LLM

    Usage:
        executor = ToolExecutor(llm_client=my_llm_client)
        result = await executor.run_with_tools(
            messages=[{"role": "user", "content": "Search the web for AI news"}],
            tool_names=["search_web", "summarize"],
            context=ToolExecutionContext(user_id="123")
        )
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize ToolExecutor

        Args:
            llm_client: LLM client implementing LLMClientProtocol (required)
            registry: ToolRegistry instance (defaults to singleton)
        """
        if llm_client is None:
            raise ValueError("llm_client is required")

        self.llm_client = llm_client
        self.registry = registry or ToolRegistry.get_instance()

    async def run_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tool_names: List[str],
        context: ToolExecutionContext,
        max_iterations: int = 10,
        llm_config: Optional[Dict[str, Any]] = None,
        media: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Run conversation with tool calling loop

        Args:
            messages: Conversation messages
            tool_names: List of tool names available to use
            context: Execution context with user info
            max_iterations: Maximum tool call iterations
            llm_config: Optional LLM configuration
            media: Optional media attachments (images) for vision models

        Returns:
            Final text response
        """
        # Get tool schemas
        tools_schema = self.registry.get_tools_schema(tool_names)

        if not tools_schema:
            logger.warning("No valid tools found, running without tools")

        # Make a copy of messages to avoid modifying original
        messages = messages.copy()

        for iteration in range(max_iterations):
            logger.debug(f"Tool loop iteration {iteration + 1}/{max_iterations}")

            # Build LLM call params
            llm_params = {
                "messages": messages,
                "tools": tools_schema if tools_schema else None,
                "config": llm_config
            }

            # Add media only on first iteration (image is already in context after that)
            if media and iteration == 0:
                llm_params["media"] = media

            # Call LLM
            response = await self.llm_client.chat_completion(**llm_params)

            # Handle onevalet LLMResponse format
            if hasattr(response, 'content'):
                # onevalet LLMResponse
                content = response.content
                tool_calls = response.tool_calls
            else:
                # OpenAI format fallback
                message = response.choices[0].message
                content = message.content
                tool_calls = getattr(message, 'tool_calls', None)

            # Check if LLM wants to call tools
            if not tool_calls:
                # No tool calls, return final response
                return content or ""

            # Add assistant message with tool calls to history
            messages.append(self._build_assistant_message(content, tool_calls))

            # Process each tool call
            for tool_call in tool_calls:
                parsed_call = self._parse_tool_call(tool_call)
                tool = self.registry.get_tool(parsed_call.name)

                if not tool:
                    # Unknown tool
                    result = ToolResult(
                        tool_call_id=parsed_call.id,
                        content=f"Error: Unknown tool '{parsed_call.name}'",
                        is_error=True
                    )
                    messages.append(self._tool_result_to_message(result))
                    continue

                # Execute tool
                result = await self._execute_tool(parsed_call, context)
                messages.append(self._tool_result_to_message(result))

                logger.info(f"Tool '{parsed_call.name}' executed: {'error' if result.is_error else 'success'}")

        # Exceeded max iterations
        logger.error(f"Tool execution exceeded {max_iterations} iterations")
        return "I'm having trouble completing this task. Please try again with a simpler request."

    async def execute_single_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolResult:
        """
        Execute a single tool directly (without LLM loop)

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            context: Execution context

        Returns:
            ToolResult
        """
        tool_call = ToolCall(
            id="direct_call",
            name=tool_name,
            arguments=arguments
        )
        return await self._execute_tool(tool_call, context)

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        context: ToolExecutionContext
    ) -> ToolResult:
        """Execute a single tool call"""
        tool = self.registry.get_tool(tool_call.name)

        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True
            )

        try:
            # Execute tool with arguments and context
            result = await tool.executor(
                **tool_call.arguments,
                context=context
            )

            # Convert result to string if needed
            if isinstance(result, dict):
                content = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                content = str(result)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                data=result if isinstance(result, dict) else None
            )

        except Exception as e:
            logger.error(f"Tool '{tool_call.name}' execution failed: {e}", exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error executing {tool_call.name}: {str(e)}",
                is_error=True
            )

    def _parse_tool_call(self, tool_call) -> ToolCall:
        """Parse LLM tool call to ToolCall object"""
        # Handle onevalet ToolCall format
        if isinstance(tool_call, ToolCall):
            return tool_call

        # Handle onevalet LLMResponse.tool_calls format
        if hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
            return ToolCall(
                id=getattr(tool_call, 'id', 'unknown'),
                name=tool_call.name,
                arguments=tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            )

        # Handle OpenAI format
        try:
            arguments = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            arguments = {}

        return ToolCall(
            id=getattr(tool_call, 'id', 'unknown'),
            name=tool_call.function.name,
            arguments=arguments
        )

    def _build_assistant_message(self, content: str, tool_calls: List) -> Dict[str, Any]:
        """Build assistant message dict with tool calls"""
        msg_dict = {
            "role": "assistant",
            "content": content
        }

        if tool_calls:
            msg_dict["tool_calls"] = []
            for tc in tool_calls:
                # Handle onevalet ToolCall format
                if hasattr(tc, 'name') and hasattr(tc, 'arguments'):
                    msg_dict["tool_calls"].append({
                        "id": getattr(tc, 'id', 'unknown'),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                        }
                    })
                # Handle OpenAI format
                elif hasattr(tc, 'function'):
                    msg_dict["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })

        return msg_dict

    def _tool_result_to_message(self, result: ToolResult) -> Dict[str, Any]:
        """Convert ToolResult to message format"""
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content
        }
