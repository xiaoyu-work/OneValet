"""
OneValet Tools - Tool calling system for LLM function calling

Provides:
- ToolDefinition: Define tools with schemas
- ToolRegistry: Register and manage tools
- ToolExecutor: Execute tools with LLM loop
- @tool decorator: Auto-register tools with type hints

Usage:
    from onevalet import tool

    @tool()
    async def send_email(to: str, subject: str, body: str) -> str:
        '''Send an email via SMTP'''
        return f"Email sent to {to}"

    # Tool is automatically registered to ToolRegistry
"""

from .models import (
    ToolCategory,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolExecutionContext,
)
from .registry import ToolRegistry
from .executor import ToolExecutor
from .decorator import (
    tool,
    get_tool_definition,
    ToolDiscovery,
    register_tools_from_module,
    register_tools_from_paths,
)

__all__ = [
    # Models
    "ToolCategory",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolExecutionContext",
    # Registry
    "ToolRegistry",
    # Executor
    "ToolExecutor",
    # Decorator
    "tool",
    "get_tool_definition",
    "ToolDiscovery",
    "register_tools_from_module",
    "register_tools_from_paths",
]
