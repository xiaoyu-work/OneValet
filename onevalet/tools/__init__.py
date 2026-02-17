"""
OneValet Tools - Data structures for LLM tool calling

Provides:
- ToolCall: Represents a tool call from LLM response
- ToolResult: Result of a tool execution
"""

from .models import (
    ToolCall,
    ToolResult,
)

__all__ = [
    "ToolCall",
    "ToolResult",
]
