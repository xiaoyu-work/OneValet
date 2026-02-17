"""
OneValet Tool Models - Data structures for LLM tool calling
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ToolCall:
    """
    Represents a tool call from LLM response

    Attributes:
        id: Unique call ID from LLM
        name: Tool name
        arguments: Parsed arguments dict
    """
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """
    Result of a tool execution

    Attributes:
        tool_call_id: ID of the tool call this result is for
        content: String result content
        is_error: Whether execution failed
        data: Optional structured data for further processing
    """
    tool_call_id: str
    content: str
    is_error: bool = False
    data: Optional[Dict[str, Any]] = None
