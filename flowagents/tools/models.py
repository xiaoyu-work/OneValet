"""
FlowAgent Tool Models - Data structures for the tool system
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
from enum import Enum


class ToolCategory(str, Enum):
    """
    Base tool categories

    Users can extend this enum for custom categories:

        class MyToolCategory(str, Enum):
            EMAIL = "email"
            CALENDAR = "calendar"
            DATABASE = "database"
    """
    UTILITY = "utility"
    WEB = "web"
    USER = "user"
    CUSTOM = "custom"


@dataclass
class ToolDefinition:
    """
    Definition of a tool that can be called by LLM

    Attributes:
        name: Unique tool identifier (e.g., "search_emails")
        description: What the tool does (shown to LLM)
        parameters: JSON Schema for parameters
        executor: Async function to execute the tool
        category: Tool category for organization

    Note:
        Approval logic is handled at the Agent level, not Tool level.
        Use `requires_approval` in agent config (flowagents.yaml) instead.

    Example:
        async def search_web(query: str, context: ToolExecutionContext) -> dict:
            return {"results": [...]}

        tool = ToolDefinition(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            executor=search_web,
            category=ToolCategory.WEB
        )
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    executor: Callable
    category: ToolCategory = ToolCategory.UTILITY

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


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


@dataclass
class ToolExecutionContext:
    """
    Context passed to tool executors

    Contains user info, account info, and other context
    needed for tool execution.

    Example:
        context = ToolExecutionContext(
            user_id="user_123",
            account_spec="work@company.com",
            metadata={"timezone": "UTC"}
        )
    """
    user_id: str
    account_spec: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata"""
        return self.metadata.get(key, default)


