"""
OneValet Models - Shared dataclasses used across the framework

This module contains the core data structures extracted from standard_agent
so that other modules can import them without pulling in the full agent class.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm.base import BaseLLMClient
    from .credentials.store import CredentialStore


# ===== Field Definition =====

@dataclass
class RequiredField:
    """
    Defines a required field for an agent

    Attributes:
        name: Field name (e.g., "recipient", "subject")
        description: Human-readable description
        prompt: Question to ask user when field is missing
        validator: Optional validation function (returns bool)
        required: Whether this field is required (default: True)

    Example:
        RequiredField(
            name="email",
            description="Recipient email address",
            prompt="What email address should I send to?",
            validator=lambda v: "@" in v,  # Custom validator
            required=True
        )
    """
    name: str
    description: str
    prompt: str
    validator: Optional[Callable[[str], bool]] = None
    required: bool = True


# ===== Agent Tool Definitions =====


@dataclass
class AgentToolContext:
    """Context passed to tool executors.

    Provides access to shared resources that tool functions need.
    Used by both agent-level tools (tools) and orchestrator-level
    builtin tools.
    """

    llm_client: Optional["BaseLLMClient"] = None
    tenant_id: str = ""
    user_profile: Optional[Dict[str, Any]] = None
    context_hints: Optional[Dict[str, Any]] = None
    credentials: Optional["CredentialStore"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTool:
    """A tool available inside a StandardAgent's mini ReAct loop.

    Attributes:
        name: Tool function name (used in LLM tool_calls).
        description: What this tool does (shown to the LLM).
        parameters: JSON Schema for tool arguments.
        executor: Async function(args: dict, context: AgentToolContext) -> str.
        needs_approval: If True, pause execution for user confirmation before running.
        risk_level: One of "read", "write", "destructive".
        get_preview: Async function to generate human-readable preview for approval.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    executor: Callable
    needs_approval: bool = False
    risk_level: str = "read"  # "read", "write", "destructive"
    category: str = "utility"
    get_preview: Optional[Callable] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function-calling tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
