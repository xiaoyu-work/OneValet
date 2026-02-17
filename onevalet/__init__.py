"""
OneValet - A zero-code AI workflow orchestration framework

OneValet provides a simple yet powerful framework for building conversational AI agents
and orchestrating multi-agent workflows with minimal code.

Key Features:
- Decorator-based agent registration (@valet)
- InputField/OutputField for clean field definitions
- State machine for conversation flow management
- Custom validators with error messages
- Built-in LLM clients (OpenAI, Anthropic, etc.)
- Built-in streaming support
- One-click logging with @logged decorator

Quick Start (Recommended):
    from onevalet import valet, StandardAgent, InputField, OutputField, AgentStatus

    @valet()
    class SendEmailAgent(StandardAgent):
        '''Send emails to users'''

        # Inputs - collected from user
        recipient = InputField(
            prompt="Who should I send to?",
            validator=lambda x: None if "@" in x else "Invalid email format",
        )
        subject = InputField("What's the subject?", required=False)

        # Outputs
        message_id = OutputField(str, "ID of sent message")

        async def on_running(self, msg):
            # Access inputs directly
            to = self.recipient

            # Set outputs
            self.message_id = "123"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Email sent to {to}!"
            )

    # Minimal version
    @valet
    class HelloAgent(StandardAgent):
        '''Say hello'''

        name = InputField("What's your name?")

        async def on_running(self, msg):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Hello, {self.name}!"
            )

Built-in LLM Client (powered by litellm):
    from onevalet.llm import LiteLLMClient

    client = LiteLLMClient(model="gpt-4o", provider_name="openai", api_key="sk-xxx")
    response = await client.chat_completion(messages=[...])

    # With streaming
    async for chunk in client.stream_completion(messages=[...]):
        print(chunk.content, end="")

One-click Logging:
    from onevalet.hooks import logged

    @logged
    class MyAgent(StandardAgent):
        # All state changes, method calls, errors logged automatically!
        ...

Streaming:
    agent = MyAgent(tenant_id="123")
    async for event in agent.stream(msg):
        if event.type == EventType.MESSAGE_CHUNK:
            print(event.data["chunk"], end="")
"""

__version__ = "0.1.1"

# Fields (InputField/OutputField descriptors)
from .fields import InputField, OutputField

# Agent Decorator
from .agents import (
    valet,
    get_agent_metadata,
    is_valet,
    AgentMetadata,
    AGENT_REGISTRY,
)

# Core Agent
from .base_agent import BaseAgent
from .standard_agent import (
    StandardAgent,
    RequiredField,
    AgentState,
    AgentTool,
    AgentToolContext,
)

# Tool Decorator
from .tool_decorator import tool

# Message System
from .message import (
    Message,
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
)

# Result
from .result import AgentResult, AgentStatus, ApprovalResult

# Protocols (for type hints and implementation)
from .protocols import (
    LLMClientProtocol,
)

# Tools
from .tools import (
    ToolCall,
    ToolResult,
)

# MCP Integration
from .mcp import (
    MCPClientProtocol,
    MCPClient,
    MCPToolProvider,
    MCPManager,
    MCPServerConfig,
    MCPTool,
    MCPResource,
)

# Config
from .config import (
    AgentRegistry,
)

# Orchestrator
from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    SessionConfig,
    AgentPoolManager,
    ReactLoopConfig,
    ReactLoopResult,
    ToolCallRecord,
    TokenUsage,
)

# Database
from .db import Database, Repository

# Credentials
from .credentials import CredentialStore

# Application Entry Point
from .app import OneValet

# Memory
from .memory import MomexMemory

# Streaming
from .streaming import (
    StreamMode,
    EventType,
    AgentEvent,
    StreamEngine,
)

# Hooks (one-click logging)
from .hooks import (
    logged,
    traced,
    metered,
    observable,
    HookManager,
    configure_hooks,
)

# LLM Clients (built-in, ready to use)
from .llm import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    StreamChunk,
    LLMRegistry,
    LLMProviderConfig,
    LiteLLMClient,
)

__all__ = [
    # Version
    "__version__",
    # Fields
    "InputField",
    "OutputField",
    # Agent Decorator
    "valet",
    "get_agent_metadata",
    "is_valet",
    "AgentMetadata",
    "AGENT_REGISTRY",
    # Core Agent
    "BaseAgent",
    "StandardAgent",
    "RequiredField",
    "AgentState",
    "AgentTool",
    "AgentToolContext",
    # Tool Decorator
    "tool",
    # Message
    "Message",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "VideoBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Result
    "AgentResult",
    "AgentStatus",
    "ApprovalResult",
    # Protocols
    "LLMClientProtocol",
    # Tools
    "ToolCall",
    "ToolResult",
    # MCP
    "MCPClientProtocol",
    "MCPClient",
    "MCPToolProvider",
    "MCPManager",
    "MCPServerConfig",
    "MCPTool",
    "MCPResource",
    # Config
    "AgentRegistry",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "SessionConfig",
    "AgentPoolManager",
    "ReactLoopConfig",
    "ReactLoopResult",
    "ToolCallRecord",
    "TokenUsage",
    # Database
    "Database",
    "Repository",
    # Credentials
    "CredentialStore",
    # Application Entry Point
    "OneValet",
    # Memory
    "MomexMemory",
    # Streaming
    "StreamMode",
    "EventType",
    "AgentEvent",
    "StreamEngine",
    # Hooks (one-click logging)
    "logged",
    "traced",
    "metered",
    "observable",
    "HookManager",
    "configure_hooks",
    # LLM Clients
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    "StreamChunk",
    "LLMRegistry",
    "LLMProviderConfig",
    "LiteLLMClient",
]
