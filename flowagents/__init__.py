"""
FlowAgents - A zero-code AI workflow orchestration framework

FlowAgents provides a simple yet powerful framework for building conversational AI agents
and orchestrating multi-agent workflows with minimal code.

Key Features:
- Decorator-based agent registration (@flowagent)
- InputField/OutputField for clean field definitions
- State machine for conversation flow management
- Custom validators with error messages
- Built-in LLM clients (OpenAI, Anthropic, etc.)
- Built-in streaming support
- One-click logging with @logged decorator

Quick Start (Recommended):
    from flowagents import flowagent, StandardAgent, InputField, OutputField, AgentStatus

    @flowagent(triggers=["send email"])
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
    @flowagent
    class HelloAgent(StandardAgent):
        '''Say hello'''

        name = InputField("What's your name?")

        async def on_running(self, msg):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Hello, {self.name}!"
            )

Built-in LLM Client:
    from flowagents.llm import OpenAIClient, AnthropicClient

    client = OpenAIClient(api_key="sk-xxx")
    response = await client.chat_completion(messages=[...])

    # With streaming
    async for chunk in client.stream_completion(messages=[...]):
        print(chunk.content, end="")

One-click Logging:
    from flowagents.hooks import logged

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
    flowagent,
    get_agent_metadata,
    is_flowagent,
    AgentMetadata,
    AGENT_REGISTRY,
)

# Core Agent
from .base_agent import BaseAgent
from .standard_agent import (
    StandardAgent,
    RequiredField,
    AgentState,
)

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
    ConfigLoaderProtocol,
)

# Tools
from .tools import (
    ToolRegistry,
    ToolExecutor,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolCategory,
    ToolExecutionContext,
    # Tool decorator
    tool,
    get_tool_definition,
    ToolDiscovery,
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

# Config (YAML-based configuration)
from .config import (
    ConfigLoader,
    AgentConfig,
    ToolConfig,
    MCPConfig,
    AgentRegistry,
)

# Orchestrator
from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    SessionConfig,
    RoutingAction,
    RoutingDecision,
    AgentPoolManager,
    MessageRouter,
)

# Formatter (Multi-model support)
from .formatter import (
    Provider,
    FormatterConfig,
    FormatterBase,
    get_formatter,
    OpenAIFormatter,
    AnthropicFormatter,
)

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
    OpenAIClient,
    AnthropicClient,
    AzureOpenAIClient,
    DashScopeClient,
    GeminiClient,
    OllamaClient,
)

__all__ = [
    # Version
    "__version__",
    # Fields
    "InputField",
    "OutputField",
    # Agent Decorator
    "flowagent",
    "get_agent_metadata",
    "is_flowagent",
    "AgentMetadata",
    "AGENT_REGISTRY",
    # Core Agent
    "BaseAgent",
    "StandardAgent",
    "RequiredField",
    "AgentState",
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
    "ConfigLoaderProtocol",
    # Tools
    "ToolRegistry",
    "ToolExecutor",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolCategory",
    "ToolExecutionContext",
    "tool",
    "get_tool_definition",
    "ToolDiscovery",
    # MCP
    "MCPClientProtocol",
    "MCPClient",
    "MCPToolProvider",
    "MCPManager",
    "MCPServerConfig",
    "MCPTool",
    "MCPResource",
    # Config
    "ConfigLoader",
    "AgentConfig",
    "ToolConfig",
    "MCPConfig",
    "AgentRegistry",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "SessionConfig",
    "RoutingAction",
    "RoutingDecision",
    "AgentPoolManager",
    "MessageRouter",
    # Formatter
    "Provider",
    "FormatterConfig",
    "FormatterBase",
    "get_formatter",
    "OpenAIFormatter",
    "AnthropicFormatter",
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
    "OpenAIClient",
    "AnthropicClient",
    "AzureOpenAIClient",
    "DashScopeClient",
    "GeminiClient",
    "OllamaClient",
]
