"""
Koa - A zero-code AI workflow orchestration framework

Koa provides a simple yet powerful framework for building conversational AI agents
and orchestrating multi-agent workflows with minimal code.

Key Features:
- Decorator-based agent registration (@valet)
- InputField/OutputField for clean field definitions
- State machine for conversation flow management
- Custom validators with error messages
- Built-in LLM clients (OpenAI, Anthropic, etc.)
- Built-in streaming support

Quick Start (Recommended):
    from koa import valet, StandardAgent, InputField, OutputField, AgentStatus

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
    from koa.llm import LiteLLMClient

    client = LiteLLMClient(model="gpt-4o", provider_name="openai", api_key="sk-xxx")
    response = await client.chat_completion(messages=[...])

Streaming:
    agent = MyAgent(tenant_id="123")
    async for event in agent.stream(msg):
        if event.type == EventType.MESSAGE_CHUNK:
            print(event.data["chunk"], end="")
"""

__version__ = "0.1.1"

# Fields (InputField/OutputField descriptors)
# Agent Decorator
from .agents import valet

# Application Entry Point
from .app import Koa

# Core Agent
from .base_agent import BaseAgent

# Errors
from .errors import E, KoaError
from .fields import InputField, OutputField

# LLM Clients (built-in, ready to use)
from .llm import (
    LiteLLMClient,
    LLMConfig,
    LLMResponse,
)

# Message System
from .message import (
    AudioBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)
from .models import AgentTool, AgentToolContext

# Orchestrator
from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
)

# Result
from .result import AgentResult, AgentStatus, ApprovalResult
from .standard_agent import StandardAgent

# Streaming
from .streaming import (
    AgentEvent,
    EventType,
    StreamMode,
)

# Tool Decorator
from .tool_decorator import tool

__all__ = [
    "__version__",
    # Fields
    "InputField",
    "OutputField",
    # Decorators
    "valet",
    "tool",
    # Core Agent
    "BaseAgent",
    "StandardAgent",
    "AgentTool",
    "AgentToolContext",
    # Message
    "Message",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "VideoBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    # Result
    "AgentResult",
    "AgentStatus",
    "ApprovalResult",
    # Orchestrator
    "Koa",
    "Orchestrator",
    "OrchestratorConfig",
    # LLM
    "LiteLLMClient",
    "LLMConfig",
    "LLMResponse",
    # Streaming
    "StreamMode",
    "EventType",
    "AgentEvent",
    # Errors
    "KoaError",
    "E",
]
