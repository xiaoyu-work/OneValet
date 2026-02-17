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
from .agents import valet

# Core Agent
from .base_agent import BaseAgent
from .standard_agent import StandardAgent
from .models import AgentTool, AgentToolContext

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
)

# Result
from .result import AgentResult, AgentStatus, ApprovalResult

# Orchestrator
from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
)

# Application Entry Point
from .app import OneValet

# Streaming
from .streaming import (
    StreamMode,
    EventType,
    AgentEvent,
)

# LLM Clients (built-in, ready to use)
from .llm import (
    LLMConfig,
    LLMResponse,
    LiteLLMClient,
)

__all__ = [
    "__version__",
    # Fields
    "InputField", "OutputField",
    # Decorators
    "valet", "tool",
    # Core Agent
    "BaseAgent", "StandardAgent",
    "AgentTool", "AgentToolContext",
    # Message
    "Message", "TextBlock", "ImageBlock", "AudioBlock", "VideoBlock",
    "ToolUseBlock", "ToolResultBlock",
    # Result
    "AgentResult", "AgentStatus", "ApprovalResult",
    # Orchestrator
    "OneValet", "Orchestrator", "OrchestratorConfig",
    # LLM
    "LiteLLMClient", "LLMConfig", "LLMResponse",
    # Streaming
    "StreamMode", "EventType", "AgentEvent",
]
