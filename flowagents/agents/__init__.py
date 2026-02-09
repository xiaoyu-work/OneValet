"""
FlowAgent Agent Decorator - Auto-register agents with @flowagent decorator

Usage:
    from flowagents import flowagent, StandardAgent, InputField, OutputField

    @flowagent(triggers=["send email"], llm="gpt4")
    class SendEmailAgent(StandardAgent):
        '''Send emails to users'''

        recipient = InputField("Who should I send to?")
        subject = InputField("Subject?", required=False)

        message_id = OutputField(str, "ID of sent message")

        async def on_running(self, msg):
            ...

    # Minimal version - no triggers, default LLM
    @flowagent
    class HelloAgent(StandardAgent):
        '''Say hello'''

        name = InputField("What's your name?")

        async def on_running(self, msg):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Hello, {self.name}!"
            )
"""

from .decorator import (
    flowagent,
    get_agent_metadata,
    is_flowagent,
    AgentMetadata,
    InputSpec,
    OutputSpec,
    AGENT_REGISTRY,
)
from .discovery import (
    AgentDiscovery,
    discover_agents,
    discover_agents_from_paths,
)

__all__ = [
    # Decorator
    "flowagent",
    "get_agent_metadata",
    "is_flowagent",
    "AgentMetadata",
    "InputSpec",
    "OutputSpec",
    "AGENT_REGISTRY",
    # Discovery
    "AgentDiscovery",
    "discover_agents",
    "discover_agents_from_paths",
]
