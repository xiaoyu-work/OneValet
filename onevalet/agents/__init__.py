"""
OneValet Agent Decorator - Auto-register agents with @valet decorator

Usage:
    from onevalet import valet, StandardAgent, InputField, OutputField

    @valet(capabilities=["email"])
    class SendEmailAgent(StandardAgent):
        '''Send emails to users'''

        recipient = InputField("Who should I send to?")
        subject = InputField("Subject?", required=False)

        message_id = OutputField(str, "ID of sent message")

        async def on_running(self, msg):
            ...

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
"""

from .decorator import (
    valet,
    get_agent_metadata,
    is_valet,
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
    "valet",
    "get_agent_metadata",
    "is_valet",
    "AgentMetadata",
    "InputSpec",
    "OutputSpec",
    "AGENT_REGISTRY",
    # Discovery
    "AgentDiscovery",
    "discover_agents",
    "discover_agents_from_paths",
]
