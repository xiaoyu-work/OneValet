"""
FlowAgent MsgHub - Multi-agent message sharing

This module provides:
- MsgHub: Share conversation context between agents
- Message: Shared message format
- Visibility modes for controlling message access

Example:
    from flowagents.msghub import MsgHub, MsgHubConfig, VisibilityMode

    # Sequential agents share context
    async with MsgHub(participants=[researcher, writer, reviewer]) as hub:
        result1 = await hub.execute(researcher, "Research Python frameworks")
        result2 = await hub.execute(writer, "Write a comparison")
        result3 = await hub.execute(reviewer, "Review the document")
"""

from .models import (
    # Message types
    Message,
    MessageRole,
    MessageType,
    # Participant
    ParticipantInfo,
    # Configuration
    MsgHubConfig,
    VisibilityMode,
    SharedContext,
    # State and results
    MsgHubState,
    HubExecutionResult,
)

from .hub import (
    MsgHub,
    MsgHubError,
    msghub,
)

__all__ = [
    # Models
    "Message",
    "MessageRole",
    "MessageType",
    "ParticipantInfo",
    "MsgHubConfig",
    "VisibilityMode",
    "SharedContext",
    "MsgHubState",
    "HubExecutionResult",
    # Hub
    "MsgHub",
    "MsgHubError",
    "msghub",
]
