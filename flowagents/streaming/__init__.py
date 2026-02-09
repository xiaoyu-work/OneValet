"""
FlowAgent Streaming System - Real-time event streaming

This module provides multi-mode streaming for real-time feedback:

Stream Modes:
- VALUES: Complete state after each update
- UPDATES: Only incremental changes
- MESSAGES: LLM messages (token-by-token)
- EVENTS: All events (state changes, tool calls)

Example usage:
    from flowagents.streaming import StreamMode, AgentEvent

    async for event in agent.stream(msg, mode=StreamMode.EVENTS):
        if event.type == "state_change":
            print(f"State changed to: {event.data['new_status']}")
        elif event.type == "message_chunk":
            print(event.data['chunk'], end='')
"""

from .models import (
    StreamMode,
    EventType,
    AgentEvent,
    StateChangeEvent,
    MessageChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
    ProgressEvent,
    ErrorEvent,
)

from .engine import (
    StreamEngine,
    StreamBuffer,
    EventEmitter,
)

__all__ = [
    # Models
    "StreamMode",
    "EventType",
    "AgentEvent",
    "StateChangeEvent",
    "MessageChunkEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ProgressEvent",
    "ErrorEvent",
    # Engine
    "StreamEngine",
    "StreamBuffer",
    "EventEmitter",
]
