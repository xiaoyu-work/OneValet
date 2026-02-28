"""
OneValet Streaming Models - Data structures for streaming events

This module defines:
- Stream modes (VALUES, UPDATES, MESSAGES, EVENTS)
- Event types and event data structures
- Helper functions for creating events
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Union
from enum import Enum


class StreamMode(str, Enum):
    """
    Streaming modes for agent execution.

    - VALUES: Complete state after each update (snapshot mode)
    - UPDATES: Only incremental changes (delta mode)
    - MESSAGES: LLM messages token-by-token (for real-time chat)
    - EVENTS: All events including state changes, tool calls, etc.
    """
    VALUES = "values"       # Complete state after each update
    UPDATES = "updates"     # Only incremental changes
    MESSAGES = "messages"   # LLM messages (token-by-token)
    EVENTS = "events"       # All events (state changes, tool calls)


class EventType(str, Enum):
    """Types of events that can be streamed"""
    # State events
    STATE_CHANGE = "state_change"
    FIELD_COLLECTED = "field_collected"
    FIELD_VALIDATED = "field_validated"

    # Message events
    MESSAGE_START = "message_start"
    MESSAGE_CHUNK = "message_chunk"
    MESSAGE_END = "message_end"

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"

    # Progress events
    PROGRESS_UPDATE = "progress_update"

    # Execution events
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"

    # Error events
    ERROR = "error"
    WARNING = "warning"

    # Planning events
    PLAN_GENERATED = "plan_generated"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"

    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"


@dataclass
class AgentEvent:
    """
    Base event structure for streaming.

    All events have:
    - type: The type of event
    - data: Event-specific data
    - timestamp: When the event occurred
    - agent_id: Which agent generated the event
    - sequence: Sequence number for ordering
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentEvent":
        """Create event from dictionary"""
        return cls(
            type=EventType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data.get("agent_id"),
            agent_type=data.get("agent_type"),
            sequence=data.get("sequence", 0),
        )


@dataclass
class StateChangeEvent(AgentEvent):
    """Event for agent state changes"""

    def __init__(
        self,
        old_status: str,
        new_status: str,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            type=EventType.STATE_CHANGE,
            data={
                "old_status": old_status,
                "new_status": new_status,
            },
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )


@dataclass
class MessageChunkEvent(AgentEvent):
    """Event for streaming message chunks (token-by-token)"""

    def __init__(
        self,
        chunk: str,
        message_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            type=EventType.MESSAGE_CHUNK,
            data={
                "chunk": chunk,
                "message_id": message_id,
            },
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )


@dataclass
class ToolCallEvent(AgentEvent):
    """Event for tool call start"""

    def __init__(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            type=EventType.TOOL_CALL_START,
            data={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "call_id": call_id,
            },
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )


@dataclass
class ToolResultEvent(AgentEvent):
    """Event for tool execution result"""

    def __init__(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            type=EventType.TOOL_RESULT,
            data={
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "error": error,
                "call_id": call_id,
            },
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )


@dataclass
class ProgressEvent(AgentEvent):
    """Event for progress updates"""

    def __init__(
        self,
        current: int,
        total: int,
        message: Optional[str] = None,
        percentage: Optional[float] = None,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        if percentage is None and total > 0:
            percentage = (current / total) * 100

        super().__init__(
            type=EventType.PROGRESS_UPDATE,
            data={
                "current": current,
                "total": total,
                "percentage": percentage,
                "message": message,
            },
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )


@dataclass
class ErrorEvent(AgentEvent):
    """Event for errors"""

    def __init__(
        self,
        error: str,
        error_type: Optional[str] = None,
        recoverable: bool = False,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            type=EventType.ERROR,
            data={
                "error": error,
                "error_type": error_type,
                "recoverable": recoverable,
            },
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )


# Factory functions for creating events
def create_state_change_event(
    old_status: str,
    new_status: str,
    agent_id: str = None,
    agent_type: str = None
) -> AgentEvent:
    """Create a state change event"""
    return AgentEvent(
        type=EventType.STATE_CHANGE,
        data={"old_status": old_status, "new_status": new_status},
        agent_id=agent_id,
        agent_type=agent_type,
    )


def create_message_chunk_event(
    chunk: str,
    message_id: str = None,
    agent_id: str = None,
    agent_type: str = None
) -> AgentEvent:
    """Create a message chunk event"""
    return AgentEvent(
        type=EventType.MESSAGE_CHUNK,
        data={"chunk": chunk, "message_id": message_id},
        agent_id=agent_id,
        agent_type=agent_type,
    )


def create_tool_call_event(
    tool_name: str,
    tool_input: Dict[str, Any],
    call_id: str = None,
    agent_id: str = None,
    agent_type: str = None
) -> AgentEvent:
    """Create a tool call event"""
    return AgentEvent(
        type=EventType.TOOL_CALL_START,
        data={"tool_name": tool_name, "tool_input": tool_input, "call_id": call_id},
        agent_id=agent_id,
        agent_type=agent_type,
    )


def create_progress_event(
    current: int,
    total: int,
    message: str = None,
    agent_id: str = None,
    agent_type: str = None
) -> AgentEvent:
    """Create a progress update event"""
    percentage = (current / total) * 100 if total > 0 else 0
    return AgentEvent(
        type=EventType.PROGRESS_UPDATE,
        data={
            "current": current,
            "total": total,
            "percentage": percentage,
            "message": message,
        },
        agent_id=agent_id,
        agent_type=agent_type,
    )


def create_error_event(
    error: str,
    error_type: str = None,
    recoverable: bool = False,
    agent_id: str = None,
    agent_type: str = None
) -> AgentEvent:
    """Create an error event"""
    return AgentEvent(
        type=EventType.ERROR,
        data={
            "error": error,
            "error_type": error_type,
            "recoverable": recoverable,
        },
        agent_id=agent_id,
        agent_type=agent_type,
    )
