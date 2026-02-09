"""
OneValet MsgHub Models - Data structures for multi-agent message sharing

This module defines:
- Message: Shared message in MsgHub
- ParticipantInfo: Information about a participant agent
- MsgHubConfig: Configuration for MsgHub
- MsgHubState: Current state of MsgHub
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Union
from enum import Enum


class MessageRole(str, Enum):
    """Role of message sender"""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Type of message"""
    TEXT = "text"
    DATA = "data"
    ACTION = "action"
    RESULT = "result"


@dataclass
class Message:
    """A message in the shared conversation"""
    id: str
    role: MessageRole
    content: str

    # Sender info
    sender_id: str
    sender_type: Optional[str] = None

    # Message type
    message_type: MessageType = MessageType.TEXT

    # Structured data (optional)
    data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Reply reference
    reply_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "sender_id": self.sender_id,
            "sender_type": self.sender_type,
            "message_type": self.message_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Create from dictionary"""
        return cls(
            id=d["id"],
            role=MessageRole(d["role"]),
            content=d["content"],
            sender_id=d["sender_id"],
            sender_type=d.get("sender_type"),
            message_type=MessageType(d.get("message_type", "text")),
            data=d.get("data", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]) if isinstance(d.get("timestamp"), str) else d.get("timestamp", datetime.now()),
            metadata=d.get("metadata", {}),
            reply_to=d.get("reply_to"),
        )


@dataclass
class ParticipantInfo:
    """Information about a participant agent"""
    agent_id: str
    agent_type: str

    # Participation status
    joined_at: datetime = field(default_factory=datetime.now)
    joined_at_message_count: int = 0  # Message count when joined (for visibility)
    is_active: bool = True

    # Message tracking
    last_seen_message_id: Optional[str] = None
    messages_sent: int = 0

    # Visibility rules
    can_see_all: bool = True  # If False, only sees messages after join
    visible_roles: List[MessageRole] = field(default_factory=lambda: list(MessageRole))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "joined_at": self.joined_at.isoformat(),
            "is_active": self.is_active,
            "last_seen_message_id": self.last_seen_message_id,
            "messages_sent": self.messages_sent,
            "can_see_all": self.can_see_all,
            "visible_roles": [r.value for r in self.visible_roles],
        }


class VisibilityMode(str, Enum):
    """How messages are shared among participants"""
    ALL = "all"          # All participants see all messages
    SEQUENTIAL = "sequential"  # Each agent sees previous agents' messages
    SELECTIVE = "selective"    # Custom visibility rules


@dataclass
class MsgHubConfig:
    """Configuration for MsgHub"""
    hub_id: Optional[str] = None

    # Visibility settings
    visibility_mode: VisibilityMode = VisibilityMode.ALL

    # Message limits
    max_messages: int = 1000
    max_message_age_seconds: Optional[int] = None  # Auto-cleanup old messages

    # Shared context
    shared_context_keys: List[str] = field(default_factory=list)

    # Behavior
    auto_broadcast: bool = True  # Auto-broadcast agent replies
    preserve_order: bool = True  # Maintain message order

    # Persistence
    persist_messages: bool = False
    storage_backend: Optional[str] = None


@dataclass
class SharedContext:
    """Shared context data accessible by all participants"""
    data: Dict[str, Any] = field(default_factory=dict)

    # Track updates
    last_updated: datetime = field(default_factory=datetime.now)
    updated_by: Optional[str] = None

    def set(self, key: str, value: Any, updater_id: Optional[str] = None) -> None:
        """Set a context value"""
        self.data[key] = value
        self.last_updated = datetime.now()
        self.updated_by = updater_id

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value"""
        return self.data.get(key, default)

    def update(self, updates: Dict[str, Any], updater_id: Optional[str] = None) -> None:
        """Update multiple values"""
        self.data.update(updates)
        self.last_updated = datetime.now()
        self.updated_by = updater_id

    def delete(self, key: str) -> bool:
        """Delete a context value"""
        if key in self.data:
            del self.data[key]
            self.last_updated = datetime.now()
            return True
        return False


@dataclass
class MsgHubState:
    """Current state of MsgHub"""
    hub_id: str

    # Messages
    messages: List[Message] = field(default_factory=list)

    # Participants
    participants: Dict[str, ParticipantInfo] = field(default_factory=dict)

    # Shared context
    context: SharedContext = field(default_factory=SharedContext)

    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

    @property
    def message_count(self) -> int:
        """Number of messages"""
        return len(self.messages)

    @property
    def participant_count(self) -> int:
        """Number of active participants"""
        return sum(1 for p in self.participants.values() if p.is_active)

    def get_messages_for_participant(
        self,
        participant_id: str,
        config: MsgHubConfig
    ) -> List[Message]:
        """Get messages visible to a participant"""
        participant = self.participants.get(participant_id)
        if not participant:
            return []

        if config.visibility_mode == VisibilityMode.ALL:
            if participant.can_see_all:
                return list(self.messages)
            else:
                # Only messages after join (use index, not timestamp)
                return list(self.messages)[participant.joined_at_message_count:]

        elif config.visibility_mode == VisibilityMode.SEQUENTIAL:
            # See all messages from agents who executed before
            visible = []
            participant_order = list(self.participants.keys())
            my_index = participant_order.index(participant_id) if participant_id in participant_order else -1

            for msg in self.messages:
                if msg.role == MessageRole.USER:
                    visible.append(msg)
                elif msg.role == MessageRole.SYSTEM:
                    visible.append(msg)
                elif msg.sender_id in participant_order:
                    sender_index = participant_order.index(msg.sender_id)
                    if sender_index < my_index:
                        visible.append(msg)
            return visible

        else:  # SELECTIVE
            # Filter by visibility rules
            visible = []
            for msg in self.messages:
                if msg.role in participant.visible_roles:
                    visible.append(msg)
            return visible

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "hub_id": self.hub_id,
            "messages": [m.to_dict() for m in self.messages],
            "participants": {k: v.to_dict() for k, v in self.participants.items()},
            "context": self.context.data,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }


@dataclass
class HubExecutionResult:
    """Result from executing agents in MsgHub"""
    hub_id: str
    status: Literal["completed", "partial", "failed"]

    # Execution results per agent
    agent_results: List[Dict[str, Any]] = field(default_factory=list)

    # Final messages
    final_messages: List[Message] = field(default_factory=list)

    # Final shared context
    final_context: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    total_agents: int = 0
    completed_agents: int = 0
    failed_agents: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_agents == 0:
            return 0.0
        return self.completed_agents / self.total_agents
