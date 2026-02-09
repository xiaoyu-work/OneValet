"""
OneValet Result - Standardized agent execution results

Provides a unified result structure for all agent operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class AgentStatus(str, Enum):
    """Standardized agent status states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

    @classmethod
    def terminal_states(cls) -> frozenset:
        """States that indicate agent execution is finished."""
        return frozenset({cls.COMPLETED, cls.CANCELLED, cls.ERROR})


class ApprovalResult(str, Enum):
    """Result of parsing user's approval response."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFY = "modify"


@dataclass
class AgentResult:
    """
    Standardized result from agent execution

    This is the primary return type for all agent operations,
    providing a consistent interface for handling agent responses.

    Attributes:
        agent_type: Class name of the agent (e.g., "SendEmailAgent")
        status: Current agent status
        data: Collected field data
        raw_message: The actual response message
        agent_id: Unique identifier for this agent instance

    Example:
        result = AgentResult(
            agent_type="SendEmailAgent",
            status=AgentStatus.WAITING_FOR_APPROVAL,
            data={"recipient": "john@example.com", "subject": "Hello"},
            raw_message="Ready to send email. Confirm?",
            agent_id="SendEmailAgent_abc123"
        )
    """
    agent_type: str
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    raw_message: str = ""
    agent_id: str = ""

    # Optional fields for specific scenarios
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    missing_field: Optional[str] = None
    missing_fields: Optional[List[str]] = None
    collected_fields: Optional[Dict[str, Any]] = None
    next_prompt: Optional[str] = None
    preview: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_completed(self) -> bool:
        """Check if agent has completed successfully"""
        return self.status == AgentStatus.COMPLETED

    def is_waiting(self) -> bool:
        """Check if agent is waiting for user input or approval"""
        return self.status in (AgentStatus.WAITING_FOR_INPUT, AgentStatus.WAITING_FOR_APPROVAL)

    def is_error(self) -> bool:
        """Check if agent encountered an error"""
        return self.status == AgentStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "agent_type": self.agent_type,
            "status": self.status.value,
            "data": self.data,
            "raw_message": self.raw_message,
            "agent_id": self.agent_id,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "missing_field": self.missing_field,
            "missing_fields": self.missing_fields,
            "collected_fields": self.collected_fields,
            "next_prompt": self.next_prompt,
            "preview": self.preview,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        """Create AgentResult from dictionary"""
        status = data.get("status")
        if isinstance(status, str):
            status = AgentStatus(status)

        return cls(
            agent_type=data.get("agent_type", ""),
            status=status,
            data=data.get("data", {}),
            raw_message=data.get("raw_message", ""),
            agent_id=data.get("agent_id", ""),
            error_message=data.get("error_message"),
            error_type=data.get("error_type"),
            missing_field=data.get("missing_field"),
            missing_fields=data.get("missing_fields"),
            collected_fields=data.get("collected_fields"),
            next_prompt=data.get("next_prompt"),
            preview=data.get("preview"),
            metadata=data.get("metadata", {}),
        )
