"""OneValet Trigger Models — data structures for the proactive trigger system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    """Trigger task lifecycle states."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    COMPLETED = "completed"
    PENDING_APPROVAL = "pending_approval"
    EXPIRED = "expired"


class TriggerType(str, Enum):
    """Types of triggers."""
    SCHEDULE = "schedule"
    EVENT = "event"
    CONDITION = "condition"


@dataclass
class TriggerConfig:
    """Configuration for a trigger."""
    type: TriggerType
    params: Dict[str, Any] = field(default_factory=dict)
    # Schedule: {"cron": "0 8 * * *"} or {"interval_minutes": 5} or {"run_at": "2024-01-01T08:00:00"}
    # Event: {"source": "email", "event_type": "new_email", "filters": {"from": "amazon.com"}}
    # Condition: {"expression": "flight_price < 500", "poll_interval_minutes": 30}


@dataclass
class ActionConfig:
    """Configuration for what to do when a trigger fires."""
    executor: str = "orchestrator"  # executor name (default: OrchestratorExecutor)
    instruction: str = ""  # instruction for the LLM (when using orchestrator executor)
    config: Dict[str, Any] = field(default_factory=dict)  # custom executor config


@dataclass
class Task:
    """A trigger task — combines trigger condition with action."""
    id: str
    user_id: str  # tenant_id
    name: str = ""
    description: str = ""
    trigger: TriggerConfig = field(default_factory=lambda: TriggerConfig(type=TriggerType.SCHEDULE))
    action: ActionConfig = field(default_factory=ActionConfig)
    status: TaskStatus = TaskStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None  # None = unlimited
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "trigger": {"type": self.trigger.type.value, "params": self.trigger.params},
            "action": {
                "executor": self.action.executor,
                "instruction": self.action.instruction,
                "config": self.action.config,
            },
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        trigger_data = data.get("trigger", {})
        action_data = data.get("action", {})
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            trigger=TriggerConfig(
                type=TriggerType(trigger_data.get("type", "schedule")),
                params=trigger_data.get("params", {}),
            ),
            action=ActionConfig(
                executor=action_data.get("executor", "orchestrator"),
                instruction=action_data.get("instruction", ""),
                config=action_data.get("config", {}),
            ),
            status=TaskStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            last_run_at=datetime.fromisoformat(data["last_run_at"]) if data.get("last_run_at") else None,
            next_run_at=datetime.fromisoformat(data["next_run_at"]) if data.get("next_run_at") else None,
            run_count=data.get("run_count", 0),
            max_runs=data.get("max_runs"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TriggerContext:
    """Context passed to executors when a trigger fires."""
    task: Task
    trigger_type: str  # "schedule", "event", "condition"
    fired_at: datetime = field(default_factory=datetime.now)
    event_data: Optional[Dict[str, Any]] = None  # for event triggers
    condition_result: Optional[Any] = None  # for condition triggers


@dataclass
class ActionResult:
    """Result from executing a triggered action."""
    success: bool = True
    output: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
