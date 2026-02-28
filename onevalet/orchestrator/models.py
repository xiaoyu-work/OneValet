"""
OneValet Orchestrator Models - Data structures for orchestration

This module defines:
- RoutingAction: What action to take for a message
- RoutingDecision: The complete routing decision
- OrchestratorConfig: Configuration for the orchestrator
- SessionConfig: Configuration for session management
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum

# Attribute name for marking callback handlers
CALLBACK_HANDLER_ATTR = '_callback_handler_name'


def callback_handler(name: str):
    """
    Decorator to register an async method as a callback handler.

    Handlers are collected at class definition time via __init_subclass__.
    The handler must be an async method.

    Usage:
        class MyOrchestrator(Orchestrator):
            @callback_handler("get_cache")
            async def get_cache(self, callback: AgentCallback) -> Any:
                return self.cache.get(callback.data["key"])

            @callback_handler("send_sms")
            async def send_sms(self, callback: AgentCallback) -> None:
                await self.sms.send(callback.data["message"])

    Then agents can call:
        result = await self.callback("get_cache", data={"key": "foo"})
        await self.callback("send_sms", data={"message": "Hello"})

    Args:
        name: The callback name that agents will use to invoke this handler
    """
    def decorator(func: Callable) -> Callable:
        setattr(func, CALLBACK_HANDLER_ATTR, name)
        return func
    return decorator


class RoutingReason(str, Enum):
    """Reason codes for routing decisions."""
    ACTIVE_AGENT_FOUND = "active_agent_found"
    LLM_ROUTING = "llm_routing"
    DEFAULT_FALLBACK = "default_fallback"
    NO_ROUTER = "no_router"


class RoutingAction(str, Enum):
    """
    Actions the orchestrator can take for routing a message.

    - ROUTE_TO_EXISTING: Route to an existing active agent
    - CREATE_NEW: Create a new agent instance
    - EXECUTE_WORKFLOW: Execute a workflow
    - ROUTE_TO_DEFAULT: Route to DefaultAgent (when no other agent matches)
    - DELEGATE: Delegate to another orchestrator
    """
    ROUTE_TO_EXISTING = "route_to_existing"
    CREATE_NEW = "create_new"
    EXECUTE_WORKFLOW = "execute_workflow"
    ROUTE_TO_DEFAULT = "route_to_default"
    DELEGATE = "delegate"


@dataclass
class RoutingDecision:
    """
    Complete routing decision made by the orchestrator.

    Attributes:
        action: The routing action to take
        agent_id: ID of existing agent (for ROUTE_TO_EXISTING)
        agent_type: Type of agent to create (for CREATE_NEW, ROUTE_TO_DEFAULT)
        workflow_id: ID of workflow to execute (for EXECUTE_WORKFLOW)
        context_hints: Extracted context from the message
        confidence: How confident the routing decision is (0-1)
        reason: Reason code for the decision (RoutingReason enum)
        delegate_to: Orchestrator to delegate to (for DELEGATE)
    """
    action: RoutingAction
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    workflow_id: Optional[str] = None
    context_hints: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    reason: Optional[RoutingReason] = None
    delegate_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action.value,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "workflow_id": self.workflow_id,
            "context_hints": self.context_hints,
            "confidence": self.confidence,
            "reason": self.reason.value if self.reason else None,
            "delegate_to": self.delegate_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """Create from dictionary"""
        reason_value = data.get("reason")
        reason = RoutingReason(reason_value) if reason_value else None
        return cls(
            action=RoutingAction(data["action"]),
            agent_id=data.get("agent_id"),
            agent_type=data.get("agent_type"),
            workflow_id=data.get("workflow_id"),
            context_hints=data.get("context_hints"),
            confidence=data.get("confidence", 1.0),
            reason=reason,
            delegate_to=data.get("delegate_to"),
        )


@dataclass
class SessionConfig:
    """
    Configuration for session management.

    Attributes:
        enabled: Whether session persistence is enabled
        session_ttl_seconds: TTL for sessions (default: 24 hours)
        auto_backup_interval_seconds: Background backup interval (default: 60s)
        auto_restore_on_start: Whether to restore sessions on server start
        lazy_restore: Whether to restore sessions on first request
        waiting_timeout_seconds: Timeout for WAITING agents before cleanup (default: 5 min)
    """
    enabled: bool = True
    session_ttl_seconds: int = 86400  # 24 hours
    auto_backup_interval_seconds: int = 60
    auto_restore_on_start: bool = True
    lazy_restore: bool = True
    waiting_timeout_seconds: int = 300  # 5 minutes default for WAITING agents

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConfig":
        """Create from dictionary"""
        return cls(
            enabled=data.get("enabled", True),
            session_ttl_seconds=data.get("session_ttl_seconds", 86400),
            auto_backup_interval_seconds=data.get("auto_backup_interval_seconds", 60),
            auto_restore_on_start=data.get("auto_restore_on_start", True),
            lazy_restore=data.get("lazy_restore", True),
            waiting_timeout_seconds=data.get("waiting_timeout_seconds", 300),
        )


@dataclass
class OrchestratorConfig:
    """
    Configuration for the orchestrator.

    Attributes:
        config_dir: Path to YAML config directory
        session: Session management configuration
        default_timeout_seconds: Default timeout for agent execution
        max_agents_per_user: Maximum concurrent agents per user
        enable_workflows: Whether workflow execution is enabled
        enable_streaming: Whether streaming is enabled
        default_agent_type: Agent type to use when no other agent matches
    """
    config_dir: str = ""
    session: SessionConfig = field(default_factory=SessionConfig)
    default_timeout_seconds: int = 300
    max_agents_per_user: int = 10
    enable_workflows: bool = True  # TODO: workflow execution not yet implemented
    enable_streaming: bool = True
    default_agent_type: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        """Create from dictionary"""
        session_data = data.get("session", {})
        session = SessionConfig.from_dict(session_data) if session_data else SessionConfig()

        return cls(
            config_dir=data.get("config_dir", ""),
            session=session,
            default_timeout_seconds=data.get("default_timeout_seconds", 300),
            max_agents_per_user=data.get("max_agents_per_user", 10),
            enable_workflows=data.get("enable_workflows", True),
            enable_streaming=data.get("enable_streaming", True),
            default_agent_type=data.get("default_agent_type", ""),
        )


@dataclass
class AgentPoolEntry:
    """
    Entry in the agent pool representing an active agent.

    Attributes:
        agent_id: Unique identifier for the agent instance
        agent_type: Type/class of the agent
        tenant_id: Tenant who owns this agent (default: "default")
        status: Current agent status
        created_at: When the agent was created
        last_activity: When the agent was last active
        collected_fields: Fields collected so far
        execution_state: Runtime execution state
        context: User context
        checkpoint_id: Last checkpoint ID (if checkpointing enabled)
        schema_version: Schema version derived from InputField definitions (default: 0)
    """
    agent_id: str
    agent_type: str
    tenant_id: str = ""
    status: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    collected_fields: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    checkpoint_id: Optional[str] = None
    schema_version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "collected_fields": self.collected_fields,
            "execution_state": self.execution_state,
            "context": self.context,
            "checkpoint_id": self.checkpoint_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPoolEntry":
        """Create from dictionary"""
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            tenant_id=data.get("tenant_id", ""),
            status=data.get("status", ""),
            created_at=datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            last_activity=datetime.fromisoformat(data["last_activity"])
                if isinstance(data.get("last_activity"), str) else data.get("last_activity", datetime.now()),
            collected_fields=data.get("collected_fields", {}),
            execution_state=data.get("execution_state", {}),
            context=data.get("context", {}),
            checkpoint_id=data.get("checkpoint_id"),
            schema_version=data.get("schema_version", 0),
        )


@dataclass
class AgentCallback:
    """
    Callback request from agent to orchestrator.

    A minimal structure for agent-to-orchestrator communication.
    All parameters are passed via the data dict.

    Attributes:
        event: Handler name (registered via @callback_handler)
        tenant_id: Tenant ID
        data: All callback parameters
        timestamp: When the callback occurred

    Example:
        # Agent requests cached data
        callback = AgentCallback(
            event="get_cache",
            tenant_id=tenant_id,
            data={"group": "email"}
        )

        # Agent sends notification
        callback = AgentCallback(
            event="send_sms",
            tenant_id=tenant_id,
            data={"message": message, "phone": phone}
        )
    """
    event: str
    tenant_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event": self.event,
            "tenant_id": self.tenant_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentCallback":
        """Create from dictionary"""
        timestamp = d.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            event=d.get("event", ""),
            tenant_id=d.get("tenant_id", ""),
            data=d.get("data", {}),
            timestamp=timestamp,
        )
