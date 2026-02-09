"""
FlowAgent Hook System Models - Data structures for built-in observability

This module defines:
- HookType: Types of hooks (logging, metrics, tracing, etc.)
- HookConfig: Configuration for hooks
- HookContext: Context passed to hook handlers
- HookResult: Result from hook execution
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal, Callable, Union
from enum import Enum


class HookType(str, Enum):
    """Types of built-in hooks"""
    LOGGING = "logging"
    METRICS = "metrics"
    TRACING = "tracing"
    LANGFUSE = "langfuse"
    RATE_LIMITING = "rate_limiting"
    CUSTOM = "custom"


class HookPhase(str, Enum):
    """When hooks are executed"""
    PRE_EXECUTE = "pre_execute"      # Before agent execution
    POST_EXECUTE = "post_execute"    # After agent execution
    ON_ERROR = "on_error"            # On error
    ON_STATE_CHANGE = "on_state_change"  # On status change


@dataclass
class HookConfig:
    """Configuration for a hook"""
    hook_type: HookType
    enabled: bool = True

    # Common settings
    log_level: str = "INFO"
    include_inputs: bool = True
    include_outputs: bool = True

    # Type-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)

    # Filters
    agent_types: Optional[List[str]] = None  # Apply to specific agents
    exclude_agent_types: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], hook_type: HookType) -> "HookConfig":
        """Create from dictionary"""
        if isinstance(d, bool):
            return cls(hook_type=hook_type, enabled=d)

        return cls(
            hook_type=hook_type,
            enabled=d.get("enabled", True),
            log_level=d.get("log_level", "INFO"),
            include_inputs=d.get("include_inputs", True),
            include_outputs=d.get("include_outputs", True),
            settings=d.get("settings", {}),
            agent_types=d.get("agent_types"),
            exclude_agent_types=d.get("exclude_agent_types"),
        )


@dataclass
class HookContext:
    """Context passed to hook handlers"""
    # Agent info
    agent_id: str
    agent_type: str
    user_id: str

    # Execution info
    phase: HookPhase
    status: Optional[str] = None
    method_name: Optional[str] = None

    # Input/Output
    input_message: Optional[Any] = None
    output_result: Optional[Any] = None

    # Error info
    error: Optional[Exception] = None
    error_type: Optional[str] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # State
    collected_fields: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "user_id": self.user_id,
            "phase": self.phase.value,
            "status": self.status,
            "method_name": self.method_name,
            "duration_ms": self.duration_ms,
            "error": str(self.error) if self.error else None,
            "error_type": self.error_type,
            "metadata": self.metadata,
        }


@dataclass
class HookResult:
    """Result from hook execution"""
    hook_type: HookType
    success: bool = True
    error: Optional[str] = None

    # Optional data returned by hook
    data: Dict[str, Any] = field(default_factory=dict)

    # For rate limiting
    should_proceed: bool = True
    retry_after: Optional[float] = None  # Seconds to wait


@dataclass
class MetricsData:
    """Metrics data collected by metrics hook"""
    agent_type: str
    user_id: str

    # Counters
    invocation_count: int = 0
    success_count: int = 0
    error_count: int = 0

    # Timing
    total_duration_ms: float = 0
    avg_duration_ms: float = 0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0

    # Token usage (if available)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Cost (if available)
    total_cost: float = 0

    def record_invocation(
        self,
        duration_ms: float,
        success: bool,
        tokens: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None
    ) -> None:
        """Record an invocation"""
        self.invocation_count += 1

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.invocation_count
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)

        if tokens:
            self.total_tokens += tokens.get("total", 0)
            self.prompt_tokens += tokens.get("prompt", 0)
            self.completion_tokens += tokens.get("completion", 0)

        if cost:
            self.total_cost += cost


@dataclass
class TracingSpan:
    """A tracing span for distributed tracing"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None

    name: str = ""
    service_name: str = "flowagent"

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    # Status
    status: Literal["ok", "error"] = "ok"
    error_message: Optional[str] = None

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute"""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span"""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })

    def end(self, status: Literal["ok", "error"] = "ok", error_message: Optional[str] = None) -> None:
        """End the span"""
        self.end_time = datetime.now()
        self.status = status
        self.error_message = error_message

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    # Per-user limits
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000

    # Per-agent limits
    max_agent_requests_per_minute: Optional[int] = None

    # Token limits
    max_tokens_per_minute: Optional[int] = None
    max_tokens_per_day: Optional[int] = None

    # Custom limits per agent type
    agent_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class RateLimitState:
    """Current rate limit state for a user"""
    user_id: str

    # Request counts
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    requests_this_day: int = 0

    # Token counts
    tokens_this_minute: int = 0
    tokens_this_day: int = 0

    # Windows
    minute_window_start: datetime = field(default_factory=datetime.now)
    hour_window_start: datetime = field(default_factory=datetime.now)
    day_window_start: datetime = field(default_factory=datetime.now)

    def check_and_update(
        self,
        config: RateLimitConfig,
        tokens: int = 0
    ) -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed and update counters.

        Returns:
            (allowed, retry_after_seconds)
        """
        now = datetime.now()

        # Reset windows if needed
        if (now - self.minute_window_start).total_seconds() >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_window_start = now

        if (now - self.hour_window_start).total_seconds() >= 3600:
            self.requests_this_hour = 0
            self.hour_window_start = now

        if (now - self.day_window_start).total_seconds() >= 86400:
            self.requests_this_day = 0
            self.tokens_this_day = 0
            self.day_window_start = now

        # Check limits
        if self.requests_this_minute >= config.max_requests_per_minute:
            retry_after = 60 - (now - self.minute_window_start).total_seconds()
            return False, max(0, retry_after)

        if self.requests_this_hour >= config.max_requests_per_hour:
            retry_after = 3600 - (now - self.hour_window_start).total_seconds()
            return False, max(0, retry_after)

        if self.requests_this_day >= config.max_requests_per_day:
            retry_after = 86400 - (now - self.day_window_start).total_seconds()
            return False, max(0, retry_after)

        if config.max_tokens_per_minute and tokens:
            if self.tokens_this_minute + tokens > config.max_tokens_per_minute:
                retry_after = 60 - (now - self.minute_window_start).total_seconds()
                return False, max(0, retry_after)

        if config.max_tokens_per_day and tokens:
            if self.tokens_this_day + tokens > config.max_tokens_per_day:
                retry_after = 86400 - (now - self.day_window_start).total_seconds()
                return False, max(0, retry_after)

        # Update counters
        self.requests_this_minute += 1
        self.requests_this_hour += 1
        self.requests_this_day += 1
        self.tokens_this_minute += tokens
        self.tokens_this_day += tokens

        return True, None
