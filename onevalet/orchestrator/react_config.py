"""ReAct loop configuration and result dataclasses.

Centralizes all tunable parameters for the ReAct orchestration loop,
along with structured types for tracking tool calls, token usage, and
loop results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..constants import COMPLETE_TASK_TOOL_NAME, COMPLETE_TASK_SCHEMA  # noqa: F401


@dataclass
class CompleteTaskResult:
    """Marker returned when the LLM calls complete_task."""

    result: str


@dataclass
class ReactLoopConfig:
    """All ReAct loop configuration centralized in one place."""

    # Loop control
    max_turns: int = 10

    # Tool execution
    tool_execution_timeout: int = 30
    """Regular Tool timeout in seconds."""
    agent_tool_execution_timeout: int = 120
    """Agent-Tool timeout in seconds."""
    max_tool_result_share: float = 0.3
    """Single tool result may consume at most 30% of the context window."""
    max_tool_result_chars: int = 400_000
    """Single tool result hard character limit."""

    # Context management
    context_token_limit: int = 128_000
    """Context window size in tokens."""
    context_trim_threshold: float = 0.8
    """Trigger history trimming when usage exceeds this fraction."""
    max_history_messages: int = 40
    """Max messages retained after trimming."""

    # LLM calls
    llm_max_retries: int = 2
    """Max LLM call retries on transient errors."""
    llm_retry_base_delay: float = 1.0
    """Retry base delay in seconds (used for exponential back-off)."""

    # complete_task enforcement
    max_complete_task_retries: int = 3
    """Max grace-turn retries when LLM forgets to call complete_task."""

    # Approval
    approval_timeout_minutes: int = 30
    """Approval auto-cancel timeout in minutes."""

    # Thinking / Extended reasoning
    thinking_score_threshold: int = 51
    """Minimum complexity score to enable thinking. Requests with
    router score >= this value will use extended thinking on the
    first turn. Set to 0 to always enable, 101 to always disable."""
    thinking_budget: int = 10000
    """Max thinking tokens when thinking is enabled."""


@dataclass
class TokenUsage:
    """Accumulated token usage counters."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCallRecord:
    """Per-call telemetry for a single tool invocation."""

    name: str
    """Tool or Agent-Tool name."""
    args_summary: Dict
    """Truncated argument snapshot for observability."""
    duration_ms: int = 0
    """Wall-clock execution time in milliseconds."""
    success: bool = True
    """Whether the call completed without exception."""
    result_status: Optional[str] = None
    """For Agent-Tools: COMPLETED / WAITING_FOR_INPUT / WAITING_FOR_APPROVAL / ERROR."""
    result_chars: int = 0
    """Result size in characters before any truncation."""
    token_attribution: Optional[TokenUsage] = None
    """Tokens consumed by the LLM turn that produced this call."""


@dataclass
class ReactLoopResult:
    """Structured result returned by the ReAct loop."""

    response: str
    """Final answer produced by the loop."""
    turns: int = 0
    """Actual loop iterations executed."""
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    """Ordered list of every tool call made during the loop."""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    """Aggregate token usage across all LLM calls."""
    duration_ms: int = 0
    """Total wall-clock duration in milliseconds."""
    pending_approvals: list = field(default_factory=list)
    """Pending ApprovalRequest objects, if any."""
