"""ReAct loop configuration and result dataclasses.

Centralizes all tunable parameters for the ReAct orchestration loop,
along with structured types for tracking tool calls, token usage, and
loop results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..constants import COMPLETE_TASK_SCHEMA, COMPLETE_TASK_TOOL_NAME  # noqa: F401


@dataclass
class CompleteTaskResult:
    """Marker returned when the LLM calls complete_task."""

    result: str


@dataclass
class ReactLoopConfig:
    """All ReAct loop configuration centralized in one place."""

    # Loop control
    max_turns: int = 60
    """Max model<->tool iterations for one request. Long tasks (multi-step research,
    trip planning, cross-agent workflows) routinely need more than a handful of turns;
    the loop still terminates on its own as soon as the model stops requesting tools."""
    react_timeout: Optional[float] = None
    """Optional wall-clock budget in seconds for the entire ReAct loop.

    ``None`` (the default) means no wall-clock kill: a task runs until it finishes,
    exhausts ``max_turns``, or is interrupted. Set a float only when a caller genuinely
    needs a deadline (e.g. a synchronous HTTP handler); it is a budget, not a safety net."""

    # Tool execution
    tool_execution_timeout: int = 30
    """Regular Tool timeout in seconds."""
    agent_tool_execution_timeout: int = 120
    """Agent-Tool timeout in seconds."""
    dag_timeout: float = 1800.0
    """Wall-clock budget for a whole multi-intent DAG execution, in seconds."""
    max_tool_result_share: float = 0.3
    """Single tool result may consume at most 30% of the context window."""
    max_tool_result_chars: int = 400_000
    """Single tool result hard character limit."""

    # Context management
    context_token_limit: int = 128_000
    """Fallback context window, used only when the LLM client does not report one.
    The real window is bound per-request via ContextManager.set_context_window()."""
    context_trim_threshold: float = 0.8
    """Trigger history trimming when usage exceeds this fraction."""
    max_history_messages: int = 40
    """Max messages retained after trimming."""

    # LLM calls
    llm_max_retries: int = 2
    """Max LLM call retries on transient errors."""
    llm_retry_base_delay: float = 1.0
    """Retry base delay in seconds (used for exponential back-off)."""

    # Approval
    approval_timeout_minutes: int = 30
    """Approval auto-cancel timeout in minutes."""

    # Model fallback
    fallback_providers: List[str] = field(default_factory=list)
    """Ordered list of LLMRegistry provider names to try when the primary
    model fails after exhausting retries. Example: ["anthropic_main", "deepseek"]."""

    # Planning
    planning_score_threshold: int = 40
    """Minimum complexity score to trigger planning phase. Requests with
    router score >= this value will generate a plan before executing.
    Set to 0 to always plan, 101 to never plan."""
    planning_requires_approval: bool = True
    """Whether to wait for user approval before executing the plan.
    If False, plan is generated and executed automatically."""

    # Extended reasoning (provider-agnostic via litellm reasoning_effort)
    reasoning_score_threshold: int = 51
    """Minimum complexity score to enable reasoning. Requests with
    router score >= this value will use extended reasoning on the
    first turn. Set to 0 to always enable, 101 to always disable."""
    reasoning_effort: str = "medium"
    """Reasoning effort level passed to litellm: "low", "medium", or "high".
    litellm translates this to each provider's native format automatically."""


@dataclass
class TokenUsage:
    """Accumulated token usage counters."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

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
