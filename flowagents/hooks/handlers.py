"""
FlowAgent Hook Handlers - Built-in hook implementations

This module provides:
- LoggingHook: Automatic logging for agent execution
- MetricsHook: Automatic metrics collection
- TracingHook: Distributed tracing support
- RateLimitingHook: Rate limiting per user/agent
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable
from collections import defaultdict

from .models import (
    HookType,
    HookPhase,
    HookConfig,
    HookContext,
    HookResult,
    MetricsData,
    TracingSpan,
    RateLimitConfig,
    RateLimitState,
)


class HookHandler(ABC):
    """Base class for hook handlers"""

    hook_type: HookType

    def __init__(self, config: HookConfig):
        self.config = config
        self.enabled = config.enabled

    def should_apply(self, agent_type: str) -> bool:
        """Check if hook should apply to this agent type"""
        if not self.enabled:
            return False

        if self.config.agent_types:
            if agent_type not in self.config.agent_types:
                return False

        if self.config.exclude_agent_types:
            if agent_type in self.config.exclude_agent_types:
                return False

        return True

    @abstractmethod
    async def on_pre_execute(self, context: HookContext) -> HookResult:
        """Called before agent execution"""
        pass

    @abstractmethod
    async def on_post_execute(self, context: HookContext) -> HookResult:
        """Called after agent execution"""
        pass

    @abstractmethod
    async def on_error(self, context: HookContext) -> HookResult:
        """Called on error"""
        pass


class LoggingHook(HookHandler):
    """
    Automatic logging for agent execution.

    Logs:
    - Agent start/end
    - Input/output (if configured)
    - Errors
    - Duration
    """

    hook_type = HookType.LOGGING

    def __init__(self, config: HookConfig):
        super().__init__(config)
        self.log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        self.logger = logging.getLogger("flowagent.hooks")

    async def on_pre_execute(self, context: HookContext) -> HookResult:
        """Log agent start"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        msg = f"[{context.agent_type}] Starting execution"
        if context.method_name:
            msg += f" ({context.method_name})"

        extra = {
            "agent_id": context.agent_id,
            "agent_type": context.agent_type,
            "user_id": context.user_id,
            "phase": "pre_execute",
        }

        if self.config.include_inputs and context.input_message:
            input_str = str(context.input_message)
            if len(input_str) > 200:
                input_str = input_str[:200] + "..."
            extra["input"] = input_str

        self.logger.log(self.log_level, msg, extra=extra)

        return HookResult(hook_type=self.hook_type)

    async def on_post_execute(self, context: HookContext) -> HookResult:
        """Log agent completion"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        duration = context.duration_ms or 0
        msg = f"[{context.agent_type}] Completed in {duration:.2f}ms"
        if context.status:
            msg += f" (status: {context.status})"

        extra = {
            "agent_id": context.agent_id,
            "agent_type": context.agent_type,
            "user_id": context.user_id,
            "phase": "post_execute",
            "duration_ms": duration,
            "status": context.status,
        }

        if self.config.include_outputs and context.output_result:
            output_str = str(context.output_result)
            if len(output_str) > 200:
                output_str = output_str[:200] + "..."
            extra["output"] = output_str

        self.logger.log(self.log_level, msg, extra=extra)

        return HookResult(hook_type=self.hook_type)

    async def on_error(self, context: HookContext) -> HookResult:
        """Log errors"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        msg = f"[{context.agent_type}] Error: {context.error_type}: {context.error}"

        extra = {
            "agent_id": context.agent_id,
            "agent_type": context.agent_type,
            "user_id": context.user_id,
            "phase": "on_error",
            "error_type": context.error_type,
            "error": str(context.error),
        }

        self.logger.error(msg, extra=extra, exc_info=context.error)

        return HookResult(hook_type=self.hook_type)


class MetricsHook(HookHandler):
    """
    Automatic metrics collection for agent execution.

    Collects:
    - Invocation count
    - Success/error rates
    - Duration statistics
    - Token usage (if available)
    """

    hook_type = HookType.METRICS

    def __init__(self, config: HookConfig):
        super().__init__(config)
        # Metrics storage: (agent_type, user_id) -> MetricsData
        self._metrics: Dict[tuple, MetricsData] = defaultdict(
            lambda: MetricsData(agent_type="", user_id="")
        )
        # Global metrics per agent type
        self._global_metrics: Dict[str, MetricsData] = defaultdict(
            lambda: MetricsData(agent_type="", user_id="global")
        )

    async def on_pre_execute(self, context: HookContext) -> HookResult:
        """Record execution start"""
        # Nothing to do on pre-execute for metrics
        return HookResult(hook_type=self.hook_type)

    async def on_post_execute(self, context: HookContext) -> HookResult:
        """Record execution metrics"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        duration = context.duration_ms or 0
        success = context.error is None

        # Extract token usage from metadata
        tokens = context.metadata.get("tokens")
        cost = context.metadata.get("cost")

        # Update per-user metrics
        key = (context.agent_type, context.user_id)
        if key not in self._metrics:
            self._metrics[key] = MetricsData(
                agent_type=context.agent_type,
                user_id=context.user_id
            )
        self._metrics[key].record_invocation(duration, success, tokens, cost)

        # Update global metrics
        if context.agent_type not in self._global_metrics:
            self._global_metrics[context.agent_type] = MetricsData(
                agent_type=context.agent_type,
                user_id="global"
            )
        self._global_metrics[context.agent_type].record_invocation(duration, success, tokens, cost)

        return HookResult(
            hook_type=self.hook_type,
            data={
                "duration_ms": duration,
                "success": success,
                "invocation_count": self._metrics[key].invocation_count,
            }
        )

    async def on_error(self, context: HookContext) -> HookResult:
        """Record error metrics"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        duration = context.duration_ms or 0

        # Update metrics with error
        key = (context.agent_type, context.user_id)
        if key not in self._metrics:
            self._metrics[key] = MetricsData(
                agent_type=context.agent_type,
                user_id=context.user_id
            )
        self._metrics[key].record_invocation(duration, success=False)

        if context.agent_type not in self._global_metrics:
            self._global_metrics[context.agent_type] = MetricsData(
                agent_type=context.agent_type,
                user_id="global"
            )
        self._global_metrics[context.agent_type].record_invocation(duration, success=False)

        return HookResult(hook_type=self.hook_type)

    def get_metrics(
        self,
        agent_type: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, MetricsData]:
        """Get collected metrics"""
        if agent_type and user_id:
            key = (agent_type, user_id)
            if key in self._metrics:
                return {f"{agent_type}:{user_id}": self._metrics[key]}
            return {}

        if agent_type:
            return {agent_type: self._global_metrics.get(agent_type, MetricsData(agent_type=agent_type, user_id="global"))}

        # Return all
        result = {}
        for key, data in self._metrics.items():
            result[f"{key[0]}:{key[1]}"] = data
        return result

    def get_global_metrics(self) -> Dict[str, MetricsData]:
        """Get global metrics per agent type"""
        return dict(self._global_metrics)


class TracingHook(HookHandler):
    """
    Distributed tracing support.

    Creates spans for:
    - Agent execution
    - State transitions
    - LLM calls (if tracked)
    """

    hook_type = HookType.TRACING

    def __init__(self, config: HookConfig):
        super().__init__(config)
        # Active spans: agent_id -> span
        self._active_spans: Dict[str, TracingSpan] = {}
        # Completed spans for export
        self._completed_spans: List[TracingSpan] = []
        # Span exporters
        self._exporters: List[Callable[[TracingSpan], Awaitable[None]]] = []

    def add_exporter(self, exporter: Callable[[TracingSpan], Awaitable[None]]) -> None:
        """Add a span exporter"""
        self._exporters.append(exporter)

    async def on_pre_execute(self, context: HookContext) -> HookResult:
        """Create and start a span"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        # Create trace ID from context or generate new
        trace_id = context.metadata.get("trace_id", uuid.uuid4().hex)
        parent_span_id = context.metadata.get("parent_span_id")

        span = TracingSpan(
            span_id=uuid.uuid4().hex[:16],
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=f"{context.agent_type}.{context.method_name or 'execute'}",
            start_time=context.started_at or datetime.now(),
        )

        # Set attributes
        span.set_attribute("agent.id", context.agent_id)
        span.set_attribute("agent.type", context.agent_type)
        span.set_attribute("user.id", context.user_id)

        if context.method_name:
            span.set_attribute("agent.method", context.method_name)

        self._active_spans[context.agent_id] = span

        return HookResult(
            hook_type=self.hook_type,
            data={"span_id": span.span_id, "trace_id": span.trace_id}
        )

    async def on_post_execute(self, context: HookContext) -> HookResult:
        """End the span"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        span = self._active_spans.pop(context.agent_id, None)
        if not span:
            return HookResult(hook_type=self.hook_type)

        # End span
        span.end(status="ok")

        # Add result attributes
        if context.status:
            span.set_attribute("agent.status", context.status)
        if context.duration_ms:
            span.set_attribute("duration_ms", context.duration_ms)

        # Add event for completion
        span.add_event("execution.completed", {
            "status": context.status,
            "duration_ms": context.duration_ms,
        })

        # Store and export
        self._completed_spans.append(span)
        await self._export_span(span)

        return HookResult(
            hook_type=self.hook_type,
            data={"span": span.to_dict()}
        )

    async def on_error(self, context: HookContext) -> HookResult:
        """End span with error"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        span = self._active_spans.pop(context.agent_id, None)
        if not span:
            return HookResult(hook_type=self.hook_type)

        # End span with error
        span.end(status="error", error_message=str(context.error))

        # Add error attributes
        span.set_attribute("error", True)
        span.set_attribute("error.type", context.error_type or "Unknown")
        span.set_attribute("error.message", str(context.error))

        # Add error event
        span.add_event("exception", {
            "exception.type": context.error_type,
            "exception.message": str(context.error),
        })

        # Store and export
        self._completed_spans.append(span)
        await self._export_span(span)

        return HookResult(
            hook_type=self.hook_type,
            data={"span": span.to_dict()}
        )

    async def _export_span(self, span: TracingSpan) -> None:
        """Export span to configured exporters"""
        for exporter in self._exporters:
            try:
                await exporter(span)
            except Exception:
                pass  # Ignore export errors

    def get_completed_spans(self, limit: int = 100) -> List[TracingSpan]:
        """Get completed spans"""
        return self._completed_spans[-limit:]

    def clear_spans(self) -> None:
        """Clear completed spans"""
        self._completed_spans.clear()


class RateLimitingHook(HookHandler):
    """
    Rate limiting per user/agent.

    Checks:
    - Requests per minute/hour/day
    - Token usage limits
    - Per-agent type limits
    """

    hook_type = HookType.RATE_LIMITING

    def __init__(self, config: HookConfig):
        super().__init__(config)
        # Parse rate limit config from settings
        settings = config.settings
        self._rate_config = RateLimitConfig(
            max_requests_per_minute=settings.get("max_requests_per_minute", 60),
            max_requests_per_hour=settings.get("max_requests_per_hour", 1000),
            max_requests_per_day=settings.get("max_requests_per_day", 10000),
            max_agent_requests_per_minute=settings.get("max_agent_requests_per_minute"),
            max_tokens_per_minute=settings.get("max_tokens_per_minute"),
            max_tokens_per_day=settings.get("max_tokens_per_day"),
            agent_limits=settings.get("agent_limits", {}),
        )
        # User rate limit states
        self._user_states: Dict[str, RateLimitState] = {}

    async def on_pre_execute(self, context: HookContext) -> HookResult:
        """Check rate limits before execution"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        # Get or create user state
        if context.user_id not in self._user_states:
            self._user_states[context.user_id] = RateLimitState(user_id=context.user_id)

        state = self._user_states[context.user_id]

        # Get config for this agent type
        config = self._rate_config
        if context.agent_type in config.agent_limits:
            # Override with agent-specific limits
            agent_config = RateLimitConfig(**{**vars(config), **config.agent_limits[context.agent_type]})
            config = agent_config

        # Check limits
        allowed, retry_after = state.check_and_update(config)

        if not allowed:
            return HookResult(
                hook_type=self.hook_type,
                success=False,
                error=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds",
                should_proceed=False,
                retry_after=retry_after,
            )

        return HookResult(hook_type=self.hook_type, should_proceed=True)

    async def on_post_execute(self, context: HookContext) -> HookResult:
        """Nothing to do on post-execute for rate limiting"""
        return HookResult(hook_type=self.hook_type)

    async def on_error(self, context: HookContext) -> HookResult:
        """Nothing to do on error for rate limiting"""
        return HookResult(hook_type=self.hook_type)

    def get_user_state(self, user_id: str) -> Optional[RateLimitState]:
        """Get rate limit state for a user"""
        return self._user_states.get(user_id)

    def reset_user_state(self, user_id: str) -> None:
        """Reset rate limit state for a user"""
        if user_id in self._user_states:
            del self._user_states[user_id]


class CustomHook(HookHandler):
    """
    Custom hook with user-defined handlers.

    Allows users to add custom logic at each hook phase.
    """

    hook_type = HookType.CUSTOM

    def __init__(
        self,
        config: HookConfig,
        pre_execute: Optional[Callable[[HookContext], Awaitable[HookResult]]] = None,
        post_execute: Optional[Callable[[HookContext], Awaitable[HookResult]]] = None,
        on_error: Optional[Callable[[HookContext], Awaitable[HookResult]]] = None,
    ):
        super().__init__(config)
        self._pre_execute = pre_execute
        self._post_execute = post_execute
        self._on_error_handler = on_error

    async def on_pre_execute(self, context: HookContext) -> HookResult:
        """Call custom pre-execute handler"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        if self._pre_execute:
            return await self._pre_execute(context)
        return HookResult(hook_type=self.hook_type)

    async def on_post_execute(self, context: HookContext) -> HookResult:
        """Call custom post-execute handler"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        if self._post_execute:
            return await self._post_execute(context)
        return HookResult(hook_type=self.hook_type)

    async def on_error(self, context: HookContext) -> HookResult:
        """Call custom error handler"""
        if not self.should_apply(context.agent_type):
            return HookResult(hook_type=self.hook_type)

        if self._on_error_handler:
            return await self._on_error_handler(context)
        return HookResult(hook_type=self.hook_type)
