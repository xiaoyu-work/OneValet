"""
OneValet Hook System - Built-in observability for agents

This module provides:
- Built-in hooks: logging, metrics, tracing, rate limiting
- YAML configuration for zero-code observability
- Decorators for one-click logging/tracing

Quick Start - Use @logged decorator:
    from onevalet.hooks import logged

    @logged
    class MyAgent(StandardAgent):
        async def on_running(self, msg):
            return self.make_result(status=AgentStatus.COMPLETED, raw_message="Done!")

    # All state transitions, method calls, and errors are logged automatically!

With options:
    @logged(level="DEBUG", include_inputs=True, include_outputs=True)
    class VerboseAgent(StandardAgent):
        ...

Full observability (logging + tracing + metrics):
    @observable
    class FullyTrackedAgent(StandardAgent):
        ...

Manual configuration:
    from onevalet.hooks import HookManager, configure_hooks

    manager = configure_hooks({
        "logging": True,
        "metrics": True,
        "tracing": True,
    })

Example YAML:
    hooks:
      logging: true
      metrics: true
      tracing: true
      rate_limiting:
        enabled: true
        settings:
          max_requests_per_minute: 60
"""

from .models import (
    # Enums
    HookType,
    HookPhase,
    # Config
    HookConfig,
    # Context and Result
    HookContext,
    HookResult,
    # Data models
    MetricsData,
    TracingSpan,
    RateLimitConfig,
    RateLimitState,
)

from .handlers import (
    # Base
    HookHandler,
    # Built-in hooks
    LoggingHook,
    MetricsHook,
    TracingHook,
    RateLimitingHook,
    CustomHook,
)

from .manager import (
    HookManager,
    HookExecutionError,
    HookableAgent,
    with_hooks,
    get_global_hook_manager,
    configure_hooks,
)

from .decorators import (
    # One-click decorators
    logged,
    traced,
    metered,
    observable,
)

__all__ = [
    # Models
    "HookType",
    "HookPhase",
    "HookConfig",
    "HookContext",
    "HookResult",
    "MetricsData",
    "TracingSpan",
    "RateLimitConfig",
    "RateLimitState",
    # Handlers
    "HookHandler",
    "LoggingHook",
    "MetricsHook",
    "TracingHook",
    "RateLimitingHook",
    "CustomHook",
    # Manager
    "HookManager",
    "HookExecutionError",
    "HookableAgent",
    "with_hooks",
    "get_global_hook_manager",
    "configure_hooks",
    # Decorators (one-click logging)
    "logged",
    "traced",
    "metered",
    "observable",
]
