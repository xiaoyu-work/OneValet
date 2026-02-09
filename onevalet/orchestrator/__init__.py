"""
OneValet Orchestrator Module

Central coordinator for all agents with support for:
- ReAct loop (Reasoning + Acting) for tool/agent execution
- Agent pool management (memory and Redis backends)
- Session persistence with TTL
- Multi-agent collaboration via Agent-Tools
- Streaming execution events
- Context management with three lines of defense

Quick Start:
    from onevalet.orchestrator import Orchestrator, OrchestratorConfig, ReactLoopConfig

    orchestrator = Orchestrator(
        config=OrchestratorConfig(),
        llm_client=llm_client,
        agent_registry=registry,
        system_prompt="You are a helpful assistant.",
        react_config=ReactLoopConfig(max_turns=10),
    )
    await orchestrator.initialize()

    # Handle message
    response = await orchestrator.handle_message(tenant_id, message)

    # Stream events
    async for event in orchestrator.stream_message(tenant_id, message):
        print(event)

Session Management:
    Sessions can be persisted using memory or Redis backend:

    config = OrchestratorConfig(
        session=SessionConfig(
            enabled=True,
            backend="redis",
            redis_url="redis://localhost:6379",
            session_ttl_seconds=86400  # 24 hours
        )
    )
"""

from .models import (
    RoutingAction,
    RoutingReason,
    RoutingDecision,
    OrchestratorConfig,
    SessionConfig,
    AgentPoolEntry,
    AgentCallback,
    callback_handler,
)

from .pool import (
    AgentPoolManager,
    PoolBackend,
    MemoryPoolBackend,
    RedisPoolBackend,
)

from .react_config import (
    ReactLoopConfig,
    ReactLoopResult,
    ToolCallRecord,
    TokenUsage,
)

from .orchestrator import Orchestrator

# Backward-compatible lazy import for MessageRouter / TriggerMatcher.
# These are deprecated; the ReAct loop replaces the router-based flow.
# Users can still do `from onevalet.orchestrator import MessageRouter`.


def __getattr__(name: str):
    if name in ("MessageRouter", "TriggerMatcher"):
        from . import router as _router_module  # noqa: F811
        return getattr(_router_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "RoutingAction",
    "RoutingReason",
    "RoutingDecision",
    "OrchestratorConfig",
    "SessionConfig",
    "AgentPoolEntry",
    "AgentCallback",
    "callback_handler",
    # Pool
    "AgentPoolManager",
    "PoolBackend",
    "MemoryPoolBackend",
    "RedisPoolBackend",
    # ReAct
    "ReactLoopConfig",
    "ReactLoopResult",
    "ToolCallRecord",
    "TokenUsage",
    # Main
    "Orchestrator",
    # Deprecated (still importable via __getattr__)
    "MessageRouter",
    "TriggerMatcher",
]
