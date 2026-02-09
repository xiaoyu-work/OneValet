"""
OneValet Orchestrator Module

Central coordinator for all agents with support for:
- Message routing to agents and workflows
- Agent pool management (memory and Redis backends)
- Session persistence with TTL
- Multi-agent collaboration
- Streaming execution events

Quick Start:
    from onevalet.orchestrator import Orchestrator, OrchestratorConfig

    orchestrator = Orchestrator(
        config=OrchestratorConfig(config_dir="./config"),
        llm_client=llm_client
    )
    await orchestrator.initialize()

    # Handle message
    response = await orchestrator.handle_message(tenant_id, message)

    # Stream events
    async for event in orchestrator.stream_message(tenant_id, message):
        print(event)

Routing:
    The orchestrator routes messages using:
    1. Active agent check (continue conversation)
    2. Workflow trigger matching
    3. Agent trigger matching
    4. LLM-based intelligent routing (optional)
    5. Fallback response

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

from .router import (
    MessageRouter,
    TriggerMatcher,
)

from .orchestrator import Orchestrator

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
    # Router
    "MessageRouter",
    "TriggerMatcher",
    # Main
    "Orchestrator",
]
