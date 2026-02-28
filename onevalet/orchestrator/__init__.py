"""
OneValet Orchestrator Module

Central coordinator for all agents with support for:
- ReAct loop (Reasoning + Acting) for tool/agent execution
- Agent pool management (memory and PostgreSQL backends)
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
        system_prompt="You are a cheerful assistant named Jarvis.",  # optional user persona
        react_config=ReactLoopConfig(max_turns=10),
    )
    await orchestrator.initialize()

    # Handle message
    response = await orchestrator.handle_message(tenant_id, message)

    # Stream events
    async for event in orchestrator.stream_message(tenant_id, message):
        print(event)

Session Management:
    Sessions are automatically persisted to PostgreSQL when a database
    is provided. Falls back to in-memory storage for testing.

    config = OrchestratorConfig(
        session=SessionConfig(
            enabled=True,
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
)

from .postgres_pool import PostgresPoolBackend

from .react_config import (
    ReactLoopConfig,
    ReactLoopResult,
    ToolCallRecord,
    TokenUsage,
    COMPLETE_TASK_TOOL_NAME,
    COMPLETE_TASK_SCHEMA,
    CompleteTaskResult,
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
    "PostgresPoolBackend",
    # ReAct
    "ReactLoopConfig",
    "ReactLoopResult",
    "ToolCallRecord",
    "TokenUsage",
    "COMPLETE_TASK_TOOL_NAME",
    "COMPLETE_TASK_SCHEMA",
    "CompleteTaskResult",
    # Main
    "Orchestrator",
]
