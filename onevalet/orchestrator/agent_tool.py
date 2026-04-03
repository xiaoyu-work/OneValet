"""
Agent-Tool Execution - Execute agents as tools in the ReAct loop

Per design doc sections 4.1 and 6:
- Creates agent instances via the orchestrator
- Passes tool_call_args as context_hints for field pre-population
- Returns structured AgentToolResult based on agent execution status
- Builds structured HandoffContext for rich agent context passing
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..message import Message
from ..result import AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class HandoffContext:
    """Structured context passed from orchestrator to agent during handoff."""

    task_summary: str  # concise description of what agent should do
    known_entities: Dict[str, Any]  # extracted entities from tool_call_args
    conversation_context: str  # summary of recent conversation (last 3 turns)
    constraints: List[str]  # any constraints mentioned by user
    session_memory: Dict[str, Any]  # orchestrator-owned working memory snapshot


@dataclass
class AgentToolResult:
    """Result from executing an agent as a tool in the ReAct loop."""

    completed: bool
    result_text: str = ""
    agent: Optional[Any] = None
    approval_request: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_recent_context(request_context: Optional[Dict[str, Any]], tenant_id: str) -> str:
    """Extract a brief summary of recent conversation from context.

    Returns a string summarizing the last 3 conversation turns (max 500 chars).
    If no history is available, returns an empty string.
    """
    try:
        history = request_context.get("conversation_history", []) if request_context else []
        if not history:
            return ""

        # Take last 6 messages (up to 3 turns of user + assistant)
        recent = history[-6:]
        parts = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 120:
                content = content[:117] + "..."
            parts.append(f"{role}: {content}")

        summary = " | ".join(parts)
        if len(summary) > 500:
            summary = summary[:497] + "..."
        return summary
    except Exception as e:
        logger.debug(f"Could not extract recent context for {tenant_id}: {e}")
        return ""


async def execute_agent_tool(
    orchestrator,
    agent_type: str,
    tenant_id: str,
    tool_call_args: Dict[str, Any],
    task_instruction: str = "",
    request_context: Optional[Dict[str, Any]] = None,
) -> AgentToolResult:
    """Execute an agent as a tool in the ReAct loop."""
    from .approval import build_approval_request

    # Build structured handoff context
    handoff = HandoffContext(
        task_summary=task_instruction or f"Execute {agent_type} task",
        known_entities={k: v for k, v in tool_call_args.items() if v is not None},
        conversation_context=_extract_recent_context(request_context, tenant_id),
        constraints=[],
        session_memory=dict((request_context or {}).get("session_working_memory") or {}),
    )

    enriched_hints = dict(tool_call_args)
    # task_instruction is popped from args in _execute_single() and passed
    # separately — put it back into hints so the agent has access to it.
    if task_instruction:
        enriched_hints["task_instruction"] = task_instruction
    if orchestrator.database:
        enriched_hints["db"] = orchestrator.database
    if orchestrator.trigger_engine:
        enriched_hints["trigger_engine"] = orchestrator.trigger_engine
        if orchestrator.trigger_engine.cron_service:
            enriched_hints["cron_service"] = orchestrator.trigger_engine.cron_service
    # Pass user images to agent tools (e.g. for receipt scanning)
    current_images = (request_context or {}).get("user_images")
    if current_images:
        enriched_hints["user_images"] = current_images
    # Pass user timezone to agent tools
    user_timezone = (request_context or {}).get("metadata", {}).get("timezone")
    if user_timezone:
        enriched_hints["timezone"] = user_timezone
    permissions = (request_context or {}).get("metadata", {}).get("permissions")
    if permissions and isinstance(permissions, dict):
        enriched_hints["permissions"] = permissions
    session_id = (request_context or {}).get("session_id")
    if session_id:
        enriched_hints["session_id"] = session_id
    recalled_memories = (request_context or {}).get("recalled_memories")
    if recalled_memories:
        enriched_hints["recalled_memories"] = recalled_memories
    # Pass user location to agent tools (lat/lng from device)
    user_location = (request_context or {}).get("metadata", {}).get("location")
    if user_location and isinstance(user_location, dict):
        enriched_hints["user_location"] = user_location
    # Pass user context for personalization
    true_memory = (request_context or {}).get("metadata", {}).get("true_memory")
    if true_memory:
        enriched_hints["true_memory"] = true_memory
    user_profile = (request_context or {}).get("metadata", {}).get("user_profile")
    if user_profile:
        enriched_hints["user_profile"] = user_profile
    # Supabase storage as default cloud storage provider (if configured)
    supabase_storage = getattr(orchestrator, "_supabase_storage", None)
    if supabase_storage and "cloud_storage_provider" not in enriched_hints:
        enriched_hints["cloud_storage_provider"] = supabase_storage.for_tenant(tenant_id)

    # Pass structured handoff via context_hints
    enriched_hints["session_working_memory"] = handoff.session_memory
    enriched_hints["handoff"] = {
        "task_summary": handoff.task_summary,
        "known_entities": handoff.known_entities,
        "conversation_context": handoff.conversation_context,
        "constraints": handoff.constraints,
        "session_memory": handoff.session_memory,
    }

    # Check if agent requires a service the tenant hasn't connected
    registry = getattr(orchestrator, "_agent_registry", None)
    if registry:
        metadata = registry.get_agent_metadata(agent_type)
        if metadata:
            required_services = metadata.extra.get("requires_service", [])
            if required_services:
                credential_store = getattr(orchestrator, "credential_store", None)
                if credential_store:
                    try:
                        accounts = await credential_store.list(tenant_id)
                        tenant_services = {a["service"] for a in accounts}
                        if not (set(required_services) & tenant_services):
                            from ..errors import OneValetError, E
                            err = OneValetError(
                                E.SERVICE_NOT_CONNECTED,
                                f"{agent_type} requires a connected service",
                                details={
                                    "agent_type": agent_type,
                                    "required_services": required_services,
                                },
                            )
                            return AgentToolResult(
                                completed=True,
                                result_text=f"[{err.code}] {err.message}",
                                metadata={"error": err.to_dict()},
                            )
                    except Exception as e:
                        logger.debug(f"Could not check tenant services: {e}")

    agent = await orchestrator.create_agent(
        tenant_id=tenant_id,
        agent_type=agent_type,
        context_hints=enriched_hints,
    )
    if agent is None:
        logger.error(
            f"[AgentTool] create_agent returned None for {agent_type}, tenant={tenant_id}"
        )
        from ..errors import E
        return AgentToolResult(
            completed=True,
            result_text=f"[{E.AGENT_FAILED}] Failed to create agent {agent_type}",
            metadata={"error": {"code": E.AGENT_FAILED, "message": f"Failed to create agent {agent_type}",
                                "details": {"agent_type": agent_type}}},
        )

    # Build message for the agent — only pass the task instruction.
    # Conversation context is available via enriched_hints["handoff"]
    # but should NOT be injected into the user message as it can trigger
    # Azure content filters (looks like prompt injection).
    msg = Message(
        name="orchestrator",
        content=task_instruction or "",
        role="user",
    )

    try:
        result = await agent.reply(msg)
        logger.info(f"[AgentTool] {agent_type} status={result.status.value}")
    except Exception as e:
        logger.error(f"Agent execution failed for {agent_type}: {e}", exc_info=True)
        await orchestrator.agent_pool.remove_agent(tenant_id, agent.agent_id)
        return AgentToolResult(
            completed=True,
            result_text=f"Error executing {agent_type}: {e}",
        )

    if result.status == AgentStatus.COMPLETED:
        # Agent completed — remove from pool to prevent memory leak.
        # It was added during create_agent() but is no longer needed.
        await orchestrator.agent_pool.remove_agent(tenant_id, agent.agent_id)
        completed_meta = dict(result.metadata or {})
        completed_meta.setdefault("agent_status", AgentStatus.COMPLETED.value)
        return AgentToolResult(
            completed=True,
            result_text=result.raw_message or "Agent completed successfully.",
            metadata=completed_meta,
        )

    if result.status == AgentStatus.WAITING_FOR_INPUT:
        waiting_meta = dict(result.metadata or {})
        waiting_meta["requires_user_input"] = True
        waiting_meta["agent_status"] = AgentStatus.WAITING_FOR_INPUT.value
        return AgentToolResult(
            completed=False,
            result_text=result.raw_message or "",
            agent=agent,
            metadata=waiting_meta,
        )

    if result.status == AgentStatus.WAITING_FOR_APPROVAL:
        approval_request = build_approval_request(agent)
        waiting_meta = dict(result.metadata or {})
        waiting_meta["requires_user_input"] = True
        waiting_meta["requires_approval"] = True
        waiting_meta["agent_status"] = AgentStatus.WAITING_FOR_APPROVAL.value
        return AgentToolResult(
            completed=False,
            result_text=result.raw_message or "",
            agent=agent,
            approval_request=approval_request,
            metadata=waiting_meta,
        )

    if result.status == AgentStatus.ERROR:
        await orchestrator.agent_pool.remove_agent(tenant_id, agent.agent_id)
        error_msg = result.error_message or result.raw_message or "Unknown error"
        error_meta = dict(result.metadata or {})
        error_meta.setdefault("agent_status", AgentStatus.ERROR.value)
        return AgentToolResult(
            completed=True,
            result_text=f"Error: {error_msg}",
            metadata=error_meta,
        )

    # Any other terminal status — clean up from pool
    await orchestrator.agent_pool.remove_agent(tenant_id, agent.agent_id)
    other_meta = dict(result.metadata or {})
    other_meta.setdefault("agent_status", result.status.value)
    return AgentToolResult(
        completed=True,
        result_text=result.raw_message or f"Agent finished with status: {result.status.value}",
        metadata=other_meta,
    )
