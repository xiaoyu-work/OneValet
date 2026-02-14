"""
Agent-Tool Execution - Execute agents as tools in the ReAct loop

Per design doc sections 4.1 and 6:
- Creates agent instances via the orchestrator
- Passes tool_call_args as context_hints for field pre-population
- Returns structured AgentToolResult based on agent execution status
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..message import Message
from ..result import AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class AgentToolResult:
    """Result from executing an agent as a tool in the ReAct loop."""

    completed: bool
    result_text: str = ""
    agent: Optional[Any] = None
    approval_request: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


async def execute_agent_tool(
    orchestrator,
    agent_type: str,
    tenant_id: str,
    tool_call_args: Dict[str, Any],
    task_instruction: str = "",
) -> AgentToolResult:
    """Execute an agent as a tool in the ReAct loop."""
    from .approval import build_approval_request

    enriched_hints = dict(tool_call_args)
    # task_instruction is popped from args in _execute_single() and passed
    # separately — put it back into hints so the agent has access to it.
    if task_instruction:
        enriched_hints["task_instruction"] = task_instruction
    if orchestrator.database:
        enriched_hints["db"] = orchestrator.database
    if orchestrator.trigger_engine:
        enriched_hints["trigger_engine"] = orchestrator.trigger_engine

    agent = await orchestrator.create_agent(
        tenant_id=tenant_id,
        agent_type=agent_type,
        context_hints=enriched_hints,
    )
    if agent is None:
        return AgentToolResult(
            completed=True,
            result_text=f"Error: Agent type '{agent_type}' not found.",
        )

    msg_content = task_instruction if task_instruction else ""
    msg = Message(
        name="orchestrator",
        content=msg_content,
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
