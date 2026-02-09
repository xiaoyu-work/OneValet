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
    completed: bool                                    # Whether the agent finished
    result_text: str = ""                              # Result text for LLM
    agent: Optional[Any] = None                        # Agent instance (non-None when WAITING)
    approval_request: Optional[Any] = None             # ApprovalRequest (non-None when WAITING_FOR_APPROVAL)


async def execute_agent_tool(
    orchestrator,  # Orchestrator instance
    agent_type: str,
    tenant_id: str,
    tool_call_args: Dict[str, Any],
    task_instruction: str = "",
) -> AgentToolResult:
    """Execute an agent as a tool in the ReAct loop.

    Per design doc sections 4.1 and 6:
    1. Create Agent via orchestrator.create_agent(), tool_call_args passed as context_hints
    2. Build Message from task_instruction (or empty)
    3. Call agent.reply(msg)
    4. Return based on AgentResult.status:
       - COMPLETED -> completed=True, result_text
       - WAITING_FOR_INPUT -> completed=False, agent instance
       - WAITING_FOR_APPROVAL -> completed=False, agent + ApprovalRequest
       - ERROR -> completed=True, error message
    """
    from .approval import build_approval_request

    # 1. Create agent instance with tool_call_args as context_hints
    agent = await orchestrator.create_agent(
        tenant_id=tenant_id,
        agent_type=agent_type,
        context_hints=tool_call_args,
    )
    if agent is None:
        return AgentToolResult(
            completed=True,
            result_text=f"Error: Agent type '{agent_type}' not found.",
        )

    # 2. Build message from task_instruction
    msg_content = task_instruction if task_instruction else ""
    msg = Message(
        name="orchestrator",
        content=msg_content,
        role="user",
    )

    # 3. Call agent.reply(msg)
    try:
        result = await agent.reply(msg)
    except Exception as e:
        logger.error(f"Agent execution failed for {agent_type}: {e}", exc_info=True)
        return AgentToolResult(
            completed=True,
            result_text=f"Error executing {agent_type}: {e}",
        )

    # 4. Return based on status
    if result.status == AgentStatus.COMPLETED:
        return AgentToolResult(
            completed=True,
            result_text=result.raw_message or "Agent completed successfully.",
        )

    elif result.status == AgentStatus.WAITING_FOR_INPUT:
        return AgentToolResult(
            completed=False,
            result_text=result.raw_message or "",
            agent=agent,
        )

    elif result.status == AgentStatus.WAITING_FOR_APPROVAL:
        approval_request = build_approval_request(agent)
        return AgentToolResult(
            completed=False,
            result_text=result.raw_message or "",
            agent=agent,
            approval_request=approval_request,
        )

    elif result.status == AgentStatus.ERROR:
        error_msg = result.error_message or result.raw_message or "Unknown error"
        return AgentToolResult(
            completed=True,
            result_text=f"Error: {error_msg}",
        )

    else:
        # CANCELLED, PAUSED, or other states
        return AgentToolResult(
            completed=True,
            result_text=result.raw_message or f"Agent finished with status: {result.status.value}",
        )
