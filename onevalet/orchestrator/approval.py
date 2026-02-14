"""
Approval System - Structured approval requests per design doc section 13.2

Provides:
- ApprovalRequest dataclass for structured approval presentation
- build_approval_request() to create requests from agent state
- collect_batch_approvals() for batching multiple approval requests
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
    """Structured approval request per design doc section 13.2."""
    agent_name: str
    action_summary: str
    risk_level: str = "write"  # "read", "write", "destructive"
    details: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=lambda: ["approve", "edit", "cancel"])
    timeout_minutes: int = 30
    allow_modification: bool = True


def build_approval_request(agent) -> ApprovalRequest:
    """Build ApprovalRequest from agent's get_approval_prompt() + collected_fields.

    Args:
        agent: A StandardAgent instance in WAITING_FOR_APPROVAL state.

    Returns:
        ApprovalRequest populated from the agent's state.
    """
    agent_name = agent.__class__.__name__

    # Get the approval prompt from the agent
    action_summary = ""
    if hasattr(agent, "get_approval_prompt"):
        action_summary = agent.get_approval_prompt()

    # Gather collected fields as details
    details: Dict[str, Any] = {}
    if hasattr(agent, "collected_fields"):
        details = dict(agent.collected_fields)

    # Extract risk_level from agent's pending tool call if available
    risk_level = "write"
    if hasattr(agent, "_pending_tool_call") and agent._pending_tool_call:
        _, tool, _ = agent._pending_tool_call
        if hasattr(tool, "risk_level"):
            risk_level = tool.risk_level

    return ApprovalRequest(
        agent_name=agent_name,
        action_summary=action_summary,
        risk_level=risk_level,
        details=details,
    )


def collect_batch_approvals(approval_requests: List[ApprovalRequest]) -> List[ApprovalRequest]:
    """When multiple Agent-Tools need approval, collect all requests for batch presentation.

    Groups approval requests together so they can be presented to the user
    in a single batch UI rather than one at a time.

    Args:
        approval_requests: List of individual ApprovalRequest objects.

    Returns:
        The same list of ApprovalRequest objects, ready for batch presentation.
        In the current implementation this is a pass-through; future versions
        may merge or reorder requests.
    """
    if not approval_requests:
        return []

    logger.debug(f"Collecting {len(approval_requests)} approval requests for batch presentation")
    return list(approval_requests)
