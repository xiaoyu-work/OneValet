"""OneValet Trigger Executors — action execution when triggers fire."""

import json
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .models import TriggerContext, ActionResult

if TYPE_CHECKING:
    from ..orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class OrchestratorExecutor:
    """
    Default executor: converts trigger events to messages and delegates
    to the Orchestrator's ReAct loop.

    The LLM autonomously decides which tools/agents to invoke based on
    the trigger context.
    """

    def __init__(self, orchestrator: "Orchestrator"):
        self.orchestrator = orchestrator

    async def execute(self, context: TriggerContext) -> ActionResult:
        """Execute by routing through the Orchestrator's ReAct loop."""
        message = self._build_message(context)
        tenant_id = context.task.user_id

        try:
            result = await self.orchestrator.handle_message(
                tenant_id=tenant_id,
                message=message,
                metadata={
                    "source": "trigger",
                    "trigger_type": context.trigger_type,
                    "task_id": context.task.id,
                },
            )

            action_result = ActionResult(
                success=True,
                output=result.raw_message or "",
                metadata={},
            )

            # Check if result has pending approvals
            if result.metadata and result.metadata.get("pending_approvals"):
                action_result.metadata["pending_approval"] = True
                action_result.metadata["agent_ids"] = [
                    a.get("agent_id") for a in result.metadata.get("pending_approvals", [])
                    if a.get("agent_id")
                ]

            return action_result

        except Exception as e:
            logger.error(f"OrchestratorExecutor failed for task {context.task.id}: {e}")
            return ActionResult(success=False, error=str(e))

    def _build_message(self, context: TriggerContext) -> str:
        """Build context message based on trigger type."""
        task = context.task
        instruction = task.action.instruction

        if context.trigger_type == "schedule":
            return f"[Scheduled Task] {instruction}" if instruction else f"[Scheduled Task] {task.name or task.description}"
        elif context.trigger_type == "event":
            event_info = json.dumps(context.event_data) if context.event_data else ""
            return f"[Event Trigger] {task.action.instruction or task.name}: {event_info}"
        elif context.trigger_type == "condition":
            expression = task.trigger.params.get("expression", "")
            return f"[Condition Trigger] Condition met: {expression}. {instruction}"
        else:
            return instruction or task.name or task.description
