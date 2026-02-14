"""OneValet Pipeline Executor — multi-step action execution with optional LLM condition gating."""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from jinja2 import Template

from .models import TriggerContext, ActionResult

if TYPE_CHECKING:
    from ..orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Multi-step executor: runs a sequence of orchestrator instructions,
    optionally gates on an LLM-evaluated condition, renders an output
    template, and sends notifications.

    ActionConfig.config format:
        {
            "steps": [
                {"instruction": "...", "output_key": "key1"},
                {"instruction": "...", "output_key": "key2"}
            ],
            "output_template": "Result: {{ key1 }}\n{{ key2 }}",
            "condition": "some condition to evaluate",
            "notify": true
        }
    """

    def __init__(
        self,
        orchestrator: "Orchestrator",
        llm_client: Optional[Any] = None,
        notification: Optional[Any] = None,
    ):
        self.orchestrator = orchestrator
        self.llm_client = llm_client
        self.notification = notification

    async def execute(self, context: TriggerContext) -> ActionResult:
        """Execute the pipeline defined in context.task.action.config."""
        config = context.task.action.config
        steps = config.get("steps", [])
        tenant_id = context.task.user_id

        metadata_base = {
            "source": "trigger",
            "trigger_type": context.trigger_type,
            "task_id": context.task.id,
        }

        # -- Run steps sequentially --
        step_outputs: Dict[str, str] = {}
        step_errors: Dict[str, str] = {}

        for step in steps:
            instruction = step.get("instruction", "")
            output_key = step.get("output_key", "")
            if not instruction or not output_key:
                continue

            try:
                result = await self.orchestrator.handle_message(
                    tenant_id=tenant_id,
                    message=instruction,
                    metadata=metadata_base,
                )
                step_outputs[output_key] = result.raw_message or ""
            except Exception as e:
                logger.error(f"Pipeline step '{output_key}' failed for task {context.task.id}: {e}")
                step_outputs[output_key] = ""
                step_errors[output_key] = str(e)

        # -- Condition gating --
        condition = config.get("condition")
        if condition and self.llm_client:
            try:
                prompt = (
                    f"Given this data: {step_outputs}, is this condition true: "
                    f"{condition}? Answer only YES or NO."
                )
                llm_response = await self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = llm_response.content.strip().upper() if llm_response.content else ""
                if answer != "YES":
                    return ActionResult(
                        success=True,
                        output="",
                        metadata={"skipped": True, "reason": "condition_not_met", "step_outputs": step_outputs},
                    )
            except Exception as e:
                logger.error(f"Pipeline condition evaluation failed for task {context.task.id}: {e}")
                # On condition evaluation failure, continue with output

        # -- Render output template --
        output_template = config.get("output_template")
        if output_template:
            try:
                rendered = Template(output_template).render(**step_outputs)
            except Exception as e:
                logger.error(f"Pipeline template render failed for task {context.task.id}: {e}")
                rendered = str(step_outputs)
        else:
            rendered = "\n".join(f"{k}: {v}" for k, v in step_outputs.items())

        # -- Notify --
        notify = config.get("notify", False)
        if notify and self.notification:
            notify_meta = {
                "task_id": context.task.id,
                "trigger_type": context.trigger_type,
                "source_event": context.event_data or {},
            }
            try:
                await self.notification.send(tenant_id, rendered, notify_meta)
            except Exception as e:
                logger.error(f"Pipeline notification failed for task {context.task.id}: {e}")

        return ActionResult(
            success=True,
            output=rendered,
            metadata={"step_outputs": step_outputs, "step_errors": step_errors},
        )
