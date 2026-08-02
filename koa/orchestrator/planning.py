"""The planning phase that runs before the ReAct loop.

Two cases share this code and are easy to confuse. Either the user is
answering a plan we showed them last turn, in which case the plan goes
back into the prompt and we do not plan again; or the request looks hard
enough to be worth laying out first, in which case we draft a plan and
either show it or execute it.

Which of the two happens is decided before any turn runs, so it lives
here rather than in the loop.
"""

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from ..constants import GENERATE_PLAN_SCHEMA
from ..streaming.models import AgentEvent, EventType
from .attendance import is_attended
from .react_config import COMPLETE_TASK_SCHEMA
from .run_state import RunState

logger = logging.getLogger(__name__)


@dataclass
class PlanOutcome:
    """What planning leaves behind for the loop."""

    #: Rebuilt prompt when a plan was injected; None means keep the original.
    messages: Optional[List[Dict[str, Any]]] = None

    #: The plan was shown and the run stops until the user answers.
    awaiting_approval: bool = False


class PlanningMixin:
    """Draft-a-plan-first behaviour, split out of the ReAct loop."""

    def _should_plan(
        self,
        context: Optional[Dict[str, Any]],
        routing_score: int,
    ) -> bool:
        """Decide whether this request warrants an explicit plan first.

        This used to read the model router's complexity score alone. That tied
        planning to an optional cost-optimisation feature: routing is opt-in,
        and routing_score stays at its -1 sentinel whenever it is off, the
        classifier fails, or a fallback model is in use -- so with the default
        config, planning could never fire at all.

        The intent analyzer runs on every request and already answers the same
        question more directly, at no extra round-trip. A request that needs
        several agents coordinated is exactly what a plan is for; the router's
        score is a fallback signal for when routing happens to be enabled.
        """
        if not self._react_config.planning_enabled:
            return False

        if routing_score >= self._react_config.planning_score_threshold:
            return True

        intent = (context or {}).get("intent_analysis")
        if intent is None:
            return False

        # Genuinely independent tasks across different agents -- the case the
        # DAG executor exists for, and the one worth showing the user first.
        sub_tasks = getattr(intent, "sub_tasks", None) or []
        if getattr(intent, "intent_type", "") == "multi" and len(sub_tasks) >= 2:
            return True

        return False

    async def _pop_pending_plan(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Take the plan this tenant was last shown, if any.

        Popped rather than read: a plan is answered once, and leaving it
        behind would make the next message look like another answer to it.
        """
        plan_store = getattr(self, "_plan_store", None)
        if plan_store is not None:
            pending = await plan_store.pop(tenant_id)
            if pending is not None:
                return pending
        return self._tenant_plans.pop(tenant_id, None)

    async def _save_pending_plan(self, tenant_id: str, plan_data: Dict[str, Any]) -> None:
        """Remember a plan we are about to show, so a restart does not lose it."""
        plan_store = getattr(self, "_plan_store", None)
        if plan_store is not None:
            await plan_store.save(tenant_id, plan_data)
        else:
            self._tenant_plans[tenant_id] = plan_data

    async def _draft_plan(
        self,
        state: RunState,
        context: Optional[Dict[str, Any]],
        user_message: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask the model for a plan. Returns None if it declined to make one."""
        plan_messages = await self._build_llm_messages(
            context,
            user_message,
            include_planning=True,
        )
        plan_response = await self._llm_call_with_retry(
            plan_messages,
            [GENERATE_PLAN_SCHEMA, COMPLETE_TASK_SCHEMA],
            tool_choice="auto",
            llm_client_override=state.llm_client,
        )
        # Planning is a real round-trip; leaving it out of the total made
        # planned requests look cheaper than they were.
        state.add_usage(getattr(plan_response, "usage", None))
        return self._extract_plan_from_response(plan_response)

    async def _plan_phase(
        self,
        state: RunState,
        outcome: PlanOutcome,
        *,
        context: Optional[Dict[str, Any]],
        user_message: str,
        metadata: Optional[Dict[str, Any]],
        enable_planning: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Run planning, yielding any plan events it produces."""
        pending_plan = await self._pop_pending_plan(state.tenant_id)
        if pending_plan and context:
            logger.info("[ReAct] Pending plan found, injecting into prompt for LLM to handle")
            outcome.messages = await self._build_llm_messages(
                context,
                user_message,
                pending_plan=self._format_plan_text(pending_plan),
            )
            return

        if not enable_planning:
            return

        logger.info(f"[ReAct] Planning phase triggered (score={state.routing_score})")
        await_approval = False
        try:
            plan_data = await self._draft_plan(state, context, user_message)
            if not plan_data:
                logger.info("[ReAct] LLM did not generate a plan, proceeding directly")
                return

            # Approval needs someone to give it. On a cron job or trigger the
            # plan would be presented to nobody and the run would stop there,
            # so those execute the plan directly.
            await_approval = self._react_config.planning_requires_approval and is_attended(
                metadata
            )
            plan_text = self._format_plan_text(plan_data)
            if await_approval:
                await self._save_pending_plan(state.tenant_id, plan_data)
            else:
                outcome.messages = await self._build_llm_messages(
                    context,
                    user_message,
                    approved_plan=plan_text,
                )
        except Exception as e:
            # A plan we cannot draft, persist, or feed back into the prompt is
            # a plan this run is better off without. Announcing one we failed
            # to apply would be worse than not planning at all.
            logger.warning(f"[ReAct] Planning phase failed, proceeding without plan: {e}")
            outcome.messages = None
            return

        yield AgentEvent(
            type=EventType.PLAN_GENERATED,
            data={"plan": plan_data, "plan_text": plan_text},
        )

        if await_approval:
            logger.info(f"[ReAct] Plan generated, awaiting approval: {plan_data.get('goal', '')}")
            state.final_response = self._format_plan_for_user(plan_data)
            state.result_status = "WAITING_FOR_APPROVAL"
            outcome.awaiting_approval = True
        else:
            logger.info(f"[ReAct] Plan auto-approved: {plan_data.get('goal', '')}")
