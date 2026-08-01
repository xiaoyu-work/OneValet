"""Request analysis — intent classification, planning, and speculative work.

Everything that runs *before* the ReAct loop to decide how a request should be
handled: which domains it touches, whether it decomposes into sub-tasks, whether
it warrants a plan the user approves first, and which tool calls are worth
starting early on a hunch.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..constants import GENERATE_PLAN_TOOL_NAME
from ..models import AgentToolContext

if TYPE_CHECKING:
    from .intent_analyzer import IntentAnalysis

logger = logging.getLogger(__name__)


class RequestAnalysisMixin:
    """Mixin providing intent analysis, plan formatting, and speculation.

    Expects the following on ``self`` (provided by Orchestrator):
    - ``llm_client``, ``task_registry``
    - ``intent_embedding_router``, ``intent_feedback_store``
    - ``_SPECULATIVE_TIMEOUT``
    """

    def _start_speculative_tasks(
        self,
        message: str,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, asyncio.Task]:
        """Start speculative tool execution for image requests.

        For requests containing images, the LLM almost always invokes
        ``google_search``.  By kicking the search off *before* the first
        LLM call, we can shave 1-3 s off the total latency.

        Returns a dict mapping task keys to ``asyncio.Task`` objects.
        The react loop checks these tasks and reuses results when the
        LLM requests a matching tool call.
        """
        tasks: Dict[str, asyncio.Task] = {}

        try:
            from ..builtin_agents.tools.google_search import google_search_executor
        except ImportError:
            logger.debug("[Speculative] google_search not available, skipping")
            return tasks

        # Speculative web search using the user's raw prompt
        async def _run_web_search():
            try:
                ctx = AgentToolContext(tenant_id=tenant_id, metadata=metadata or {})
                return await asyncio.wait_for(
                    google_search_executor(
                        {"query": message, "num_results": 5, "search_type": "web"},
                        ctx,
                    ),
                    timeout=self._SPECULATIVE_TIMEOUT,
                )
            except Exception as e:
                logger.info(f"[Speculative] web search failed (non-fatal): {e}")
                return None

        # Speculative image search using the user's raw prompt
        async def _run_image_search():
            try:
                ctx = AgentToolContext(tenant_id=tenant_id, metadata=metadata or {})
                return await asyncio.wait_for(
                    google_search_executor(
                        {"query": message, "num_results": 5, "search_type": "image"},
                        ctx,
                    ),
                    timeout=self._SPECULATIVE_TIMEOUT,
                )
            except Exception as e:
                logger.info(f"[Speculative] image search failed (non-fatal): {e}")
                return None

        tasks["google_search:web"] = self.task_registry.create_task(
            _run_web_search(), name="speculative:google_search:web"
        )
        tasks["google_search:image"] = self.task_registry.create_task(
            _run_image_search(), name="speculative:google_search:image"
        )

        logger.info(f"[Speculative] Started {len(tasks)} speculative tasks for image request")
        return tasks

    @staticmethod
    def _cancel_speculative_tasks(
        speculative: Dict[str, asyncio.Task],
    ) -> None:
        """Cancel any speculative tasks that were not consumed."""
        for key, task in speculative.items():
            if task and not task.done():
                task.cancel()
                logger.info(f"[Speculative] Cancelled unused task: {key}")

    @staticmethod
    def _extract_plan_from_response(response: Any) -> Optional[Dict[str, Any]]:
        """Extract generate_plan tool call arguments from an LLM response."""
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return None
        for tc in tool_calls:
            if tc.name == GENERATE_PLAN_TOOL_NAME:
                args = tc.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                return args
        return None

    @staticmethod
    def _format_plan_text(plan_data: Dict[str, Any]) -> str:
        """Format structured plan data into readable text for prompt injection."""
        lines = [f"**Goal:** {plan_data.get('goal', '')}"]
        for step in plan_data.get("steps", []):
            deps = step.get("depends_on", [])
            dep_str = (
                f" (after step {', '.join(map(str, deps))})" if deps else " (can start immediately)"
            )
            lines.append(f"{step['id']}. [{step.get('agent', '?')}] {step['action']}{dep_str}")
            if step.get("reason"):
                lines.append(f"   Reason: {step['reason']}")
        return "\n".join(lines)

    @staticmethod
    def _format_plan_for_user(plan_data: Dict[str, Any]) -> str:
        """Format plan as a friendly user-facing message."""
        lines = [plan_data.get("goal", "")]
        lines.append("")
        for step in plan_data.get("steps", []):
            deps = step.get("depends_on", [])
            if deps:
                dep_str = f" (after step {', '.join(map(str, deps))})"
            else:
                dep_str = ""
            lines.append(f"{step['id']}. {step['action']}{dep_str}")
        lines.append("")
        lines.append("Ready to execute. You can approve, modify, or cancel.")
        return "\n".join(lines)

    async def _analyze_intent(
        self,
        message: str,
        context: Dict[str, Any],
    ) -> "IntentAnalysis":
        """Analyze user message to determine intent type and domain(s).

        Single lightweight LLM call (~200 tokens).  Pre-empted by the
        fast-path regex classifier for trivial utterances and, when
        configured, by the embedding L1 router for messages close to
        known-intent centroids.
        Falls back to all-domains on failure.
        """
        from .intent_analyzer import IntentAnalyzer

        analyzer = IntentAnalyzer(
            self.llm_client,
            embedding_router=self.intent_embedding_router,
        )
        history = context.get("conversation_history", [])
        metadata = context.get("metadata", {})
        intent = await analyzer.analyze(
            message,
            conversation_history=history,
            metadata=metadata,
        )

        logger.info(
            "[IntentAnalyzer] source=%s type=%s domains=%s "
            "needs_memory=%s confidence=%.2f clarify=%s sub_tasks=%d",
            intent.source,
            intent.intent_type,
            intent.domains,
            intent.needs_memory,
            intent.confidence,
            intent.needs_clarification,
            len(intent.sub_tasks),
        )
        return intent

    def _record_intent_feedback(
        self,
        *,
        tenant_id: str,
        intent: "IntentAnalysis",
        outcome: str,
        parent_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fire-and-forget feedback write.

        Intentionally swallows all exceptions — feedback recording must
        never block or fail a user request.
        """
        store = self.intent_feedback_store
        if store is None:
            return
        try:
            coro = store.record(
                tenant_id=tenant_id,
                user_message=intent.raw_message or "",
                intent_type=intent.intent_type,
                domains=list(intent.domains),
                confidence=float(intent.confidence),
                source=intent.source,
                outcome=outcome,
                parent_id=parent_id,
                extra=extra,
            )
            self.task_registry.create_task(coro, name="intent_feedback")
        except Exception as exc:
            logger.debug("intent feedback record failed: %s", exc)
