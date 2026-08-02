"""ReAct loop mixin for the Orchestrator.

Contains the core ReAct (Reasoning + Acting) loop implementation
and its helper methods.
"""

import asyncio
import dataclasses
import json
import logging
import random
import time
from collections import namedtuple
from typing import Any, AsyncIterator, Dict, List, Optional

from ..constants import GENERATE_PLAN_SCHEMA
from ..llm.tool_validator import ToolSchemaValidator
from ..models import ToolOutput
from ..streaming.models import AgentEvent, EventType
from .agent_tool import AgentToolResult
from .approval import collect_batch_approvals
from .attendance import is_attended
from .error_classifier import LLMErrorKind
from .react_config import (
    COMPLETE_TASK_SCHEMA,
    COMPLETE_TASK_TOOL_NAME,
    CompleteTaskResult,
    TokenUsage,
    ToolCallRecord,
)
from .transcript_repair import repair_transcript

logger = logging.getLogger(__name__)

TimedResult = namedtuple("TimedResult", ["result", "duration_ms"])

#: Sentinel distinguishing "the run was interrupted" from a real (possibly
#: falsy) result when racing an awaitable against the cancel signal.
_INTERRUPTED = object()


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _media_key(item: Dict[str, Any]) -> str:
    if item.get("type") == "inline_cards":
        data = item.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
        return f"inline_cards:{_stable_json(data)}"
    return _stable_json(
        {
            "type": item.get("type"),
            "data": item.get("data"),
            "media_type": item.get("media_type"),
            "metadata": item.get("metadata"),
        }
    )


def _append_unique_media(target: List[Dict[str, Any]], media: List[Dict[str, Any]]) -> None:
    seen = {_media_key(item) for item in target}
    for item in media:
        key = _media_key(item)
        if key in seen:
            continue
        seen.add(key)
        target.append(item)


class _ReactLoopLLMError(Exception):
    """Raised when the ReAct loop LLM call fails after all retries.

    Carries classification metadata so the caller can decide whether
    to attempt model-level fallback.
    """

    def __init__(self, original: Exception, error_kind: "LLMErrorKind", turn: int):  # noqa: F821
        self.original = original
        self.error_kind = error_kind
        self.turn = turn
        super().__init__(str(original))


# ── Tool acknowledgment messages ──────────────────────────────────
# Shown to the user before tool execution so they know Koi is working.
# Only emitted on turn 1 (first tool invocation); subsequent turns in
# the same ReAct loop skip the acknowledgment to avoid clutter.

_CASUAL_ACKS = [
    "On it!",
    "Got it, one sec…",
    "Sure, let me check…",
    "One moment…",
    "Let me look into that…",
    "Sure thing…",
    "On it, give me a sec…",
    "Let me see…",
]


def _tool_acknowledgment(tool_names: List[str], turn: int) -> Optional[str]:
    """Return a short, casual acknowledgment string, or None to skip."""
    if turn > 1:
        return None

    # Skip for simple utility tools that resolve instantly
    skip = {"complete_task", "generate_plan"}
    if all(n in skip for n in tool_names):
        return None

    return random.choice(_CASUAL_ACKS)


class ReactLoopMixin:
    """Mixin providing the ReAct loop and its helpers.

    Expects the following attributes on ``self`` (provided by Orchestrator):
    - ``llm_client``
    - ``_react_config``
    - ``_context_manager``
    - ``_model_router``
    - ``_audit``
    - ``_run_controls``
    - ``agent_pool``
    - ``database``
    - ``_tenant_plans``

    Also expects the following methods (from other mixins or Orchestrator):
    - ``_llm_call_with_retry()`` (LLMManagerMixin)
    - ``_execute_with_timeout()`` (ToolManagerMixin)
    - ``_is_agent_tool()`` (ToolManagerMixin)
    - ``_cap_tool_result()`` (ToolManagerMixin)
    - ``_build_llm_messages()`` (Orchestrator)
    """

    def _compensate_pending_tool_calls(
        self,
        tool_calls: List[Any],
        messages: List[Dict[str, Any]],
        reason: str = "Interrupted by user before execution.",
    ) -> None:
        """Give every tool call that will not run an error result.

        An assistant message carrying tool_calls with no matching tool results
        is rejected outright by several hosted chat templates, and any resumed
        or replayed transcript would re-prompt them. So on the stop path each
        pending call still gets an answer, and the history stays well-formed.
        """
        answered = {
            m.get("tool_call_id") for m in messages if m.get("role") == "tool"
        }
        for tc in tool_calls:
            if tc.id in answered:
                continue
            messages.append(self._build_tool_result_message(tc.id, reason, is_error=True))

    def _apply_steering(
        self,
        control: Any,
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """Append any queued steering messages as user turns. Returns what was applied."""
        pending = control.drain_steering()
        for text in pending:
            messages.append({"role": "user", "content": text})
        if pending:
            logger.info(f"[ReAct] Injected {len(pending)} steering message(s)")
        return pending

    async def _yield_chunked_response(
        self,
        text: str,
        turn: int,
    ) -> AsyncIterator[AgentEvent]:
        """Yield response text in paragraph-sized chunks for progressive rendering.

        Splits on double-newline boundaries so the frontend can display
        each paragraph as soon as it arrives instead of waiting for the
        entire response.
        """
        yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
        paragraphs = text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            chunk = paragraph
            if i < len(paragraphs) - 1:
                chunk += "\n\n"
            if chunk:
                yield AgentEvent(
                    type=EventType.MESSAGE_CHUNK,
                    data={"chunk": chunk},
                )
                await asyncio.sleep(0)  # yield control so SSE can flush
        yield AgentEvent(type=EventType.MESSAGE_END, data={})

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

    async def _react_loop_events(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        tenant_id: str,
        first_turn_tool_choice: Any = "auto",
        retry_with_required_on_empty: bool = False,
        context: Optional[Dict[str, Any]] = None,
        user_message: str = "",
        media: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_tools: Optional[List] = None,
        _llm_client_override: Optional[Any] = None,
        routing_task: Optional["asyncio.Task"] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Unified ReAct loop implementation yielding streaming events.

        Both stream_message() and handle_message() delegate to this single
        implementation, eliminating the previous code duplication between
        the inline stream_message loop and react_loop().

        The final EXECUTION_END event carries all metadata (final_response,
        pending_approvals, token_usage, tool_calls records, etc.) so callers
        can build AgentResult or persist to memory as needed.
        """
        # --- Change A: Context window pre-flight guard ---
        CONTEXT_HARD_MIN = 16_000
        CONTEXT_WARN_BELOW = 32_000
        context_tokens = getattr(self.llm_client, "context_window", 128_000)
        if context_tokens < CONTEXT_HARD_MIN:
            yield AgentEvent(
                type=EventType.ERROR,
                data={
                    "message": f"Model context window too small: {context_tokens} tokens (minimum: {CONTEXT_HARD_MIN})"
                },
            )
            return
        if context_tokens < CONTEXT_WARN_BELOW:
            logger.warning(f"Low context window: {context_tokens} tokens")

        # Bind the real window of the model in use so trimming thresholds track
        # the model rather than the static config default.
        self._context_manager.set_context_window(context_tokens)
        # A fresh request: no measurement applies to this message list yet.
        self._context_manager.invalidate_usage()

        # Register this run so surfaces can interrupt or steer it.
        control = self._run_controls.start(tenant_id)

        start_time = time.monotonic()
        turn = 0
        all_tool_records: List[ToolCallRecord] = []
        total_usage = TokenUsage()
        pending_approvals = []
        final_response = ""
        result_status = None
        _recent_tool_names: List[str] = []  # watchdog loop detection
        _recent_tool_fingerprints: List[str] = []  # tool name + args hash
        _recent_result_hashes: List[str] = []  # result content hash
        _response_media: List[Dict[str, Any]] = []  # images for client storage
        interrupted = False

        logger.info(f"[ReAct] tenant={tenant_id}")

        yield AgentEvent(
            type=EventType.EXECUTION_START,
            data={"tenant_id": tenant_id},
        )

        # Model routing: classify once before the loop, reuse for all turns.
        # If a model-level override is provided (e.g. from fallback), use it
        # directly and skip routing. When the caller pre-started the
        # classification (see _start_routing), await that instead of issuing a
        # second round-trip here.
        routed_llm_client = _llm_client_override
        routing_score = -1
        if routed_llm_client is None and (routing_task is not None or self._model_router):
            try:
                if routing_task is not None:
                    decision = await routing_task
                else:
                    decision = await self._model_router.route(messages)
                if decision is not None:
                    routing_score = decision.score
                    routed_llm_client = self._model_router.registry.get(decision.provider)
                    if routed_llm_client:
                        logger.info(
                            f"[ReAct] ModelRouter selected provider='{decision.provider}' "
                            f"(score={decision.score}, {decision.latency_ms:.0f}ms)"
                        )
            except Exception as e:
                logger.warning(f"[ReAct] ModelRouter failed, using default LLM: {e}")

        # Enable reasoning for complex requests on the first turn. This still
        # rides on the router's score, since "is this hard enough to think
        # harder" is the same question the router already asked.
        enable_reasoning = routing_score >= self._react_config.reasoning_score_threshold
        if enable_reasoning:
            logger.info(
                f"[ReAct] Reasoning enabled (score={routing_score}, effort={self._react_config.reasoning_effort})"
            )

        # -- Planning phase --
        enable_planning = self._should_plan(context, routing_score)

        # Case 1: Pending plan from previous turn -- user is responding to it
        # Check both persistent store and in-memory fallback.
        plan_store = getattr(self, "_plan_store", None)
        pending_plan_data = None
        if plan_store is not None:
            pending_plan_data = await plan_store.pop(tenant_id)
        if pending_plan_data is None:
            pending_plan_data = self._tenant_plans.pop(tenant_id, None)

        if pending_plan_data and context:
            pending_plan_text = self._format_plan_text(pending_plan_data)

            logger.info("[ReAct] Pending plan found, injecting into prompt for LLM to handle")
            messages = await self._build_llm_messages(
                context,
                user_message,
                pending_plan=pending_plan_text,
            )
            enable_planning = False  # don't re-plan

        # Case 2: New complex request -- generate plan and present to user
        elif enable_planning:
            logger.info(f"[ReAct] Planning phase triggered (score={routing_score})")
            try:
                plan_messages = await self._build_llm_messages(
                    context,
                    user_message,
                    include_planning=True,
                )
                plan_schemas = [GENERATE_PLAN_SCHEMA, COMPLETE_TASK_SCHEMA]
                plan_response = await self._llm_call_with_retry(
                    plan_messages,
                    plan_schemas,
                    tool_choice="auto",
                    llm_client_override=routed_llm_client,
                )
                plan_data = self._extract_plan_from_response(plan_response)
                # Approval needs someone to give it. On a cron job or trigger
                # the plan would be presented to nobody and the run would stop
                # there, so those execute the plan directly.
                await_approval = self._react_config.planning_requires_approval and is_attended(
                    metadata
                )
                if plan_data and await_approval:
                    # Present plan to user, pause execution
                    plan_text = self._format_plan_text(plan_data)
                    friendly = self._format_plan_for_user(plan_data)
                    # Persist plan to survive restarts
                    if plan_store is not None:
                        await plan_store.save(tenant_id, plan_data)
                    else:
                        self._tenant_plans[tenant_id] = plan_data
                    logger.info(
                        f"[ReAct] Plan generated, awaiting approval: {plan_data.get('goal', '')}"
                    )
                    yield AgentEvent(
                        type=EventType.PLAN_GENERATED,
                        data={"plan": plan_data, "plan_text": plan_text},
                    )
                    # End this turn -- return plan as the response
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    yield AgentEvent(
                        type=EventType.EXECUTION_END,
                        data={
                            "final_response": friendly,
                            "result_status": "WAITING_FOR_APPROVAL",
                            "turns": 0,
                            "tool_calls": [],
                            "token_usage": {"input_tokens": 0, "output_tokens": 0},
                            "duration_ms": duration_ms,
                            "pending_approvals": [],
                        },
                    )
                    self._run_controls.finish(tenant_id, control)
                    return  # stop the generator -- user needs to respond

                elif plan_data:
                    # Auto-execute without approval
                    plan_text = self._format_plan_text(plan_data)
                    yield AgentEvent(
                        type=EventType.PLAN_GENERATED,
                        data={"plan": plan_data, "plan_text": plan_text},
                    )
                    logger.info(f"[ReAct] Plan auto-approved: {plan_data.get('goal', '')}")
                    messages = await self._build_llm_messages(
                        context,
                        user_message,
                        approved_plan=plan_text,
                    )
                else:
                    logger.info("[ReAct] LLM did not generate a plan, proceeding directly")
            except Exception as e:
                logger.warning(f"[ReAct] Planning phase failed, proceeding without plan: {e}")

        for turn in range(1, self._react_config.max_turns + 1):
            budget = self._react_config.react_timeout
            if budget is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > budget:
                    logger.warning(
                        f"[ReAct] Wall-clock budget exhausted after {elapsed:.1f}s (limit={budget}s)"
                    )
                    yield AgentEvent(
                        type=EventType.ERROR,
                        data={
                            "error": "Request timed out. Please try again with a simpler request."
                        },
                    )
                    break

            # Stop boundary: the user asked to stop between turns.
            if control.cancelled:
                logger.info(f"[ReAct] Interrupted at turn boundary (turn={turn})")
                interrupted = True
                break

            # Steering boundary: apply any messages the user sent mid-run so the
            # next model call sees the redirection.
            applied = self._apply_steering(control, messages)
            if applied:
                yield AgentEvent(
                    type=EventType.STEERING_APPLIED,
                    data={"messages": applied, "turn": turn},
                )

            # Context guard with summarization
            messages = await self._summarize_and_trim(messages)

            # Transcript repair before LLM call
            messages = repair_transcript(messages)

            # LLM call
            try:
                tool_choice = first_turn_tool_choice if turn == 1 else "auto"
                # Enable reasoning only on the first turn for complex requests
                extra_kwargs = {}
                if enable_reasoning and turn == 1:
                    extra_kwargs["reasoning_effort"] = self._react_config.reasoning_effort
                # Pass images only on the first turn
                if media and turn == 1:
                    extra_kwargs["media"] = media
                response = await control.race(
                    self._llm_call_with_retry(
                        messages,
                        tool_schemas,
                        tool_choice=tool_choice,
                        llm_client_override=routed_llm_client,
                        **extra_kwargs,
                    ),
                    interrupted=_INTERRUPTED,
                )
                if response is _INTERRUPTED:
                    logger.info(f"[ReAct] Interrupted during LLM call (turn={turn})")
                    interrupted = True
                    break
            except Exception as e:
                # Classify the error to decide whether to retry at model level
                from .error_classifier import LLMErrorKind, classify_llm_error

                error_kind = classify_llm_error(e)

                if error_kind == LLMErrorKind.AUTH:
                    # Auth errors are not recoverable at model level
                    from .error_classifier import error_code_for_kind

                    yield AgentEvent(
                        type=EventType.ERROR,
                        data={
                            "code": error_code_for_kind(error_kind),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    self._run_controls.finish(tenant_id, control)
                    return

                # For other errors: signal the caller to attempt model-level
                # fallback by raising with classification metadata. The caller
                # re-enters this loop, which registers a fresh control, so the
                # entry left behind here is replaced rather than leaked.
                raise _ReactLoopLLMError(e, error_kind, turn) from e

            # Accumulate token usage
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                total_usage.input_tokens += prompt_tokens
                total_usage.output_tokens += getattr(usage, "completion_tokens", 0)
                total_usage.cost_usd += getattr(usage, "cost", 0) or 0
                # The prompt-side total that actually occupied the window on this
                # round-trip -- the trigger signal for the next iteration's guard.
                self._context_manager.observe_usage(prompt_tokens)

            tool_calls = response.tool_calls

            # No tool calls → the model is done. Its text is the final answer.
            # (There is no completion tool to forget, so there is nothing to
            # recover from here: an assistant turn without tool calls simply
            # ends the loop.)
            if not tool_calls:
                final_response = (getattr(response, "content", None) or "").strip()
                self._audit.log_react_turn(
                    turn=turn,
                    tool_calls=[],
                    final_answer=True,
                    tenant_id=tenant_id,
                )
                if final_response:
                    async for event in self._yield_chunked_response(final_response, turn):
                        yield event
                break

            if tool_calls:
                # Append assistant message with tool_calls
                messages.append(self._assistant_message_from_response(response))

                # ----------------------------------------------------------
                # Intercept complete_task: handle synchronously, skip execution
                # ----------------------------------------------------------
                complete_task_result: Optional[CompleteTaskResult] = None
                _complete_task_tc_id: Optional[str] = None
                remaining_tool_calls = []
                for tc in tool_calls:
                    if tc.name == COMPLETE_TASK_TOOL_NAME:
                        try:
                            _ct_args = (
                                tc.arguments
                                if isinstance(tc.arguments, dict)
                                else json.loads(tc.arguments)
                            )
                        except (json.JSONDecodeError, TypeError):
                            _ct_args = {}
                        _ct_text = _ct_args.get("result", "")
                        if _ct_text:
                            complete_task_result = CompleteTaskResult(result=_ct_text)
                            _complete_task_tc_id = tc.id
                            logger.info(
                                f"[ReAct] turn={turn} complete_task called ({len(_ct_text)} chars)"
                            )
                        else:
                            # Missing result -- append error, let LLM retry
                            messages.append(
                                self._build_tool_result_message(
                                    tc.id,
                                    'Error: "result" argument is required for complete_task.',
                                    is_error=True,
                                )
                            )
                            remaining_tool_calls.append(tc)
                    else:
                        remaining_tool_calls.append(tc)

                # Pure complete_task with no other tools -- break immediately
                if complete_task_result and not remaining_tool_calls:
                    messages.append(
                        self._build_tool_result_message(_complete_task_tc_id, "Task completed.")
                    )
                    all_tool_records.append(
                        ToolCallRecord(
                            name=COMPLETE_TASK_TOOL_NAME,
                            args_summary={"result": complete_task_result.result[:100]},
                            duration_ms=0,
                            success=True,
                            result_status="COMPLETED",
                            result_chars=len(complete_task_result.result),
                        )
                    )
                    final_response = complete_task_result.result
                    self._audit.log_react_turn(
                        turn=turn,
                        tool_calls=[COMPLETE_TASK_TOOL_NAME],
                        final_answer=True,
                        tenant_id=tenant_id,
                    )
                    async for event in self._yield_chunked_response(final_response, turn):
                        yield event
                    break

                # complete_task was called alongside other tools -- add its result
                tool_calls = remaining_tool_calls if remaining_tool_calls else tool_calls

                # Stop boundary: the assistant message with its tool_calls is
                # already in history, so every call must still be answered even
                # though none of them will run.
                if control.cancelled:
                    logger.info(
                        f"[ReAct] Interrupted before executing {len(tool_calls)} "
                        f"tool call(s) (turn={turn})"
                    )
                    self._compensate_pending_tool_calls(tool_calls, messages)
                    interrupted = True
                    break

                # ----------------------------------------------------------
                # Validate tool calls against the schema sent to the LLM.
                # Reject hallucinated tool names and schema-mismatched args.
                # Rejected calls become error tool_results so the model can
                # self-correct on the next turn instead of crashing the loop.
                # ----------------------------------------------------------
                validator = ToolSchemaValidator.from_openai_tools(tool_schemas)
                validated_tool_calls = []
                for tc in tool_calls:
                    # complete_task is synthetic; always allow.
                    if tc.name == COMPLETE_TASK_TOOL_NAME:
                        validated_tool_calls.append(tc)
                        continue
                    try:
                        args_for_validation = (
                            tc.arguments
                            if isinstance(tc.arguments, dict)
                            else json.loads(tc.arguments or "{}")
                        )
                    except (json.JSONDecodeError, TypeError):
                        args_for_validation = None
                    if args_for_validation is None:
                        reason = "arguments_not_json"
                        details: Dict[str, Any] = {"name": tc.name}
                    else:
                        vr = validator.validate(tc.name, args_for_validation)
                        if vr.ok:
                            validated_tool_calls.append(tc)
                            continue
                        reason = vr.reason
                        details = vr.details or {}
                    logger.warning(
                        "[ReAct] Rejecting tool call %r: %s %s",
                        tc.name,
                        reason,
                        details,
                    )
                    self._audit.log_tool_execution(
                        tool_name=tc.name,
                        args_summary={"rejected_reason": reason},
                        success=False,
                        duration_ms=0,
                        error=f"schema_validation:{reason}",
                        tenant_id=tenant_id,
                    )
                    messages.append(
                        self._build_tool_result_message(
                            tc.id,
                            f"Error: tool call rejected by schema validation "
                            f"({reason}). details={json.dumps(details, default=str)[:256]}. "
                            f"Allowed tools: {', '.join(validator.known_names[:20])}",
                            is_error=True,
                        )
                    )
                    all_tool_records.append(
                        ToolCallRecord(
                            name=tc.name,
                            args_summary={"rejected_reason": reason},
                            duration_ms=0,
                            success=False,
                            result_status="REJECTED",
                            result_chars=0,
                        )
                    )

                if not validated_tool_calls:
                    # All calls rejected — continue the loop so the model
                    # can retry with valid tools.  Guarded by max_turns.
                    continue
                tool_calls = validated_tool_calls

                tool_names = [tc.name for tc in tool_calls]
                logger.info(f"[ReAct] turn={turn} calling: {', '.join(tool_names)}")

                # Emit a brief acknowledgment before tool execution so the
                # user sees "Looking into that..." while tools run.
                ack = _tool_acknowledgment(tool_names, turn)
                if ack:
                    yield AgentEvent(type=EventType.ACKNOWLEDGMENT, data={"text": ack})

                # Yield tool call start events
                for tc in tool_calls:
                    yield AgentEvent(
                        type=EventType.TOOL_CALL_START,
                        data={"tool_name": tc.name, "call_id": tc.id},
                    )

                # Execute all tool calls concurrently with per-tool timing.
                # When speculative tasks exist (image requests), check if a
                # matching result is already available before executing.
                speculative = (context or {}).get("_speculative_tasks", {})

                async def _timed_execute(tc):
                    t0 = time.monotonic()

                    # Try to reuse a speculative result
                    if speculative and tc.name == "google_search":
                        try:
                            args = (
                                tc.arguments
                                if isinstance(tc.arguments, dict)
                                else json.loads(tc.arguments)
                            )
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        search_type = args.get("search_type", "web")
                        spec_key = f"google_search:{search_type}"
                        spec_task = speculative.pop(spec_key, None)
                        if spec_task is not None:
                            try:
                                result = await spec_task
                                if result is not None:
                                    elapsed = int((time.monotonic() - t0) * 1000)
                                    logger.info(
                                        f"[Speculative] ♻️  Reused {spec_key} "
                                        f"(waited {elapsed}ms for pre-started task)"
                                    )
                                    return TimedResult(result=result, duration_ms=elapsed)
                            except Exception as e:
                                logger.info(f"[Speculative] {spec_key} failed, falling back: {e}")

                    # Normal execution path
                    try:
                        r = await control.race(
                            self._execute_with_timeout(
                                tc,
                                tenant_id,
                                metadata=metadata,
                                request_tools=request_tools,
                                request_context=context,
                            ),
                            interrupted=_INTERRUPTED,
                        )
                        if r is _INTERRUPTED:
                            r = InterruptedError("Interrupted by user during execution.")
                    except BaseException as exc:
                        return TimedResult(
                            result=exc, duration_ms=int((time.monotonic() - t0) * 1000)
                        )
                    return TimedResult(result=r, duration_ms=int((time.monotonic() - t0) * 1000))

                # Execute tools eagerly: yield results as each completes
                # instead of waiting for all to finish (asyncio.gather).
                async def _timed_with_index(idx, tc):
                    return (idx, await _timed_execute(tc))

                # Token attribution for this turn
                turn_tokens = None
                if usage:
                    turn_tokens = TokenUsage(
                        input_tokens=getattr(usage, "prompt_tokens", 0),
                        output_tokens=getattr(usage, "completion_tokens", 0),
                    )

                loop_broken = False
                loop_broken_text = None

                # Ordered results for watchdog/passthrough downstream
                timed_results = [None] * len(tool_calls)

                for _coro in asyncio.as_completed(
                    [_timed_with_index(i, tc) for i, tc in enumerate(tool_calls)]
                ):
                    _idx, timed = await _coro
                    tc = tool_calls[_idx]
                    timed_results[_idx] = timed
                    tc_name = tc.name
                    is_agent = self._is_agent_tool(tc_name)
                    kind = "agent" if is_agent else "tool"
                    # Unwrap TimedResult
                    if isinstance(timed, TimedResult):
                        result = timed.result
                        tc_duration = timed.duration_ms
                    else:
                        result = timed
                        tc_duration = 0

                    try:
                        args_summary = (
                            tc.arguments
                            if isinstance(tc.arguments, dict)
                            else json.loads(tc.arguments)
                        )
                    except (json.JSONDecodeError, TypeError):
                        args_summary = {}
                    args_summary = {k: str(v)[:100] for k, v in args_summary.items()}

                    if isinstance(result, BaseException):
                        logger.warning(f"[ReAct]   {kind}={tc_name} ERROR: {result}")
                        error_text = f"Error executing {tc_name}: {result}"
                        messages.append(
                            self._build_tool_result_message(tc.id, error_text, is_error=True)
                        )
                        all_tool_records.append(
                            ToolCallRecord(
                                name=tc_name,
                                args_summary=args_summary,
                                duration_ms=tc_duration,
                                success=False,
                                result_chars=len(error_text),
                                token_attribution=turn_tokens,
                            )
                        )
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name,
                                "call_id": tc.id,
                                "kind": kind,
                                "success": False,
                                "error": str(result),
                                "result_preview": error_text[:240],
                            },
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name,
                            args_summary=args_summary,
                            success=False,
                            duration_ms=tc_duration,
                            error=str(result),
                            tenant_id=tenant_id,
                        )

                    elif isinstance(result, AgentToolResult) and not result.completed:
                        logger.info(f"[ReAct]   {kind}={tc_name} WAITING")
                        if result.agent:
                            await self.agent_pool.add_agent(result.agent)
                        if result.approval_request:
                            pending_approvals.append(result.approval_request)
                        waiting_text = result.result_text or "Agent is waiting for input."
                        messages.append(self._build_tool_result_message(tc.id, waiting_text))
                        waiting_status = (
                            "WAITING_FOR_APPROVAL"
                            if result.approval_request
                            else "WAITING_FOR_INPUT"
                        )
                        all_tool_records.append(
                            ToolCallRecord(
                                name=tc_name,
                                args_summary=args_summary,
                                duration_ms=tc_duration,
                                success=True,
                                result_status=waiting_status,
                                result_chars=len(waiting_text),
                                token_attribution=turn_tokens,
                            )
                        )
                        tool_trace: List = []
                        if isinstance(result.metadata, dict):
                            tool_trace = result.metadata.get("tool_trace") or []
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name,
                                "call_id": tc.id,
                                "kind": "agent",
                                "success": True,
                                "waiting": True,
                                "status": waiting_status,
                                "result_preview": waiting_text[:240],
                                "tool_trace": tool_trace,
                            },
                        )
                        yield AgentEvent(
                            type=EventType.STATE_CHANGE,
                            data={"agent_type": tc_name, "status": waiting_status},
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name,
                            args_summary=args_summary,
                            success=True,
                            duration_ms=tc_duration,
                            tenant_id=tenant_id,
                        )
                        loop_broken = True
                        loop_broken_text = waiting_text

                    else:
                        # Extract text and optional media from the result
                        result_media = []
                        if isinstance(result, ToolOutput):
                            result_text = result.text
                            result_media = result.media or []
                            tool_trace = []
                        elif isinstance(result, AgentToolResult):
                            result_text = result.result_text
                            r_meta = result.metadata if isinstance(result.metadata, dict) else {}
                            tool_trace = r_meta.get("tool_trace") or []
                            # Collect media forwarded from agent's internal tools
                            agent_media = r_meta.get("media") or []
                            result_media = agent_media
                        else:
                            result_text = str(result) if result is not None else ""
                            tool_trace = []
                        result_chars_original = len(result_text)
                        original_len = len(result_text)
                        # Hard cap on tool result size
                        result_text = self._cap_tool_result(result_text)
                        result_text = self._context_manager.truncate_tool_result(result_text)
                        if is_agent and len(result_text) > 2000:
                            result_text = (
                                result_text[:1500]
                                + f"\n...[truncated from {original_len} to 1500 chars]"
                            )
                        elif len(result_text) < original_len:
                            result_text += (
                                f"\n...[truncated from {original_len} to {len(result_text)} chars]"
                            )
                        logger.info(
                            f"[ReAct]   {kind}={tc_name} OK ({len(result_text)} chars, media={len(result_media)})"
                        )
                        messages.append(
                            self._build_tool_result_message(
                                tc.id,
                                result_text,
                                media=result_media,
                            )
                        )

                        # Collect media for the final response:
                        # - for_storage=True media (images) for client persistence
                        # - inline_cards for frontend card rendering
                        for m in result_media:
                            meta = m.get("metadata", {})
                            if meta.get("for_storage") or m.get("type") == "inline_cards":
                                _append_unique_media(_response_media, [m])

                        all_tool_records.append(
                            ToolCallRecord(
                                name=tc_name,
                                args_summary=args_summary,
                                duration_ms=tc_duration,
                                success=True,
                                result_status="COMPLETED"
                                if isinstance(result, AgentToolResult)
                                else None,
                                result_chars=result_chars_original,
                                token_attribution=turn_tokens,
                            )
                        )
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name,
                                "call_id": tc.id,
                                "kind": kind,
                                "success": True,
                                "result_preview": result_text[:240],
                                "tool_trace": tool_trace,
                            },
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name,
                            args_summary=args_summary,
                            success=True,
                            duration_ms=tc_duration,
                            tenant_id=tenant_id,
                        )

                # complete_task was called alongside other tools -- add its result
                # AFTER all other tools' results have been appended to messages
                if complete_task_result:
                    messages.append(
                        self._build_tool_result_message(_complete_task_tc_id, "Task completed.")
                    )
                    all_tool_records.append(
                        ToolCallRecord(
                            name=COMPLETE_TASK_TOOL_NAME,
                            args_summary={"result": complete_task_result.result[:100]},
                            duration_ms=0,
                            success=True,
                            result_status="COMPLETED",
                            result_chars=len(complete_task_result.result),
                        )
                    )
                    final_response = complete_task_result.result
                    self._audit.log_react_turn(
                        turn=turn,
                        tool_calls=tool_names + [COMPLETE_TASK_TOOL_NAME],
                        final_answer=True,
                        tenant_id=tenant_id,
                    )
                    async for event in self._yield_chunked_response(final_response, turn):
                        yield event
                    break

                # Watchdog: detect loops (enhanced with args + result hashes)
                import hashlib

                for tc, timed in zip(tool_calls, timed_results):
                    _recent_tool_names.append(tc.name)
                    # Fingerprint: tool name + serialized args
                    try:
                        args_str = json.dumps(
                            tc.arguments
                            if isinstance(tc.arguments, dict)
                            else json.loads(tc.arguments),
                            sort_keys=True,
                        )
                    except (json.JSONDecodeError, TypeError):
                        args_str = str(tc.arguments)
                    fp = f"{tc.name}:{hashlib.md5(args_str.encode()).hexdigest()[:8]}"
                    _recent_tool_fingerprints.append(fp)
                    # Result hash
                    result_val = timed.result if isinstance(timed, TimedResult) else timed
                    result_str = str(result_val) if result_val is not None else ""
                    rh = hashlib.md5(result_str[:2000].encode()).hexdigest()[:8]
                    _recent_result_hashes.append(rh)

                loop_desc = self._detect_loop(
                    _recent_tool_names,
                    _recent_tool_fingerprints,
                    _recent_result_hashes,
                )
                if loop_desc:
                    logger.warning(f"[ReAct] {loop_desc}")
                    final_response = "I noticed I was repeating the same actions without making progress. Let me provide what I have so far."
                    async for event in self._yield_chunked_response(final_response, turn):
                        yield event
                    break

                # Audit: log turn summary
                self._audit.log_react_turn(
                    turn=turn,
                    tool_calls=tool_names,
                    final_answer=False,
                    tenant_id=tenant_id,
                )

                # Stop boundary: tool results are recorded, so the transcript is
                # complete and it is safe to unwind here.
                if control.cancelled:
                    logger.info(f"[ReAct] Interrupted after tool execution (turn={turn})")
                    interrupted = True
                    break

                if loop_broken:
                    final_response = loop_broken_text or ""
                    result_status = (
                        "WAITING_FOR_APPROVAL" if pending_approvals else "WAITING_FOR_INPUT"
                    )
                    if pending_approvals:
                        pending_approvals = collect_batch_approvals(pending_approvals)
                    if loop_broken_text:
                        async for event in self._yield_chunked_response(loop_broken_text, turn):
                            yield event
                    break

                # Agent passthrough: single completed agent-tool skips LLM re-summary
                _first_result = (
                    timed_results[0].result
                    if isinstance(timed_results[0], TimedResult)
                    else timed_results[0]
                )
                if (
                    len(tool_calls) == 1
                    and self._is_agent_tool(tool_calls[0].name)
                    and isinstance(_first_result, AgentToolResult)
                    and _first_result.completed
                ):
                    agent_text = _first_result.result_text
                    logger.info(
                        f"[ReAct] turn={turn} agent_passthrough "
                        f"({len(agent_text)} chars from {tool_calls[0].name})"
                    )
                    final_response = agent_text
                    async for event in self._yield_chunked_response(agent_text, turn):
                        yield event
                    break

        else:
            # max_turns reached: ask LLM for summary without tools
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have used all available turns. Please provide your best "
                        "final answer based on the information gathered so far."
                    ),
                }
            )
            try:
                response = await self._llm_call_with_retry(
                    messages,
                    tool_schemas=None,
                    llm_client_override=routed_llm_client,
                )
                final_text = response.content or ""
                usage = getattr(response, "usage", None)
                if usage:
                    total_usage.input_tokens += getattr(usage, "prompt_tokens", 0)
                    total_usage.output_tokens += getattr(usage, "completion_tokens", 0)
                    total_usage.cost_usd += getattr(usage, "cost", 0) or 0
            except Exception as e:
                logger.warning(f"[ReAct] Summary call after max_turns failed: {e}")
                final_text = "I was unable to complete the request within the allowed turns."

            final_response = final_text
            async for event in self._yield_chunked_response(final_text, turn):
                yield event

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Cancel any speculative tasks that were not consumed
        speculative = (context or {}).get("_speculative_tasks", {})
        for key, task in speculative.items():
            if task and not task.done():
                task.cancel()
                logger.info(f"[Speculative] Cancelled unused task: {key}")

        # This run is no longer signalable.
        self._run_controls.finish(tenant_id, control)

        if interrupted:
            if not final_response:
                final_response = "Stopped."
            if result_status is None:
                result_status = "INTERRUPTED"
            yield AgentEvent(
                type=EventType.INTERRUPTED,
                data={
                    "reason": control.cancel_reason,
                    "turns": turn,
                    "tool_calls_count": len(all_tool_records),
                },
            )

        yield AgentEvent(
            type=EventType.EXECUTION_END,
            data={
                "duration_ms": duration_ms,
                "turns": turn,
                "tool_calls_count": len(all_tool_records),
                "final_response": final_response,
                "result_status": result_status,
                "interrupted": interrupted,
                "pending_approvals": pending_approvals,
                "media": _response_media or None,
                "token_usage": {
                    "input_tokens": total_usage.input_tokens,
                    "output_tokens": total_usage.output_tokens,
                    "cost_usd": round(total_usage.cost_usd, 6),
                },
                "tool_calls": [dataclasses.asdict(r) for r in all_tool_records],
            },
        )

    async def _save_tool_call_history(
        self,
        tenant_id: str,
        tool_calls: list,
    ) -> None:
        """Persist tool call records to the database (fire-and-forget)."""
        if not self.database or not tool_calls:
            return
        try:
            from ..builtin_agents.tools.action_history import save_tool_call_history

            await save_tool_call_history(self.database, tenant_id, tool_calls)
        except Exception as e:
            logger.warning(f"Failed to save tool call history: {e}")

    async def _summarize_and_trim(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize old messages via LLM before trimming, preserving context.

        If context is within threshold, returns messages unchanged.
        Otherwise, splits messages into old/recent, summarizes old via LLM,
        and replaces them with a single summary message.
        Falls back to simple trim if summarization fails.
        """
        split = self._context_manager.split_for_summarization(messages)
        if split is None:
            return messages

        system_msgs, old_msgs, recent_msgs = split

        # Token-budget-aware truncation: allocate a per-message budget
        # based on the summarizer's total budget rather than a fixed
        # character count.  This preserves more content from tool results
        # that carry important data in the second half.
        SUMMARIZER_BUDGET_CHARS = 12_000  # ~3k tokens for the summarizer input
        per_msg_budget = max(200, SUMMARIZER_BUDGET_CHARS // max(len(old_msgs), 1))

        old_text_parts = []
        for msg in old_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                if len(content) > per_msg_budget:
                    # Keep head and tail to preserve both context and conclusions
                    head_len = int(per_msg_budget * 0.6)
                    tail_len = int(per_msg_budget * 0.35)
                    content = (
                        content[:head_len]
                        + f"\n...[{len(content) - head_len - tail_len} chars omitted]...\n"
                        + content[-tail_len:]
                    )
                old_text_parts.append(f"{role}: {content}")

        if not old_text_parts:
            return self._context_manager.trim_if_needed(messages)

        old_text = "\n".join(old_text_parts)
        # Final safety cap for the summarizer input
        if len(old_text) > SUMMARIZER_BUDGET_CHARS:
            old_text = old_text[:SUMMARIZER_BUDGET_CHARS] + "\n...[truncated]"

        try:
            summary_response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation excerpt. Preserve:\n"
                            "- All specific data values (names, dates, numbers, IDs, URLs)\n"
                            "- Tool call results and their key findings\n"
                            "- Decisions made and actions taken\n"
                            "Keep the summary concise but factual. Use bullet points for structured data."
                        ),
                    },
                    {"role": "user", "content": old_text},
                ],
            )
            summary = (summary_response.content or "").strip()
            if summary:
                logger.info(
                    f"[Context] Summarized {len(old_msgs)} old messages "
                    f"({len(old_text)} chars -> {len(summary)} chars)"
                )
                # The message list is being rewritten, so the last reported
                # prompt-token count no longer describes what will be sent.
                self._context_manager.invalidate_usage()
                return self._context_manager.build_summarized_messages(
                    system_msgs,
                    summary,
                    recent_msgs,
                )
        except Exception as e:
            logger.warning(f"[Context] Summarization failed, falling back to trim: {e}")

        return self._context_manager.trim_if_needed(messages)

    def _build_tool_result_message(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
        media: list = None,
    ) -> Dict[str, Any]:
        """Build a tool result message for the LLM messages list.

        When *media* is provided (e.g. thumbnail images from an image search),
        the content is formatted as a multimodal content array so that
        vision-capable LLMs can inspect the images.
        """
        if is_error:
            content = f"[ERROR] {content}"

        if not media:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }

        # Build multimodal content: text + image_url blocks
        parts: list = [{"type": "text", "text": content}]
        for item in media:
            if item.get("type") == "image":
                data = item.get("data", "")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data, "detail": "low"},
                    }
                )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": parts,
        }

    @staticmethod
    def _assistant_message_from_response(response: Any) -> Dict[str, Any]:
        """Convert LLMResponse to dict for the messages list."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(response, "content", None),
        }
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return msg

    @staticmethod
    def _detect_loop(
        tool_history: list,
        fingerprint_history: Optional[list] = None,
        result_hash_history: Optional[list] = None,
    ) -> Optional[str]:
        """Detect if the LLM is repeating the same actions.

        Checks three layers:
        1. Same tool+args called consecutively (exact repeat)
        2. Different tools but identical results (no progress)
        3. Tool name pattern cycles (A-B-A-B)
        """
        # Layer 1: Same tool + same args 3 times (exact repeat — strongest signal)
        if fingerprint_history and len(fingerprint_history) >= 3:
            if len(set(fingerprint_history[-3:])) == 1:
                return f"Exact repeat: {fingerprint_history[-1]} called 3 times with same args"

        # Layer 2: Consecutive identical results (no progress)
        if result_hash_history and len(result_hash_history) >= 2:
            if result_hash_history[-1] == result_hash_history[-2]:
                # Same tool producing same output — wasting tokens
                if fingerprint_history and len(fingerprint_history) >= 2:
                    if fingerprint_history[-1] == fingerprint_history[-2]:
                        return "No progress: same tool returned identical results twice"

        # Layer 3: Tool name pattern cycles (fallback to original logic)
        if len(tool_history) >= 3 and len(set(tool_history[-3:])) == 1:
            # Only flag if we don't have fingerprint data or fingerprints also match
            if not fingerprint_history or len(set(fingerprint_history[-3:])) == 1:
                return f"Loop detected: {tool_history[-1]} called 3 times consecutively"

        for cycle_len in range(2, 5):
            needed = cycle_len * 2
            if len(tool_history) < needed:
                continue
            tail = tool_history[-needed:]
            cycle = tail[:cycle_len]
            if all(tail[i] == cycle[i % cycle_len] for i in range(needed)):
                pattern = "↔".join(cycle)
                return f"Cycle detected: {pattern} repeated {needed // cycle_len} times"

        return None
