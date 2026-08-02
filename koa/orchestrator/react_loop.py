"""ReAct loop mixin for the Orchestrator.

Contains the core ReAct (Reasoning + Acting) loop implementation
and its helper methods.
"""

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from ..streaming.models import AgentEvent, EventType
from . import watchdog
from .approval import collect_batch_approvals
from .error_classifier import LLMErrorKind
from .planning import PlanningMixin, PlanOutcome
from .run_state import RunState
from .tool_execution import ToolExecutionMixin, TurnOutcome
from .transcript_repair import repair_transcript
from .turn_gate import TurnGateMixin

logger = logging.getLogger(__name__)

#: Sentinel distinguishing "the run was interrupted" from a real (possibly
#: falsy) result when racing an awaitable against the cancel signal.
_INTERRUPTED = object()

#: A window this small cannot hold a system prompt plus a few tool results,
#: so a run against it fails before starting rather than partway through.
_CONTEXT_HARD_MIN = 16_000
_CONTEXT_WARN_BELOW = 32_000


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


class _ReactLoopAuthError(_ReactLoopLLMError):
    """The provider rejected our credentials.

    Separate from its parent because it is the one LLM failure worth
    reporting to the user immediately: switching models cannot fix
    credentials, so the fallback the caller would otherwise attempt only
    delays the same answer.
    """


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


class ReactLoopMixin(ToolExecutionMixin, PlanningMixin, TurnGateMixin):
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

    async def _persist_transcript(
        self,
        context: Optional[Dict[str, Any]],
        tenant_id: str,
        messages: List[Dict[str, Any]],
        user_message: str,
        metadata: Optional[Dict[str, Any]],
        turn: int,
        status: str = "running",
    ) -> None:
        """Checkpoint the run so it can outlive this process.

        Keyed by the request id already threaded through the audit log. A run
        with no request id (a direct loop call in a test) is simply not stored.
        """
        store = getattr(self, "_transcript_store", None)
        if store is None or not store.enabled:
            return
        run_id = (context or {}).get("request_id")
        if not run_id:
            return
        await store.save(
            run_id,
            tenant_id,
            messages,
            user_message=user_message,
            metadata=metadata or {},
            turn=turn,
            status=status,
        )

    async def _finish_transcript(
        self,
        context: Optional[Dict[str, Any]],
        status: str,
    ) -> None:
        """Move a run's stored transcript to its final state."""
        store = getattr(self, "_transcript_store", None)
        if store is None or not store.enabled:
            return
        run_id = (context or {}).get("request_id")
        if run_id:
            await store.mark(run_id, status)

    async def _route_for_run(
        self,
        state: RunState,
        messages: List[Dict[str, Any]],
        override: Optional[Any],
        routing_task: Optional["asyncio.Task"],
    ) -> None:
        """Pick the model for this run and decide whether to let it think.

        Classification happens once, before the first turn, and every turn
        reuses the result: the request does not get harder halfway through,
        and re-asking would add a round-trip per turn.
        """
        state.llm_client = override
        if override is None and (routing_task is not None or self._model_router):
            try:
                # The caller may have started classification in parallel with
                # other setup (see _start_routing); await that rather than
                # issuing a second identical call here.
                if routing_task is not None:
                    decision = await routing_task
                else:
                    decision = await self._model_router.route(messages)
                if decision is not None:
                    state.routing_score = decision.score
                    state.llm_client = self._model_router.registry.get(decision.provider)
                    if state.llm_client:
                        logger.info(
                            f"[ReAct] ModelRouter selected provider='{decision.provider}' "
                            f"(score={decision.score}, {decision.latency_ms:.0f}ms)"
                        )
            except Exception as e:
                logger.warning(f"[ReAct] ModelRouter failed, using default LLM: {e}")

        # "Is this hard enough to think harder about" is the same question the
        # router already asked, so the answer rides on its score.
        state.enable_reasoning = (
            state.routing_score >= self._react_config.reasoning_score_threshold
        )
        if state.enable_reasoning:
            logger.info(
                f"[ReAct] Reasoning enabled (score={state.routing_score}, "
                f"effort={self._react_config.reasoning_effort})"
            )

    async def _call_model(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        state: RunState,
        control: Any,
        *,
        tool_choice: Any,
        media: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """One model round-trip, or _INTERRUPTED if the user stopped it.

        Raises _ReactLoopLLMError so the caller can try a different model.
        Auth failures raise _ReactLoopAuthError instead: no other model will
        accept credentials this one rejected, so retrying only costs time.
        """
        extra_kwargs: Dict[str, Any] = {}
        # Reasoning and images both belong to the opening turn: the first is
        # about the request, and later turns are about tool output.
        if state.turn == 1:
            if state.enable_reasoning:
                extra_kwargs["reasoning_effort"] = self._react_config.reasoning_effort
            if media:
                extra_kwargs["media"] = media

        try:
            return await control.race(
                self._llm_call_with_retry(
                    messages,
                    tool_schemas,
                    tool_choice=tool_choice,
                    llm_client_override=state.llm_client,
                    **extra_kwargs,
                ),
                interrupted=_INTERRUPTED,
            )
        except Exception as e:
            from .error_classifier import classify_llm_error

            error_kind = classify_llm_error(e)
            if error_kind == LLMErrorKind.AUTH:
                raise _ReactLoopAuthError(e, error_kind, state.turn) from e
            raise _ReactLoopLLMError(e, error_kind, state.turn) from e

    async def _summarize_after_max_turns(
        self,
        messages: List[Dict[str, Any]],
        state: RunState,
    ) -> str:
        """Get a closing answer once the turn budget is spent."""
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
                llm_client_override=state.llm_client,
            )
        except Exception as e:
            logger.warning(f"[ReAct] Summary call after max_turns failed: {e}")
            return "I was unable to complete the request within the allowed turns."
        state.add_usage(getattr(response, "usage", None))
        return response.content or ""

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

        The final EXECUTION_END event carries all metadata (state.final_response,
        state.pending_approvals, token_usage, tool_calls records, etc.) so callers
        can build AgentResult or persist to memory as needed.
        """
        # A model whose window cannot hold a system prompt plus a few tool
        # results has no useful run to attempt, so this fails before starting
        # rather than partway through.
        context_tokens = getattr(self.llm_client, "context_window", 128_000)
        if context_tokens < _CONTEXT_HARD_MIN:
            yield AgentEvent(
                type=EventType.ERROR,
                data={
                    "message": (
                        f"Model context window too small: {context_tokens} tokens "
                        f"(minimum: {_CONTEXT_HARD_MIN})"
                    )
                },
            )
            return
        if context_tokens < _CONTEXT_WARN_BELOW:
            logger.warning(f"Low context window: {context_tokens} tokens")

        # Bind the real window of the model in use so trimming thresholds track
        # the model rather than the static config default.
        self._context_manager.set_context_window(context_tokens)
        # A fresh request: no measurement applies to this message list yet.
        self._context_manager.invalidate_usage()

        # Register this run so surfaces can interrupt or steer it.
        control = self._run_controls.start(tenant_id)

        state = RunState(tenant_id=tenant_id)

        logger.info(f"[ReAct] tenant={tenant_id}")

        yield AgentEvent(
            type=EventType.EXECUTION_START,
            data={"tenant_id": tenant_id},
        )

        # Model routing: classify once before the loop, reuse for all turns.
        await self._route_for_run(state, messages, _llm_client_override, routing_task)

        # -- Planning phase --
        plan_outcome = PlanOutcome()
        async for event in self._plan_phase(
            state,
            plan_outcome,
            context=context,
            user_message=user_message,
            metadata=metadata,
            enable_planning=self._should_plan(context, state.routing_score),
        ):
            yield event

        if plan_outcome.messages is not None:
            messages = plan_outcome.messages

        if plan_outcome.awaiting_approval:
            yield AgentEvent(
                type=EventType.EXECUTION_END,
                data=state.execution_end_payload(),
            )
            self._run_controls.finish(tenant_id, control)
            return  # stop the generator -- user needs to respond

        for turn in range(1, self._react_config.max_turns + 1):
            state.turn = turn
            budget = self._react_config.react_timeout
            if budget is not None:
                elapsed = state.elapsed_seconds
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
                state.interrupted = True
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
                response = await self._call_model(
                    messages,
                    tool_schemas,
                    state,
                    control,
                    tool_choice=first_turn_tool_choice if turn == 1 else "auto",
                    media=media,
                )
            except _ReactLoopAuthError as e:
                from .error_classifier import error_code_for_kind

                yield AgentEvent(
                    type=EventType.ERROR,
                    data={
                        "code": error_code_for_kind(e.error_kind),
                        "error": str(e.original),
                        "error_type": type(e.original).__name__,
                    },
                )
                self._run_controls.finish(tenant_id, control)
                return

            if response is _INTERRUPTED:
                logger.info(f"[ReAct] Interrupted during LLM call (turn={turn})")
                state.interrupted = True
                break

            # Accumulate token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = state.add_usage(usage)
            if prompt_tokens:
                # The prompt-side total that actually occupied the window on this
                # round-trip -- the trigger signal for the next iteration's guard.
                self._context_manager.observe_usage(prompt_tokens)

            tool_calls = response.tool_calls

            # No tool calls → the model is done. Its text is the final answer.
            # (There is no completion tool to forget, so there is nothing to
            # recover from here: an assistant turn without tool calls simply
            # ends the loop.)
            if not tool_calls:
                state.final_response = (getattr(response, "content", None) or "").strip()
                self._audit.log_react_turn(
                    turn=turn,
                    tool_calls=[],
                    final_answer=True,
                    tenant_id=tenant_id,
                )
                if state.final_response:
                    async for event in self._yield_chunked_response(state.final_response, turn):
                        yield event
                break

            # Append assistant message with tool_calls
            messages.append(self._assistant_message_from_response(response))

            # ----------------------------------------------------------
            # complete_task never executes: it is the model saying it is
            # done. Pull it out, then screen what is left.
            # ----------------------------------------------------------
            intercept = self._intercept_complete_task(tool_calls, messages)

            if intercept.ends_turn_alone:
                self._settle_complete_task(intercept, messages, state)
                async for event in self._yield_chunked_response(state.final_response, turn):
                    yield event
                break

            tool_calls = intercept.remaining or tool_calls

            # Stop boundary: the assistant message with its tool_calls is
            # already in history, so every call must still be answered even
            # though none of them will run.
            if control.cancelled:
                logger.info(
                    f"[ReAct] Interrupted before executing {len(tool_calls)} "
                    f"tool call(s) (turn={turn})"
                )
                self._compensate_pending_tool_calls(tool_calls, messages)
                state.interrupted = True
                break

            validated_tool_calls = self._validate_tool_calls(
                tool_calls, tool_schemas, messages, state
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

            # Token attribution for this turn
            turn_tokens = state.turn_tokens(usage)

            outcome = TurnOutcome()
            async for event in self._run_tool_calls(
                tool_calls,
                messages,
                state,
                control,
                outcome,
                metadata=metadata,
                request_tools=request_tools,
                context=context,
                turn_tokens=turn_tokens,
                interrupted_sentinel=_INTERRUPTED,
            ):
                yield event
            timed_results = outcome.timed_results

            # complete_task was called alongside other tools -- add its result
            # AFTER all other tools' results have been appended to messages
            if intercept.result:
                self._settle_complete_task(intercept, messages, state, tool_names)
                async for event in self._yield_chunked_response(state.final_response, turn):
                    yield event
                break

            # Watchdog: a run that keeps making the same call is not going to
            # reach a different answer by making it again.
            loop_desc = watchdog.verdict(state, tool_calls, timed_results)
            if loop_desc:
                logger.warning(f"[ReAct] {loop_desc}")
                state.final_response = watchdog.STUCK_MESSAGE
                async for event in self._yield_chunked_response(state.final_response, turn):
                    yield event
                break

            # Audit: log turn summary
            self._audit.log_react_turn(
                turn=turn,
                tool_calls=tool_names,
                final_answer=False,
                tenant_id=tenant_id,
            )

            # Checkpoint the transcript now that this round's tool results
            # are recorded. Every persisted copy therefore describes work
            # that actually completed, so a resume never re-runs a tool
            # whose result is already in the messages.
            await self._persist_transcript(
                context, tenant_id, messages, user_message, metadata, turn
            )

            # Stop boundary: tool results are recorded, so the transcript is
            # complete and it is safe to unwind here.
            if control.cancelled:
                logger.info(f"[ReAct] Interrupted after tool execution (turn={turn})")
                state.interrupted = True
                break

            if outcome.waiting:
                # An agent needs the user before it can go further, so the run
                # ends here holding whatever it was waiting on.
                state.final_response = outcome.waiting_text or ""
                state.result_status = (
                    "WAITING_FOR_APPROVAL" if state.pending_approvals else "WAITING_FOR_INPUT"
                )
                if state.pending_approvals:
                    state.pending_approvals = collect_batch_approvals(state.pending_approvals)
                if outcome.waiting_text:
                    async for event in self._yield_chunked_response(outcome.waiting_text, turn):
                        yield event
                break

            # One agent, finished, nothing else called: its answer is already
            # the answer, and asking the model to restate it costs a round-trip
            # and loses detail.
            passthrough = self._agent_passthrough_text(tool_calls, timed_results)
            if passthrough is not None:
                logger.info(
                    f"[ReAct] turn={turn} agent_passthrough "
                    f"({len(passthrough)} chars from {tool_calls[0].name})"
                )
                state.final_response = passthrough
                async for event in self._yield_chunked_response(passthrough, turn):
                    yield event
                break

        else:
            # Every turn was used and the model still had not finished. Ask it
            # once more with no tools, so it has to answer from what it has
            # rather than reaching for another call it cannot make.
            state.final_response = await self._summarize_after_max_turns(messages, state)
            async for event in self._yield_chunked_response(state.final_response, turn):
                yield event

        # Cancel any speculative tasks that were not consumed
        speculative = (context or {}).get("_speculative_tasks", {})
        for key, task in speculative.items():
            if task and not task.done():
                task.cancel()
                logger.info(f"[Speculative] Cancelled unused task: {key}")

        # This run is no longer signalable.
        self._run_controls.finish(tenant_id, control)

        # An interrupted run keeps its transcript: the user stopped it, and it
        # may be worth continuing. A finished one is terminal and gets pruned.
        await self._finish_transcript(context, "suspended" if state.interrupted else "completed")

        if state.interrupted:
            # A run the user stopped still owes them a reply, and "Stopped."
            # is a truer one than the empty string it would otherwise carry.
            if not state.final_response:
                state.final_response = "Stopped."
            if state.result_status is None:
                state.result_status = "INTERRUPTED"
            yield AgentEvent(
                type=EventType.INTERRUPTED,
                data={
                    "reason": control.cancel_reason,
                    "turns": state.turn,
                    "tool_calls_count": len(state.tool_records),
                },
            )

        yield AgentEvent(type=EventType.EXECUTION_END, data=state.execution_end_payload())

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

