"""ReAct loop mixin for the Orchestrator.

Contains the core ReAct (Reasoning + Acting) loop implementation
and its helper methods.
"""

import asyncio
import dataclasses
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from ..streaming.models import AgentEvent, EventType
from .agent_tool import AgentToolResult
from .approval import collect_batch_approvals
from .react_config import (
    ToolCallRecord, TokenUsage,
    COMPLETE_TASK_TOOL_NAME, COMPLETE_TASK_SCHEMA, CompleteTaskResult,
)
from ..constants import GENERATE_PLAN_TOOL_NAME, GENERATE_PLAN_SCHEMA
from .transcript_repair import repair_transcript

logger = logging.getLogger(__name__)


class ReactLoopMixin:
    """Mixin providing the ReAct loop and its helpers.

    Expects the following attributes on ``self`` (provided by Orchestrator):
    - ``llm_client``
    - ``_react_config``
    - ``_context_manager``
    - ``_model_router``
    - ``_audit``
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
        context_tokens = getattr(self.llm_client, 'context_window', 128_000)
        if context_tokens < CONTEXT_HARD_MIN:
            yield AgentEvent(
                type=EventType.ERROR,
                data={"message": f"Model context window too small: {context_tokens} tokens (minimum: {CONTEXT_HARD_MIN})"},
            )
            return
        if context_tokens < CONTEXT_WARN_BELOW:
            logger.warning(f"Low context window: {context_tokens} tokens")

        start_time = time.monotonic()
        turn = 0
        all_tool_records: List[ToolCallRecord] = []
        total_usage = TokenUsage()
        pending_approvals = []
        final_response = ""
        result_status = None
        _recent_tool_names: List[str] = []  # watchdog loop detection

        logger.info(f"[ReAct] tenant={tenant_id}")

        yield AgentEvent(
            type=EventType.EXECUTION_START,
            data={"tenant_id": tenant_id},
        )

        # Model routing: classify once before the loop, reuse for all turns.
        routed_llm_client = None
        routing_score = -1
        if self._model_router:
            try:
                from ..llm.router import RoutingDecision
                decision = await self._model_router.route(messages)
                routing_score = decision.score
                routed_llm_client = self._model_router.registry.get(decision.provider)
                if routed_llm_client:
                    logger.info(
                        f"[ReAct] ModelRouter selected provider='{decision.provider}' "
                        f"(score={decision.score}, {decision.latency_ms:.0f}ms)"
                    )
            except Exception as e:
                logger.warning(f"[ReAct] ModelRouter failed, using default LLM: {e}")

        # Enable reasoning for complex requests on the first turn
        enable_reasoning = routing_score >= self._react_config.reasoning_score_threshold
        if enable_reasoning:
            logger.info(f"[ReAct] Reasoning enabled (score={routing_score}, effort={self._react_config.reasoning_effort})")

        # -- Planning phase --
        enable_planning = routing_score >= self._react_config.planning_score_threshold

        # Case 1: Pending plan from previous turn -- user is responding to it
        if self._tenant_plans.get(tenant_id) and context:
            pending_plan_text = self._format_plan_text(self._tenant_plans.pop(tenant_id))

            logger.info("[ReAct] Pending plan found, injecting into prompt for LLM to handle")
            messages = await self._build_llm_messages(
                context, user_message, pending_plan=pending_plan_text,
            )
            enable_planning = False  # don't re-plan

        # Case 2: New complex request -- generate plan and present to user
        elif enable_planning:
            logger.info(f"[ReAct] Planning phase triggered (score={routing_score})")
            try:
                plan_messages = await self._build_llm_messages(
                    context, user_message, include_planning=True,
                )
                plan_schemas = [GENERATE_PLAN_SCHEMA, COMPLETE_TASK_SCHEMA]
                plan_response = await self._llm_call_with_retry(
                    plan_messages, plan_schemas, tool_choice="auto",
                    llm_client_override=routed_llm_client,
                )
                plan_data = self._extract_plan_from_response(plan_response)
                if plan_data and self._react_config.planning_requires_approval:
                    # Present plan to user, pause execution
                    plan_text = self._format_plan_text(plan_data)
                    friendly = self._format_plan_for_user(plan_data)
                    self._tenant_plans[tenant_id] = plan_data
                    logger.info(f"[ReAct] Plan generated, awaiting approval: {plan_data.get('goal', '')}")
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
                        context, user_message, approved_plan=plan_text,
                    )
                else:
                    logger.info("[ReAct] LLM did not generate a plan, proceeding directly")
            except Exception as e:
                logger.warning(f"[ReAct] Planning phase failed, proceeding without plan: {e}")

        for turn in range(1, self._react_config.max_turns + 1):
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
                response = await self._llm_call_with_retry(
                    messages, tool_schemas, tool_choice=tool_choice,
                    llm_client_override=routed_llm_client,
                    **extra_kwargs,
                )
            except Exception as e:
                yield AgentEvent(
                    type=EventType.ERROR,
                    data={"error": str(e), "error_type": type(e).__name__},
                )
                return

            # Accumulate token usage
            usage = getattr(response, "usage", None)
            if usage:
                total_usage.input_tokens += getattr(usage, "prompt_tokens", 0)
                total_usage.output_tokens += getattr(usage, "completion_tokens", 0)

            tool_calls = response.tool_calls

            # No tool calls -> LLM forgot to call complete_task.
            # Retry up to max_complete_task_retries times with tool_choice="required".
            if not tool_calls:
                max_retries = self._react_config.max_complete_task_retries
                for retry in range(1, max_retries + 1):
                    grace_msg = (
                        "You must call the `complete_task` tool with your final "
                        "response in the `result` parameter to finish. Do not "
                        "respond with plain text. Call `complete_task` now."
                    )
                    logger.warning(
                        f"[ReAct] turn={turn} no tool calls, "
                        f"grace retry {retry}/{max_retries}"
                    )
                    messages.append(self._assistant_message_from_response(response))
                    messages.append({"role": "user", "content": grace_msg})
                    try:
                        response = await self._llm_call_with_retry(
                            messages, tool_schemas, tool_choice="required",
                            llm_client_override=routed_llm_client,
                        )
                    except Exception as e:
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data={"error": str(e), "error_type": type(e).__name__},
                        )
                        return
                    usage_retry = getattr(response, "usage", None)
                    if usage_retry:
                        total_usage.input_tokens += getattr(usage_retry, "prompt_tokens", 0)
                        total_usage.output_tokens += getattr(usage_retry, "completion_tokens", 0)
                    tool_calls = response.tool_calls
                    if tool_calls:
                        break  # success -- proceed to tool execution

                # Exhausted all retries -- ask LLM to produce a user-friendly
                # error message (no tools, so it can only return text).
                if not tool_calls:
                    logger.error(
                        f"[ReAct] turn={turn} exhausted {max_retries} "
                        f"grace retries, LLM still did not call complete_task"
                    )
                    messages.append(self._assistant_message_from_response(response))
                    messages.append({
                        "role": "user",
                        "content": (
                            "There was an internal issue processing your request. "
                            "Generate a short, friendly apology to the user in "
                            "their language, and suggest they try again later."
                        ),
                    })
                    try:
                        fallback_resp = await self._llm_call_with_retry(
                            messages, tool_schemas=[], tool_choice=None,
                            llm_client_override=routed_llm_client,
                        )
                        final_response = (
                            getattr(fallback_resp, "content", None)
                            or fallback_resp.choices[0].message.content
                            or "Sorry, something went wrong. Please try again later."
                        )
                    except Exception:
                        final_response = "Sorry, something went wrong. Please try again later."
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=[], final_answer=True,
                        tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    return

            if tool_calls:
                # Append assistant message with tool_calls
                messages.append(self._assistant_message_from_response(response))

                # ----------------------------------------------------------
                # Intercept complete_task: handle synchronously, skip execution
                # ----------------------------------------------------------
                complete_task_result: Optional[CompleteTaskResult] = None
                remaining_tool_calls = []
                for tc in tool_calls:
                    if tc.name == COMPLETE_TASK_TOOL_NAME:
                        try:
                            _ct_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                        except (json.JSONDecodeError, TypeError):
                            _ct_args = {}
                        _ct_text = _ct_args.get("result", "")
                        if _ct_text:
                            complete_task_result = CompleteTaskResult(result=_ct_text)
                            messages.append(self._build_tool_result_message(tc.id, "Task completed."))
                            all_tool_records.append(ToolCallRecord(
                                name=COMPLETE_TASK_TOOL_NAME,
                                args_summary={"result": _ct_text[:100]},
                                duration_ms=0, success=True,
                                result_status="COMPLETED",
                                result_chars=len(_ct_text),
                            ))
                            logger.info(f"[ReAct] turn={turn} complete_task called ({len(_ct_text)} chars)")
                        else:
                            # Missing result -- append error, let LLM retry
                            messages.append(self._build_tool_result_message(
                                tc.id,
                                'Error: "result" argument is required for complete_task.',
                                is_error=True,
                            ))
                            remaining_tool_calls.append(tc)
                    else:
                        remaining_tool_calls.append(tc)

                # Pure complete_task with no other tools -- break immediately
                if complete_task_result and not remaining_tool_calls:
                    final_response = complete_task_result.result
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=[COMPLETE_TASK_TOOL_NAME],
                        final_answer=True, tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

                # Use remaining tools for execution (or original list if complete_task had no result)
                tool_calls = remaining_tool_calls if remaining_tool_calls else tool_calls

                tool_names = [tc.name for tc in tool_calls]
                logger.info(f"[ReAct] turn={turn} calling: {', '.join(tool_names)}")

                # Yield tool call start events
                for tc in tool_calls:
                    yield AgentEvent(
                        type=EventType.TOOL_CALL_START,
                        data={"tool_name": tc.name, "call_id": tc.id},
                    )

                # Execute all tool calls concurrently
                tc_batch_start = time.monotonic()
                results = await asyncio.gather(
                    *[self._execute_with_timeout(tc, tenant_id, metadata=metadata, request_tools=request_tools, request_context=context) for tc in tool_calls],
                    return_exceptions=True,
                )
                tc_batch_duration = int((time.monotonic() - tc_batch_start) * 1000)

                # Token attribution for this turn
                turn_tokens = None
                if usage:
                    turn_tokens = TokenUsage(
                        input_tokens=getattr(usage, "prompt_tokens", 0),
                        output_tokens=getattr(usage, "completion_tokens", 0),
                    )

                loop_broken = False
                loop_broken_text = None

                for tc, result in zip(tool_calls, results):
                    tc_name = tc.name
                    is_agent = self._is_agent_tool(tc_name)
                    kind = "agent" if is_agent else "tool"

                    try:
                        args_summary = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args_summary = {}
                    args_summary = {k: str(v)[:100] for k, v in args_summary.items()}

                    if isinstance(result, BaseException):
                        logger.warning(f"[ReAct]   {kind}={tc_name} ERROR: {result}")
                        error_text = f"Error executing {tc_name}: {result}"
                        messages.append(self._build_tool_result_message(tc.id, error_text, is_error=True))
                        all_tool_records.append(ToolCallRecord(
                            name=tc_name, args_summary=args_summary,
                            duration_ms=tc_batch_duration, success=False,
                            result_chars=len(error_text), token_attribution=turn_tokens,
                        ))
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name, "call_id": tc.id,
                                "kind": kind, "success": False,
                                "error": str(result),
                                "result_preview": error_text[:240],
                            },
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name, args_summary=args_summary,
                            success=False, duration_ms=tc_batch_duration,
                            error=str(result), tenant_id=tenant_id,
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
                            "WAITING_FOR_APPROVAL" if result.approval_request
                            else "WAITING_FOR_INPUT"
                        )
                        all_tool_records.append(ToolCallRecord(
                            name=tc_name, args_summary=args_summary,
                            duration_ms=tc_batch_duration, success=True,
                            result_status=waiting_status,
                            result_chars=len(waiting_text), token_attribution=turn_tokens,
                        ))
                        tool_trace = []
                        if isinstance(result.metadata, dict):
                            tool_trace = result.metadata.get("tool_trace") or []
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name, "call_id": tc.id,
                                "kind": "agent", "success": True,
                                "waiting": True, "status": waiting_status,
                                "result_preview": waiting_text[:240],
                                "tool_trace": tool_trace,
                            },
                        )
                        yield AgentEvent(
                            type=EventType.STATE_CHANGE,
                            data={"agent_type": tc_name, "status": waiting_status},
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name, args_summary=args_summary,
                            success=True, duration_ms=tc_batch_duration,
                            tenant_id=tenant_id,
                        )
                        loop_broken = True
                        loop_broken_text = waiting_text

                    else:
                        if isinstance(result, AgentToolResult):
                            result_text = result.result_text
                            r_meta = result.metadata if isinstance(result.metadata, dict) else {}
                            tool_trace = r_meta.get("tool_trace") or []
                        else:
                            result_text = str(result) if result is not None else ""
                            tool_trace = []
                        result_chars_original = len(result_text)
                        # Hard cap on tool result size
                        result_text = self._cap_tool_result(result_text)
                        result_text = self._context_manager.truncate_tool_result(result_text)
                        # Context isolation for agent-tools
                        if is_agent and len(result_text) > 2000:
                            result_text = result_text[:1500] + "\n...[full result available in agent context]"
                        logger.info(f"[ReAct]   {kind}={tc_name} OK ({len(result_text)} chars)")
                        messages.append(self._build_tool_result_message(tc.id, result_text))
                        all_tool_records.append(ToolCallRecord(
                            name=tc_name, args_summary=args_summary,
                            duration_ms=tc_batch_duration, success=True,
                            result_status="COMPLETED" if isinstance(result, AgentToolResult) else None,
                            result_chars=result_chars_original, token_attribution=turn_tokens,
                        ))
                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool_name": tc_name, "call_id": tc.id,
                                "kind": kind, "success": True,
                                "result_preview": result_text[:240],
                                "tool_trace": tool_trace,
                            },
                        )
                        self._audit.log_tool_execution(
                            tool_name=tc_name, args_summary=args_summary,
                            success=True, duration_ms=tc_batch_duration,
                            tenant_id=tenant_id,
                        )

                # complete_task was called alongside other tools -- use its result
                if complete_task_result:
                    final_response = complete_task_result.result
                    self._audit.log_react_turn(
                        turn=turn, tool_calls=tool_names + [COMPLETE_TASK_TOOL_NAME],
                        final_answer=True, tenant_id=tenant_id,
                    )
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

                # Watchdog: detect loops (same tool 3x OR cycle pattern)
                for tn in tool_names:
                    _recent_tool_names.append(tn)
                if len(_recent_tool_names) >= 3:
                    last_3 = _recent_tool_names[-3:]
                    # Same tool 3x in a row
                    if len(set(last_3)) == 1:
                        logger.warning(f"[ReAct] Loop detected: {last_3[0]} called 3 times consecutively")
                        final_response = "I noticed I was repeating the same action without making progress. Let me provide what I have so far."
                        yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                        yield AgentEvent(type=EventType.MESSAGE_END, data={})
                        break
                if len(_recent_tool_names) >= 6:
                    last_6 = _recent_tool_names[-6:]
                    # A->B->A->B->A->B pattern (3 cycles of 2)
                    if last_6[0::2] == last_6[0::2][:1] * 3 and last_6[1::2] == last_6[1::2][:1] * 3:
                        logger.warning(f"[ReAct] Oscillation detected: {last_6[0]}↔{last_6[1]}")
                        final_response = "I noticed I was going back and forth between actions without making progress. Let me provide what I have so far."
                        yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_response})
                        yield AgentEvent(type=EventType.MESSAGE_END, data={})
                        break

                # Audit: log turn summary
                self._audit.log_react_turn(
                    turn=turn,
                    tool_calls=tool_names,
                    final_answer=False,
                    tenant_id=tenant_id,
                )

                if loop_broken:
                    final_response = loop_broken_text or ""
                    result_status = "WAITING_FOR_APPROVAL" if pending_approvals else "WAITING_FOR_INPUT"
                    if pending_approvals:
                        pending_approvals = collect_batch_approvals(pending_approvals)
                    if loop_broken_text:
                        yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                        yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": loop_broken_text})
                        yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

                # Agent passthrough: single completed agent-tool skips LLM re-summary
                if (
                    len(tool_calls) == 1
                    and self._is_agent_tool(tool_calls[0].name)
                    and isinstance(results[0], AgentToolResult)
                    and results[0].completed
                ):
                    agent_text = results[0].result_text
                    logger.info(
                        f"[ReAct] turn={turn} agent_passthrough "
                        f"({len(agent_text)} chars from {tool_calls[0].name})"
                    )
                    final_response = agent_text
                    yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
                    yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": agent_text})
                    yield AgentEvent(type=EventType.MESSAGE_END, data={})
                    break

        else:
            # max_turns reached: ask LLM for summary without tools
            messages.append({
                "role": "user",
                "content": (
                    "You have used all available turns. Please provide your best "
                    "final answer based on the information gathered so far."
                ),
            })
            try:
                response = await self._llm_call_with_retry(
                    messages, tool_schemas=None,
                    llm_client_override=routed_llm_client,
                )
                final_text = response.content or ""
                usage = getattr(response, "usage", None)
                if usage:
                    total_usage.input_tokens += getattr(usage, "prompt_tokens", 0)
                    total_usage.output_tokens += getattr(usage, "completion_tokens", 0)
            except Exception:
                final_text = "I was unable to complete the request within the allowed turns."

            final_response = final_text
            yield AgentEvent(type=EventType.MESSAGE_START, data={"turn": turn})
            yield AgentEvent(type=EventType.MESSAGE_CHUNK, data={"chunk": final_text})
            yield AgentEvent(type=EventType.MESSAGE_END, data={})

        duration_ms = int((time.monotonic() - start_time) * 1000)

        yield AgentEvent(
            type=EventType.EXECUTION_END,
            data={
                "duration_ms": duration_ms,
                "turns": turn,
                "tool_calls_count": len(all_tool_records),
                "final_response": final_response,
                "result_status": result_status,
                "pending_approvals": pending_approvals,
                "token_usage": {
                    "input_tokens": total_usage.input_tokens,
                    "output_tokens": total_usage.output_tokens,
                },
                "tool_calls": [dataclasses.asdict(r) for r in all_tool_records],
            },
        )

    async def _save_tool_call_history(
        self, tenant_id: str, tool_calls: list,
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

        # Build a compact representation of old messages for summarization
        old_text_parts = []
        for msg in old_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                # Truncate very long individual messages for the summary request
                if len(content) > 500:
                    content = content[:497] + "..."
                old_text_parts.append(f"{role}: {content}")

        if not old_text_parts:
            return self._context_manager.trim_if_needed(messages)

        old_text = "\n".join(old_text_parts)
        # Cap the input to the summarizer to avoid nested overflow
        if len(old_text) > 8000:
            old_text = old_text[:8000] + "\n...[truncated]"

        try:
            summary_response = await self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation excerpt in 2-4 sentences. "
                            "Preserve key facts, decisions, entities, and user preferences. "
                            "Be concise but retain actionable context."
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
                return self._context_manager.build_summarized_messages(
                    system_msgs, summary, recent_msgs,
                )
        except Exception as e:
            logger.warning(f"[Context] Summarization failed, falling back to trim: {e}")

        return self._context_manager.trim_if_needed(messages)

    def _build_tool_result_message(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        """Build a tool result message for the LLM messages list."""
        if is_error:
            content = f"[ERROR] {content}"
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
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
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return msg
