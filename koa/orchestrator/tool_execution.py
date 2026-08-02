"""Executing one turn's tool calls and turning each result into events.

Split out of the ReAct loop, where it was the largest single block. The
work here is genuinely one job -- run the calls the model asked for, put
an answer for every call back into the transcript, and tell the caller
what happened -- and it reads better as one job than as the middle third
of a much longer function.

Every path appends exactly one tool result message per call. That is not
a style preference: a transcript with an unanswered tool_call is rejected
by the provider on the next turn, so an early return that skips a result
would break the run rather than just lose information.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from ..models import ToolOutput
from ..streaming.models import AgentEvent, EventType
from .agent_tool import AgentToolResult
from .run_state import RunState

logger = logging.getLogger(__name__)

#: Agent results are summaries already; past this they crowd out the
#: transcript they are supposed to inform.
_AGENT_RESULT_CAP = 2000
_AGENT_RESULT_KEEP = 1500


@dataclass
class TurnOutcome:
    """What one turn's tool execution leaves for the loop to act on."""

    #: Results in call order -- the watchdog and the passthrough check both
    #: need position, which completion order does not preserve.
    timed_results: List[Any] = field(default_factory=list)

    #: An agent asked for input or approval, so the run stops after this turn.
    waiting: bool = False
    waiting_text: Optional[str] = None


def _args_summary(arguments: Any) -> Dict[str, Any]:
    """Truncated argument map for logs and records."""
    try:
        parsed = arguments if isinstance(arguments, dict) else json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: str(v)[:100] for k, v in parsed.items()}


def _unpack(timed: Any) -> tuple:
    """Split a TimedResult into (result, duration_ms)."""
    result = getattr(timed, "result", timed)
    return result, getattr(timed, "duration_ms", 0)


class ToolExecutionMixin:
    """Runs a turn's tool calls and yields an event per result."""

    async def _execute_one_timed(
        self,
        tc: Any,
        control: Any,
        *,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]],
        request_tools: Optional[List],
        context: Optional[Dict[str, Any]],
        speculative: Dict[str, Any],
        interrupted_sentinel: Any,
    ) -> Any:
        """Run one tool call, timing it and never raising."""
        from .react_loop import TimedResult

        t0 = time.monotonic()

        reused = await self._try_speculative(tc, speculative, t0)
        if reused is not None:
            return reused

        try:
            r = await control.race(
                self._execute_with_timeout(
                    tc,
                    tenant_id,
                    metadata=metadata,
                    request_tools=request_tools,
                    request_context=context,
                ),
                interrupted=interrupted_sentinel,
            )
            if r is interrupted_sentinel:
                r = InterruptedError("Interrupted by user during execution.")
        except BaseException as exc:
            # Returned rather than raised: the caller must still write a
            # tool result for this call, and a raise here would skip that.
            return TimedResult(result=exc, duration_ms=int((time.monotonic() - t0) * 1000))
        return TimedResult(result=r, duration_ms=int((time.monotonic() - t0) * 1000))

    async def _try_speculative(self, tc: Any, speculative: Dict[str, Any], t0: float) -> Any:
        """Return a pre-started result for this call, or None to run it."""
        from .react_loop import TimedResult

        if not speculative or tc.name != "google_search":
            return None
        try:
            args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        except (json.JSONDecodeError, TypeError):
            args = {}
        spec_key = f"google_search:{args.get('search_type', 'web')}"
        spec_task = speculative.pop(spec_key, None)
        if spec_task is None:
            return None
        try:
            result = await spec_task
        except Exception as e:
            logger.info(f"[Speculative] {spec_key} failed, falling back: {e}")
            return None
        if result is None:
            return None
        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(f"[Speculative] Reused {spec_key} (waited {elapsed}ms for pre-started task)")
        return TimedResult(result=result, duration_ms=elapsed)

    async def _run_tool_calls(
        self,
        tool_calls: List[Any],
        messages: List[Dict[str, Any]],
        state: RunState,
        control: Any,
        outcome: TurnOutcome,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        request_tools: Optional[List] = None,
        context: Optional[Dict[str, Any]] = None,
        turn_tokens: Any = None,
        interrupted_sentinel: Any = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a turn's tool calls, yielding a result event for each.

        Results are handled as they land rather than after all finish, so a
        fast tool is not held behind a slow one.
        """
        speculative = (context or {}).get("_speculative_tasks", {})
        outcome.timed_results = [None] * len(tool_calls)

        async def _indexed(idx: int, tc: Any):
            return idx, await self._execute_one_timed(
                tc,
                control,
                tenant_id=state.tenant_id,
                metadata=metadata,
                request_tools=request_tools,
                context=context,
                speculative=speculative,
                interrupted_sentinel=interrupted_sentinel,
            )

        pending = [_indexed(i, tc) for i, tc in enumerate(tool_calls)]
        for completed in asyncio.as_completed(pending):
            idx, timed = await completed
            tc = tool_calls[idx]
            outcome.timed_results[idx] = timed

            result, duration_ms = _unpack(timed)
            is_agent = self._is_agent_tool(tc.name)
            args = _args_summary(tc.arguments)

            if isinstance(result, BaseException):
                handler = self._on_tool_error(
                    tc, result, duration_ms, args, messages, state, turn_tokens
                )
            elif isinstance(result, AgentToolResult) and not result.completed:
                handler = self._on_tool_waiting(
                    tc, result, duration_ms, args, messages, state, outcome, turn_tokens
                )
            else:
                handler = self._on_tool_success(
                    tc, result, duration_ms, args, messages, state, is_agent, turn_tokens
                )

            async for event in handler:
                yield event

    # ── result handlers ───────────────────────────────────────────────

    async def _on_tool_error(
        self,
        tc: Any,
        result: BaseException,
        duration_ms: int,
        args: Dict[str, Any],
        messages: List[Dict[str, Any]],
        state: RunState,
        turn_tokens: Any,
    ) -> AsyncIterator[AgentEvent]:
        kind = "agent" if self._is_agent_tool(tc.name) else "tool"
        logger.warning(f"[ReAct]   {kind}={tc.name} ERROR: {result}")
        error_text = f"Error executing {tc.name}: {result}"
        messages.append(self._build_tool_result_message(tc.id, error_text, is_error=True))
        state.record_tool_call(
            name=tc.name,
            args_summary=args,
            duration_ms=duration_ms,
            success=False,
            result_chars=len(error_text),
            token_attribution=turn_tokens,
        )
        yield AgentEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool_name": tc.name,
                "call_id": tc.id,
                "kind": kind,
                "success": False,
                "error": str(result),
                "result_preview": error_text[:240],
            },
        )
        self._audit.log_tool_execution(
            tool_name=tc.name,
            args_summary=args,
            success=False,
            duration_ms=duration_ms,
            error=str(result),
            tenant_id=state.tenant_id,
        )

    async def _on_tool_waiting(
        self,
        tc: Any,
        result: AgentToolResult,
        duration_ms: int,
        args: Dict[str, Any],
        messages: List[Dict[str, Any]],
        state: RunState,
        outcome: TurnOutcome,
        turn_tokens: Any,
    ) -> AsyncIterator[AgentEvent]:
        logger.info(f"[ReAct]   agent={tc.name} WAITING")
        if result.agent:
            await self.agent_pool.add_agent(result.agent)
        if result.approval_request:
            state.pending_approvals.append(result.approval_request)

        waiting_text = result.result_text or "Agent is waiting for input."
        messages.append(self._build_tool_result_message(tc.id, waiting_text))
        status = "WAITING_FOR_APPROVAL" if result.approval_request else "WAITING_FOR_INPUT"
        state.record_tool_call(
            name=tc.name,
            args_summary=args,
            duration_ms=duration_ms,
            success=True,
            result_status=status,
            result_chars=len(waiting_text),
            token_attribution=turn_tokens,
        )

        trace = result.metadata.get("tool_trace") or [] if isinstance(result.metadata, dict) else []
        yield AgentEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool_name": tc.name,
                "call_id": tc.id,
                "kind": "agent",
                "success": True,
                "waiting": True,
                "status": status,
                "result_preview": waiting_text[:240],
                "tool_trace": trace,
            },
        )
        yield AgentEvent(
            type=EventType.STATE_CHANGE,
            data={"agent_type": tc.name, "status": status},
        )
        self._audit.log_tool_execution(
            tool_name=tc.name,
            args_summary=args,
            success=True,
            duration_ms=duration_ms,
            tenant_id=state.tenant_id,
        )
        outcome.waiting = True
        outcome.waiting_text = waiting_text

    async def _on_tool_success(
        self,
        tc: Any,
        result: Any,
        duration_ms: int,
        args: Dict[str, Any],
        messages: List[Dict[str, Any]],
        state: RunState,
        is_agent: bool,
        turn_tokens: Any,
    ) -> AsyncIterator[AgentEvent]:
        kind = "agent" if is_agent else "tool"
        text, media, trace = _extract_result_parts(result)
        original_len = len(text)
        text = self._fit_tool_result(text, is_agent)

        logger.info(f"[ReAct]   {kind}={tc.name} OK ({len(text)} chars, media={len(media)})")
        messages.append(self._build_tool_result_message(tc.id, text, media=media))

        # Media the client has to keep (images) or render (cards) travels
        # with the final response; the rest only mattered to the model.
        from .react_loop import _append_unique_media

        for m in media:
            meta = m.get("metadata", {})
            if meta.get("for_storage") or m.get("type") == "inline_cards":
                _append_unique_media(state.response_media, [m])

        state.record_tool_call(
            name=tc.name,
            args_summary=args,
            duration_ms=duration_ms,
            success=True,
            result_status="COMPLETED" if isinstance(result, AgentToolResult) else None,
            result_chars=original_len,
            token_attribution=turn_tokens,
        )
        yield AgentEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool_name": tc.name,
                "call_id": tc.id,
                "kind": kind,
                "success": True,
                "result_preview": text[:240],
                "tool_trace": trace,
            },
        )
        self._audit.log_tool_execution(
            tool_name=tc.name,
            args_summary=args,
            success=True,
            duration_ms=duration_ms,
            tenant_id=state.tenant_id,
        )

    def _fit_tool_result(self, text: str, is_agent: bool) -> str:
        """Shrink a result to fit the window, saying so when it is cut."""
        original_len = len(text)
        text = self._cap_tool_result(text)
        text = self._context_manager.truncate_tool_result(text)
        if is_agent and len(text) > _AGENT_RESULT_CAP:
            return (
                text[:_AGENT_RESULT_KEEP]
                + f"\n...[truncated from {original_len} to {_AGENT_RESULT_KEEP} chars]"
            )
        if len(text) < original_len:
            return text + f"\n...[truncated from {original_len} to {len(text)} chars]"
        return text


def _extract_result_parts(result: Any) -> tuple:
    """Pull (text, media, tool_trace) out of whatever a tool returned."""
    if isinstance(result, ToolOutput):
        return result.text, result.media or [], []
    if isinstance(result, AgentToolResult):
        meta = result.metadata if isinstance(result.metadata, dict) else {}
        return result.result_text, meta.get("media") or [], meta.get("tool_trace") or []
    return (str(result) if result is not None else ""), [], []
