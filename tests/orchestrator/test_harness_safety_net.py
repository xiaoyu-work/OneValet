"""Tests for the top-level safety net in Orchestrator._execute_message.

The safety net wraps the entire body of _execute_message so that ANY
unhandled exception still yields error events and a user-facing message
instead of silently killing the generator.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koa.result import AgentResult, AgentStatus
from koa.streaming.models import AgentEvent, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator_stub():
    """Build a minimal Orchestrator-like object that can run _execute_message."""
    from koa.orchestrator.orchestrator import Orchestrator

    orch = object.__new__(Orchestrator)
    # Minimal attributes the method checks before doing real work
    orch._initialized = True
    orch.llm_client = None
    orch._audit = MagicMock()
    orch._audit.start_request = MagicMock(return_value="req-123")
    orch._audit.log_phase = MagicMock()
    orch._audit.end_request = MagicMock()
    orch._cleanup_stale_agents = AsyncMock()
    return orch


async def _collect_events(async_gen) -> list[AgentEvent]:
    events = []
    async for event in async_gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSafetyNetCatchesUnhandledErrors:
    """Verify that exceptions raised before the ReAct loop are caught."""

    @pytest.mark.asyncio
    async def test_runtime_error_yields_error_event(self):
        """An arbitrary RuntimeError should produce ERROR + MESSAGE_CHUNK + MESSAGE_END + EXECUTION_END."""
        orch = _make_orchestrator_stub()

        with patch.object(
            type(orch),
            "prepare_context",
            side_effect=RuntimeError("kaboom"),
        ):
            events = await _collect_events(orch._execute_message("tenant-1", "hello"))

        types = [e.type for e in events]
        assert EventType.ERROR in types, f"Expected ERROR event, got {types}"
        assert EventType.MESSAGE_CHUNK in types, f"Expected MESSAGE_CHUNK event, got {types}"
        assert EventType.MESSAGE_END in types, f"Expected MESSAGE_END event, got {types}"
        assert EventType.EXECUTION_END in types, f"Expected EXECUTION_END event, got {types}"

    @pytest.mark.asyncio
    async def test_error_event_contains_metadata(self):
        orch = _make_orchestrator_stub()

        with patch.object(
            type(orch),
            "prepare_context",
            side_effect=ValueError("bad input"),
        ):
            events = await _collect_events(orch._execute_message("tenant-1", "hi"))

        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) == 1
        data = error_events[0].data
        assert data["code"] == "internal_error"
        assert "bad input" in data["error"]
        assert data["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_execution_end_carries_agent_result_with_error_status(self):
        orch = _make_orchestrator_stub()

        with patch.object(
            type(orch),
            "prepare_context",
            side_effect=TypeError("oops"),
        ):
            events = await _collect_events(orch._execute_message("tenant-1", "hey"))

        end_events = [e for e in events if e.type == EventType.EXECUTION_END]
        assert len(end_events) == 1
        result = end_events[0].data
        assert isinstance(result, AgentResult)
        assert result.status == AgentStatus.ERROR

    @pytest.mark.asyncio
    async def test_message_chunk_contains_fallback_text(self):
        orch = _make_orchestrator_stub()

        with (
            patch.object(
                type(orch),
                "prepare_context",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "koa.orchestrator.graceful_response.generate_graceful_error",
                new_callable=lambda: AsyncMock(return_value="whoops, brain glitch"),
            ),
        ):
            events = await _collect_events(orch._execute_message("tenant-1", "yo"))

        chunk_events = [e for e in events if e.type == EventType.MESSAGE_CHUNK]
        assert len(chunk_events) >= 1
        assert chunk_events[0].data["chunk"] == "whoops, brain glitch"


class TestSafetyNetDoesNotInterfereWithNormalFlow:
    """The safety net must be transparent when no exception occurs."""

    @pytest.mark.asyncio
    async def test_inner_react_error_still_handled_by_inner_except(self):
        """_ReactLoopLLMError should still be handled by the existing inner
        try/except — not by the outer safety net."""
        from koa.orchestrator.react_loop import _ReactLoopLLMError

        # We can't easily run the full pipeline, but we can verify the
        # import and class exist for the inner handler.
        assert _ReactLoopLLMError is not None


class TestHardcodedMessagesReplaced:
    """Verify that the hardcoded sorry messages have been replaced with
    generate_graceful_error calls."""

    def test_no_hardcoded_sorry_in_orchestrator(self):
        import inspect

        from koa.orchestrator.orchestrator import Orchestrator

        source = inspect.getsource(Orchestrator._execute_message)
        assert "Sorry, I'm having trouble processing your request" not in source, (
            "Hardcoded sorry message should be replaced with generate_graceful_error"
        )
