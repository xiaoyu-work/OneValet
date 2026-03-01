"""Core multi-turn conversation wrapper for LLM integration tests.

Usage::

    conv = Conversation(handler=orch, recorder=recorder)
    await conv.send_until_tool_called("Find Italian restaurants near me")
    conv.assert_tool_called("search_places")
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import assertions as _asrt
from .types import MessageHandler, ToolRecorder

# Statuses that mean the conversation has reached a terminal state.
# Works with both str values and enum members thanks to ``in`` check.
_TERMINAL_STATUSES = frozenset({"completed", "cancelled", "error"})

# Status that means the framework is waiting for user approval.
_APPROVAL_STATUS = "waiting_for_approval"


class ConversationError(Exception):
    """Raised when a conversation flow hits an unexpected state."""


class Conversation:
    """Stateful wrapper for multi-turn LLM integration tests.

    Wraps a message handler (orchestrator) and a tool-call recorder,
    providing high-level methods for common multi-turn test patterns.

    Parameters
    ----------
    handler:
        Object with an ``async handle_message(user_id, message)`` method.
    recorder:
        Object with a ``tool_calls`` list attribute.
    user_id:
        Tenant / user id used for all messages.
    """

    def __init__(
        self,
        handler: MessageHandler,
        recorder: ToolRecorder,
        user_id: str = "test_user",
    ) -> None:
        self.handler = handler
        self.recorder = recorder
        self.user_id = user_id
        self.turns: List[Any] = []
        self._tool_offsets: List[int] = []

    # ------------------------------------------------------------------
    # Core turn methods
    # ------------------------------------------------------------------

    async def send(self, message: str) -> Any:
        """Send a single message and record the result as one turn."""
        self._tool_offsets.append(len(self.recorder.tool_calls))
        result = await self.handler.handle_message(self.user_id, message)
        self.turns.append(result)
        return result

    async def send_until_tool_called(
        self,
        message: str,
        *,
        auto_reply: str = "yes, go ahead",
        max_turns: int = 5,
    ) -> Any:
        """Send *message*, auto-reply until a NEW tool call is recorded.

        Some agents ask for text confirmation before calling a tool.
        This method keeps replying with *auto_reply* until
        ``recorder.tool_calls`` grows.
        """
        baseline = len(self.recorder.tool_calls)
        result = await self.send(message)

        for _ in range(max_turns - 1):
            if len(self.recorder.tool_calls) > baseline:
                return result
            # If already terminal with no tool call, nudge the LLM
            result = await self.send(auto_reply)

        if len(self.recorder.tool_calls) > baseline:
            return result

        raise ConversationError(
            f"No tool was called after {max_turns} turns. "
            f"Tools recorded: {self.tools_called}"
        )

    async def send_until_status(
        self,
        message: str,
        status: Any,
        *,
        auto_reply: str = "yes, go ahead",
        max_turns: int = 5,
    ) -> Any:
        """Send *message*, auto-reply until ``result.status == status``.

        Useful for reaching ``WAITING_FOR_APPROVAL`` through possible
        text-confirmation rounds.
        """
        result = await self.send(message)

        for _ in range(max_turns - 1):
            if self._status_matches(result, status):
                return result
            # Stop if we hit an unexpected terminal state
            if self._is_terminal(result) and not self._status_matches(result, status):
                return result
            result = await self.send(auto_reply)

        return result

    async def auto_complete(
        self,
        message: str,
        *,
        max_turns: int = 8,
    ) -> Any:
        """Send *message* and auto-approve everything until a terminal status.

        Handles both text-confirmation rounds (replies "yes, go ahead")
        and framework approval gates (replies "yes, approve it").
        """
        result = await self.send(message)

        for _ in range(max_turns - 1):
            if self._is_terminal(result):
                return result
            if self._status_matches(result, _APPROVAL_STATUS):
                result = await self.send("yes, approve it")
            else:
                result = await self.send("yes, go ahead")

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> Any:
        """The result from the most recent turn."""
        if not self.turns:
            raise ConversationError("No turns yet")
        return self.turns[-1]

    @property
    def last_message(self) -> str:
        """``raw_message`` from the most recent turn."""
        return self.last_result.raw_message

    @property
    def last_status(self) -> Any:
        """``status`` from the most recent turn."""
        return self.last_result.status

    @property
    def tools_called(self) -> List[str]:
        """All tool names called across all turns, in order."""
        return [c["tool_name"] for c in self.recorder.tool_calls]

    def get_tool_calls(self, tool_name: str) -> List[Dict[str, Any]]:
        """Return all recorded calls to *tool_name*."""
        return [
            c for c in self.recorder.tool_calls
            if c["tool_name"] == tool_name
        ]

    def get_tool_args(self, tool_name: str) -> List[Dict[str, Any]]:
        """Return the ``arguments`` dict for each call to *tool_name*."""
        return [c["arguments"] for c in self.get_tool_calls(tool_name)]

    def tools_in_turn(self, turn: int) -> List[str]:
        """Tool names called during a specific turn (0-indexed)."""
        if turn < 0 or turn >= len(self.turns):
            raise IndexError(
                f"Turn {turn} out of range (have {len(self.turns)} turns)"
            )
        start = self._tool_offsets[turn]
        end = (
            self._tool_offsets[turn + 1]
            if turn + 1 < len(self._tool_offsets)
            else len(self.recorder.tool_calls)
        )
        return [c["tool_name"] for c in self.recorder.tool_calls[start:end]]

    # ------------------------------------------------------------------
    # Assertion helpers (delegate to assertions module)
    # ------------------------------------------------------------------

    def assert_tool_called(self, tool_name: str) -> None:
        """Assert *tool_name* was called at least once."""
        _asrt.assert_tool_called(self, tool_name)

    def assert_any_tool_called(self, tool_names: List[str]) -> None:
        """Assert at least one of *tool_names* was called."""
        _asrt.assert_any_tool_called(self, tool_names)

    def assert_tool_args(self, tool_name: str, **expected: Any) -> None:
        """Assert *tool_name* was called with matching arguments."""
        _asrt.assert_tool_args(self, tool_name, **expected)

    def assert_status(self, expected: Any) -> None:
        """Assert the last turn's status matches *expected*."""
        _asrt.assert_status(self, expected)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _status_matches(result: Any, status: Any) -> bool:
        """Compare result.status with *status*, handling str/enum mix."""
        actual = result.status
        # Support both enum and string comparison
        if hasattr(actual, "value"):
            return actual.value == str(status) or actual == status
        return actual == status

    @staticmethod
    def _is_terminal(result: Any) -> bool:
        """Check if *result* has a terminal status."""
        actual = result.status
        val = actual.value if hasattr(actual, "value") else actual
        return val in _TERMINAL_STATUSES
