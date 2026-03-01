"""Assertion helpers for multi-turn conversation tests.

These can be used as standalone functions or via the convenience
methods on :class:`~.conversation.Conversation`.

::

    from tests.integration.framework import assert_tool_called

    assert_tool_called(conv, "send_email")
    # -- or equivalently --
    conv.assert_tool_called("send_email")
"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import Conversation


def assert_tool_called(conv: Conversation, tool_name: str) -> None:
    """Assert that *tool_name* was called at least once (any turn)."""
    assert tool_name in conv.tools_called, (
        f"Expected '{tool_name}' to be called, but got: {conv.tools_called}"
    )


def assert_any_tool_called(conv: Conversation, tool_names: List[str]) -> None:
    """Assert that at least one of *tool_names* was called."""
    assert any(t in conv.tools_called for t in tool_names), (
        f"Expected one of {tool_names}, but got: {conv.tools_called}"
    )


def assert_tool_args(conv: Conversation, tool_name: str, **expected: Any) -> None:
    """Assert that *tool_name* was called with arguments matching *expected*.

    String values use a case-insensitive ``in`` check (fuzzy match).
    All other types use ``==``.
    """
    calls = conv.get_tool_calls(tool_name)
    assert calls, (
        f"'{tool_name}' was never called. Tools called: {conv.tools_called}"
    )

    args = calls[0]["arguments"]
    for key, expected_val in expected.items():
        actual = args.get(key, "")
        if isinstance(expected_val, str) and isinstance(actual, str):
            assert expected_val.lower() in actual.lower(), (
                f"Expected {tool_name}.{key} to contain '{expected_val}', "
                f"got '{actual}'"
            )
        else:
            assert actual == expected_val, (
                f"Expected {tool_name}.{key} == {expected_val!r}, "
                f"got {actual!r}"
            )


def assert_status(conv: Conversation, expected: Any) -> None:
    """Assert the last turn's status matches *expected*."""
    actual = conv.last_status
    if hasattr(actual, "value") and isinstance(expected, str):
        assert actual.value == expected, (
            f"Expected status '{expected}', got '{actual.value}'"
        )
    else:
        assert actual == expected, (
            f"Expected status {expected!r}, got {actual!r}"
        )
