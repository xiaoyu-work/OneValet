"""Integration tests for orchestrator edge cases.

Covers ambiguous input, multi-item requests, unknown intents, very short
input, and purely conversational messages.  Uses a real LLM to verify the
orchestrator handles these gracefully.

Requires INTEGRATION_TEST_API_KEY to be set (see tests/integration/README.md).
"""

import pytest

from onevalet.result import AgentStatus

pytestmark = [pytest.mark.integration]


async def test_ambiguous_input_completes(orchestrator_factory):
    """Ambiguous input should be routed to a reasonable agent and complete
    without error, even if the exact agent choice is debatable."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        tenant_id="test_user",
        message="remind me to check email",
    )

    routed_agents = [c["agent_type"] for c in recorder.agent_calls]
    acceptable = {"TodoAgent", "EmailAgent"}
    assert any(a in acceptable for a in routed_agents), (
        f"Expected TodoAgent or EmailAgent for ambiguous reminder/email input, "
        f"got {routed_agents}"
    )
    assert result.status != AgentStatus.ERROR, (
        f"Ambiguous input should not cause an error, but got status={result.status}"
    )


async def test_multiple_expenses_calls_tool_multiple_times(orchestrator_factory):
    """Listing several expenses in one message should invoke log_expense
    at least twice."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        tenant_id="test_user",
        message="lunch $15, uber $12, coffee $5",
    )

    log_expense_calls = [
        tc for tc in recorder.tool_calls if tc["tool_name"] == "log_expense"
    ]
    assert len(log_expense_calls) >= 2, (
        f"Expected log_expense to be called at least 2 times for multiple "
        f"expenses, but it was called {len(log_expense_calls)} time(s). "
        f"All tool calls: {[tc['tool_name'] for tc in recorder.tool_calls]}"
    )


async def test_unknown_intent_does_not_crash(orchestrator_factory):
    """A message with no clear agent intent should still complete
    gracefully (status COMPLETED, not ERROR)."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        tenant_id="test_user",
        message="tell me a joke",
    )

    assert result.status == AgentStatus.COMPLETED, (
        f"Unknown intent should complete gracefully, but got status={result.status}"
    )


async def test_very_short_input_routes_correctly(orchestrator_factory):
    """A single keyword should still be routed to the right agent."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        tenant_id="test_user",
        message="expenses",
    )

    routed_agents = [c["agent_type"] for c in recorder.agent_calls]
    assert "ExpenseAgent" in routed_agents, (
        f"Short input 'expenses' should route to ExpenseAgent, "
        f"got {routed_agents}"
    )


async def test_conversational_message_completes_with_text(orchestrator_factory):
    """A purely conversational greeting should complete with a non-empty
    text response.  It may or may not invoke an agent."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        tenant_id="test_user",
        message="hello, how are you?",
    )

    assert result.status == AgentStatus.COMPLETED, (
        f"Conversational message should complete, but got status={result.status}"
    )
    assert result.raw_message, (
        "Conversational message should produce a non-empty text response"
    )
