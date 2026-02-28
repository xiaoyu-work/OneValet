"""Integration tests for orchestrator routing.

Verifies that the orchestrator correctly routes user messages to the
expected agent type using a real LLM.  Each test case sends a natural-
language request and asserts that at least one of the expected agents
was invoked.

Requires INTEGRATION_TEST_API_KEY to be set (see tests/integration/README.md).
"""

import pytest

pytestmark = [pytest.mark.integration]

ROUTING_CASES = [
    ("What's on my calendar today?", ["CalendarAgent"]),
    ("Send an email to john@example.com", ["EmailAgent"]),
    ("I spent $15 on lunch", ["ExpenseAgent"]),
    ("How much did I spend this month?", ["ExpenseAgent"]),
    ("Set a budget of $500 for food", ["ExpenseAgent"]),
    ("Set a reminder to call mom tomorrow", ["TodoAgent"]),
    ("Add buy groceries to my todo list", ["TodoAgent"]),
    ("Find a good Italian restaurant nearby", ["MapsAgent"]),
    ("How do I get to the airport from here?", ["MapsAgent"]),
    ("Track my package 1Z999AA10123456784", ["ShippingAgent"]),
    ("What's my morning briefing?", ["BriefingAgent"]),
    ("Set up daily briefing at 8am", ["BriefingAgent"]),
    ("Plan a 3-day trip to Tokyo", ["TripPlannerAgent"]),
    ("Turn off the living room lights", ["SmartHomeAgent"]),
    ("Create a GitHub issue for the login bug", ["GitHubComposioAgent"]),
    ("Post a tweet about our new product", ["TwitterComposioAgent"]),
    ("Send a Slack message to the engineering channel", ["SlackComposioAgent"]),
    ("Search my Google Drive for the Q4 report", ["GoogleWorkspaceAgent", "CloudStorageAgent"]),
    ("Schedule a recurring task every Monday at 9am", ["CronAgent", "TodoAgent"]),
    ("Search my Notion for meeting notes", ["NotionAgent"]),
    ("Generate an image of a sunset over the ocean", ["ImageAgent"]),
]


@pytest.mark.parametrize(
    "user_input,expected_agents",
    ROUTING_CASES,
    ids=[c[0][:40] for c in ROUTING_CASES],
)
async def test_routes_to_correct_agent(orchestrator_factory, user_input, expected_agents):
    """The orchestrator should delegate the user message to the correct agent."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(tenant_id="test_user", message=user_input)

    routed_agents = [c["agent_type"] for c in recorder.agent_calls]
    tool_names = [tc["tool_name"] for tc in recorder.tool_calls]
    print(f"\n  routed={routed_agents} tools={tool_names}")
    print(f"  response={result.raw_message[:200] if result.raw_message else '(empty)'}")
    assert any(agent in routed_agents for agent in expected_agents), (
        f"Expected one of {expected_agents}, got {routed_agents}"
    )
