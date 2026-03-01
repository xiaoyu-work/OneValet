"""Integration tests for CloudStorageAgent — routing, field extraction, response quality.

CloudStorageAgent uses InputField-based flow (not tool-based ReAct), so we test:
  - The orchestrator routes to the correct agent type
  - Field extraction produces the correct action/query/provider
  - Response quality via LLM judge

Note: Since CloudStorageAgent doesn't expose tools via the `tools` tuple,
the recorder.tool_calls will be empty. We verify agent routing via
recorder.agent_calls instead.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.productivity]


# ---------------------------------------------------------------------------
# Agent routing — verify orchestrator selects CloudStorageAgent
# ---------------------------------------------------------------------------

ROUTING_CASES = [
    "Find my Q4 report in Google Drive",
    "Show my recent files",
    "How much storage space do I have?",
    "Search for budget.xlsx in Dropbox",
    "Download the project proposal PDF",
]


@pytest.mark.parametrize(
    "user_input",
    ROUTING_CASES,
    ids=[c[:40] for c in ROUTING_CASES],
)
async def test_routes_to_cloud_storage_agent(orchestrator_factory, user_input):
    """The orchestrator should route cloud storage requests to CloudStorageAgent."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", user_input)

    agent_types = [c["agent_type"] for c in recorder.agent_calls]
    assert any("cloud" in t.lower() or "storage" in t.lower() for t in agent_types), (
        f"Expected CloudStorageAgent routing, got agent_calls: {recorder.agent_calls}"
    )


# ---------------------------------------------------------------------------
# Field extraction — verify the agent extracts correct action and query
# ---------------------------------------------------------------------------

async def test_search_action_extracted(orchestrator_factory):
    """A file search request should extract action=search and a query keyword."""
    orch, recorder = await orchestrator_factory()

    # Access the agent directly for field extraction testing
    from onevalet.builtin_agents.cloud_storage.agent import CloudStorageAgent

    agent = CloudStorageAgent.__new__(CloudStorageAgent)
    agent.llm_client = orch.llm_client
    agent.collected_fields = {}

    fields = await agent.extract_fields("Find my Q4 report on Google Drive")
    assert fields.get("action") == "search", f"Expected action=search, got {fields}"
    assert fields.get("query"), f"Expected a query keyword, got {fields}"


async def test_usage_action_extracted(orchestrator_factory):
    """A storage usage request should extract action=usage."""
    orch, recorder = await orchestrator_factory()

    from onevalet.builtin_agents.cloud_storage.agent import CloudStorageAgent

    agent = CloudStorageAgent.__new__(CloudStorageAgent)
    agent.llm_client = orch.llm_client
    agent.collected_fields = {}

    fields = await agent.extract_fields("How much storage space do I have?")
    assert fields.get("action") == "usage", f"Expected action=usage, got {fields}"


async def test_recent_action_extracted(orchestrator_factory):
    """Asking for recent files should extract action=recent."""
    orch, recorder = await orchestrator_factory()

    from onevalet.builtin_agents.cloud_storage.agent import CloudStorageAgent

    agent = CloudStorageAgent.__new__(CloudStorageAgent)
    agent.llm_client = orch.llm_client
    agent.collected_fields = {}

    fields = await agent.extract_fields("Show me my recent files")
    assert fields.get("action") == "recent", f"Expected action=recent, got {fields}"


async def test_share_action_extracts_target(orchestrator_factory):
    """A share request should extract the target email address."""
    orch, recorder = await orchestrator_factory()

    from onevalet.builtin_agents.cloud_storage.agent import CloudStorageAgent

    agent = CloudStorageAgent.__new__(CloudStorageAgent)
    agent.llm_client = orch.llm_client
    agent.collected_fields = {}

    fields = await agent.extract_fields("Share the budget report with alice@example.com")
    assert fields.get("action") == "share", f"Expected action=share, got {fields}"
    assert fields.get("target") == "alice@example.com", (
        f"Expected target=alice@example.com, got {fields}"
    )


async def test_provider_extraction(orchestrator_factory):
    """Mentioning a specific provider should extract it."""
    orch, recorder = await orchestrator_factory()

    from onevalet.builtin_agents.cloud_storage.agent import CloudStorageAgent

    agent = CloudStorageAgent.__new__(CloudStorageAgent)
    agent.llm_client = orch.llm_client
    agent.collected_fields = {}

    fields = await agent.extract_fields("Find my report on Dropbox")
    assert fields.get("provider") == "dropbox", f"Expected provider=dropbox, got {fields}"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_routing(orchestrator_factory, llm_judge):
    """Cloud storage requests should get a relevant response (even if provider is unavailable)."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", "Search for Q4 report in my cloud storage")
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Search for Q4 report in my cloud storage",
        response=response,
        criteria=(
            "The response should be about cloud storage / file search. "
            "It may indicate no accounts are connected or show results. "
            "It should NOT be about an unrelated topic."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
