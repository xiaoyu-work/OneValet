"""Integration tests for ImageAgent.

ImageAgent uses InputField-based flow rather than tools. It routes through
the orchestrator, collects fields (prompt, provider, size, quality), and
transitions through INITIALIZING -> WAITING_FOR_APPROVAL states.

Tests verify:
- The orchestrator routes image requests to ImageAgent
- The agent collects the prompt field correctly
- Response quality is appropriate for image generation requests
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifestyle]


# ---------------------------------------------------------------------------
# Routing — verify orchestrator dispatches to ImageAgent
# ---------------------------------------------------------------------------

ROUTING_CASES = [
    "Generate an image of a sunset over the ocean",
    "Create a picture of a cat wearing a top hat",
    "Draw me a futuristic cityscape at night",
    "Make an image of a mountain landscape in watercolor style",
]


@pytest.mark.parametrize(
    "user_input",
    ROUTING_CASES,
    ids=[c[:40] for c in ROUTING_CASES],
)
async def test_routes_to_image_agent(orchestrator_factory, user_input):
    """The orchestrator should route image generation requests to ImageAgent."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message("test_user", user_input)

    agent_types = [c["agent_type"] for c in recorder.agent_calls]
    assert any("image" in t.lower() for t in agent_types), (
        f"Expected routing to ImageAgent, got agent_calls: {agent_types}"
    )


# ---------------------------------------------------------------------------
# Field extraction — ImageAgent should extract the prompt
# ---------------------------------------------------------------------------

async def test_extracts_prompt_field(orchestrator_factory):
    """ImageAgent should extract the image description as the prompt field."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user",
        "Generate an image of a golden retriever playing in the snow",
    )

    # ImageAgent uses InputField flow; the result should contain the prompt
    # in collected_fields or the approval message should reference it
    response_text = result.raw_message.lower()
    assert "golden retriever" in response_text or "snow" in response_text or "dog" in response_text, (
        f"Expected the response to reference the prompt content, got: {result.raw_message}"
    )


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_generation(orchestrator_factory, llm_judge):
    """Image generation request should produce an appropriate response."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user",
        "Create an image of a Japanese garden with cherry blossoms",
    )

    passed = await llm_judge(
        "Create an image of a Japanese garden with cherry blossoms",
        result.raw_message,
        "The response should acknowledge the image generation request. "
        "It might ask for confirmation/approval, show the extracted prompt, "
        "mention the provider, or indicate the image is being generated. "
        "It should not be an unrelated error or a refusal.",
    )
    assert passed, f"LLM judge failed. Response: {result.raw_message}"
