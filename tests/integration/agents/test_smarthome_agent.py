"""Integration tests for SmartHomeAgent — tool selection, argument extraction, response quality.

SmartHomeAgent tools:
  control_lights  — Philips Hue (on, off, brightness, color, scene, status)
  control_speaker — Sonos (play, pause, skip, volume, mute, status, favorites)
"""

import pytest

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_CASES = [
    ("Turn off the living room lights", ["control_lights"]),
    ("Set the bedroom lights to 50% brightness", ["control_lights"]),
    ("Change the kitchen lights to blue", ["control_lights"]),
    ("Play music on the living room speaker", ["control_speaker"]),
    ("Pause the speaker", ["control_speaker"]),
    ("Set the volume to 30%", ["control_speaker"]),
    ("Turn on all the lights", ["control_lights"]),
    ("Skip to the next song", ["control_speaker"]),
]


@pytest.mark.parametrize(
    "user_input,expected_tools",
    TOOL_SELECTION_CASES,
    ids=[c[0][:40] for c in TOOL_SELECTION_CASES],
)
async def test_tool_selection(orchestrator_factory, user_input, expected_tools):
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", user_input)
    tools_called = [c["tool_name"] for c in recorder.tool_calls]
    assert any(t in tools_called for t in expected_tools), (
        f"Expected one of {expected_tools}, got {tools_called}"
    )


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

async def test_lights_off_extracts_action_and_target(orchestrator_factory):
    """Turning off living room lights should pass action=off and target=living room."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Turn off the living room lights")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "control_lights"]
    assert calls, "Expected control_lights to be called"

    args = calls[0]["arguments"]
    action = args.get("action", "")
    target = args.get("target", "")
    assert action == "off", f"Expected action=off, got {args}"
    assert "living room" in target.lower(), f"Expected 'living room' in target, got {args}"


async def test_lights_brightness_extracts_value(orchestrator_factory):
    """Setting brightness should extract the percentage value."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Set the bedroom lights to 50% brightness")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "control_lights"]
    assert calls, "Expected control_lights to be called"

    args = calls[0]["arguments"]
    action = args.get("action", "")
    assert action == "brightness", f"Expected action=brightness, got {args}"
    # Value should be 50 (or "50")
    value = args.get("value")
    assert value is not None, f"Expected value in args, got {args}"
    assert str(value).strip("%") == "50" or value == 50, (
        f"Expected value=50, got {value}"
    )


async def test_speaker_play_action(orchestrator_factory):
    """Playing music should use control_speaker with action=play."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Play music on the living room speaker")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "control_speaker"]
    assert calls, "Expected control_speaker to be called"

    args = calls[0]["arguments"]
    action = args.get("action", "")
    assert action == "play", f"Expected action=play, got {args}"


async def test_speaker_volume_extracts_value(orchestrator_factory):
    """Setting volume should extract the percentage value."""
    orch, recorder = await orchestrator_factory()
    await orch.handle_message("test_user", "Set the speaker volume to 30%")

    calls = [c for c in recorder.tool_calls if c["tool_name"] == "control_speaker"]
    assert calls, "Expected control_speaker to be called"

    args = calls[0]["arguments"]
    action = args.get("action", "")
    assert action == "volume", f"Expected action=volume, got {args}"
    value = args.get("value")
    assert value is not None, f"Expected value in args, got {args}"


# ---------------------------------------------------------------------------
# Response quality
# ---------------------------------------------------------------------------

async def test_response_quality_lights_off(orchestrator_factory, llm_judge):
    """Turning off lights should confirm the action."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Turn off the living room lights"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Turn off the living room lights",
        response=response,
        criteria=(
            "The response should confirm that the living room lights were turned off. "
            "It should be brief and clear, not ask unnecessary questions."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"


async def test_response_quality_speaker_play(orchestrator_factory, llm_judge):
    """Playing music should confirm playback started."""
    orch, recorder = await orchestrator_factory()
    result = await orch.handle_message(
        "test_user", "Play music on the living room speaker"
    )
    response = result.raw_message if hasattr(result, "raw_message") else str(result)

    passed = await llm_judge(
        user_input="Play music on the living room speaker",
        response=response,
        criteria=(
            "The response should confirm that music playback started on the "
            "living room speaker. It should be concise."
        ),
    )
    assert passed, f"Response quality check failed. Response: {response}"
