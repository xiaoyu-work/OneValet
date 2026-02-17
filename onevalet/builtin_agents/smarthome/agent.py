"""
SmartHomeAgent - Domain agent for all smart home control requests.

Replaces the separate LightControlAgent and SpeakerControlAgent
with a single agent that has its own mini ReAct loop. The orchestrator sees
only one "SmartHomeAgent" tool instead of two separate ones.

The internal LLM decides which tools to call (control_lights, control_speaker)
based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import control_lights, control_speaker


@valet(capabilities=["smarthome"])
class SmartHomeAgent(StandardAgent):
    """Control smart lights and speakers. Use when the user wants to turn on/off lights, change brightness or color, play/pause music, or adjust volume."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a smart home control assistant with access to real-time device control tools.

Available tools:
- control_lights: Control Philips Hue lights. Actions: on, off, brightness, color, color_temperature, scene, status.
- control_speaker: Control Sonos speakers. Actions: play, pause, skip_next, skip_previous, volume, mute, unmute, status, play_favorite, favorites.

Today's date: {today} ({weekday})

Instructions:
1. If the user's request is unclear about which device or action, ASK the user for clarification in your text response WITHOUT calling any tools.
2. Once you understand the intent, call the relevant tool with the correct action and parameters.
3. For light commands, always include the action, target (room/light name or "all"), and value (if applicable).
4. For speaker commands, always include the action, and optionally target (speaker/room name) and value.
5. After getting tool results, present them clearly to the user.
6. If a command fails, suggest alternatives or troubleshooting steps."""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    tools = [
        control_lights,
        control_speaker,
    ]
