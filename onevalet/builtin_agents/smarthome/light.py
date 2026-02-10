"""
Light Control Agent - Control Philips Hue lights

Handles natural language commands for smart lighting:
- Turn lights on/off
- Set brightness levels
- Change colors
- Adjust color temperature
- Activate scenes
- Query light status

This is an instant-action agent (no approval needed).
"""
import logging
import json
from typing import Any, Dict, Optional, Tuple

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class LightControlAgent(StandardAgent):
    """Smart light control agent for Philips Hue"""

    action = InputField(
        prompt="What would you like to do with the lights?",
        description="Light action: on, off, brightness, color, scene, status",
    )
    target = InputField(
        prompt="Which light or room?",
        description="Light name, room name, or 'all'",
        required=False,
    )
    value = InputField(
        prompt="What value?",
        description="Brightness (0-100), color name, or scene name",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs,
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Use LLM to parse natural language into structured light commands."""
        if not self.llm_client:
            return {"action": user_input}

        try:
            prompt = f"""Extract a smart-light control command from the user's message.

User message: "{user_input}"

Return a JSON object with these fields:
- action: one of "on", "off", "brightness", "color", "color_temperature", "scene", "status"
- target: light name, room name, or "all" (default "all" if not specified)
- value: brightness 0-100, color name, scene name, or temperature description (warm/cool/neutral/daylight)

Examples:
- "turn off living room lights" -> {{"action": "off", "target": "living room"}}
- "set bedroom to 50% brightness" -> {{"action": "brightness", "target": "bedroom", "value": "50"}}
- "make kitchen warm white" -> {{"action": "color_temperature", "target": "kitchen", "value": "warm"}}
- "set lights to blue" -> {{"action": "color", "target": "all", "value": "blue"}}
- "activate movie scene" -> {{"action": "scene", "value": "movie"}}
- "what lights are on?" -> {{"action": "status"}}
- "dim the lights to 30" -> {{"action": "brightness", "target": "all", "value": "30"}}
- "turn on the porch light" -> {{"action": "on", "target": "porch"}}

Return ONLY the JSON object, no explanations.

JSON Output:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract smart-light commands from text and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format="json_object",
                enable_thinking=False,
            )

            content = result.content.strip()
            extracted = json.loads(content)

            if not extracted or "action" not in extracted:
                extracted = {"action": user_input}

            # Normalise target default
            if "target" not in extracted or not extracted["target"]:
                extracted["target"] = "all"

            logger.info(f"Extracted light command: {extracted}")
            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}", exc_info=True)
            return {"action": user_input}

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute the light control command."""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.smarthome.philips_hue import PhilipsHueProvider

        fields = self.collected_fields
        action = fields.get("action", "").lower().strip()
        target = fields.get("target", "all").strip()
        value = fields.get("value", "").strip() if fields.get("value") else None

        logger.info(f"Light control: action={action}, target={target}, value={value}")

        try:
            # Resolve Philips Hue credentials
            resolver = AccountResolver()
            account = await resolver._resolve_account_for_service(
                self.tenant_id, "philips_hue", "primary"
            )

            if not account:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No Philips Hue account found. Please connect your Hue Bridge in settings first.",
                )

            provider = PhilipsHueProvider(credentials=account)

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Your Hue connection has expired. Please reconnect in settings.",
                )

            # Dispatch by action
            if action == "status":
                return await self._handle_status(provider, target)
            elif action in ("on", "off"):
                return await self._handle_on_off(provider, action, target)
            elif action == "brightness":
                return await self._handle_brightness(provider, target, value)
            elif action == "color":
                return await self._handle_color(provider, target, value)
            elif action == "color_temperature":
                return await self._handle_color_temperature(provider, target, value)
            elif action == "scene":
                return await self._handle_scene(provider, value)
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I'm not sure how to handle the action \"{action}\". "
                                "Try: on, off, brightness, color, scene, or status.",
                )

        except ImportError:
            logger.error("PhilipsHueProvider not available")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Philips Hue support is not available yet. Please check back later.",
            )
        except Exception as e:
            logger.error(f"Light control failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong controlling your lights. Want me to try again?",
            )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _handle_status(self, provider, target: str) -> AgentResult:
        """List lights and their current states."""
        try:
            lights = await provider.list_lights()
            if not lights.get("success"):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Couldn't retrieve your lights. Is your Hue Bridge online?",
                )

            light_list = lights.get("data", [])
            if not light_list:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No lights found on your Hue Bridge.",
                )

            # Filter by target if not "all"
            if target and target.lower() != "all":
                light_list = [
                    lt for lt in light_list
                    if target.lower() in lt.get("name", "").lower()
                    or target.lower() in lt.get("room", "").lower()
                ]

            if not light_list:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No lights found matching \"{target}\".",
                )

            lines = [f"Found {len(light_list)} light(s):\n"]
            for lt in light_list:
                name = lt.get("name", "Unknown")
                state = "on" if lt.get("on") else "off"
                brightness = lt.get("brightness")
                room = lt.get("room", "")
                room_str = f" ({room})" if room else ""

                line = f"- {name}{room_str}: {state}"
                if brightness is not None and lt.get("on"):
                    line += f", {brightness}% brightness"
                lines.append(line)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="\n".join(lines),
            )
        except Exception as e:
            logger.error(f"Status check failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Had trouble checking your light status. Try again?",
            )

    async def _handle_on_off(self, provider, action: str, target: str) -> AgentResult:
        """Turn lights on or off."""
        on = action == "on"

        try:
            if target and target.lower() != "all":
                result = await provider.control_room(
                    room_name=target,
                    on=on,
                )
            else:
                if on:
                    result = await provider.turn_on()
                else:
                    result = await provider.turn_off()

            if result.get("success"):
                state_word = "on" if on else "off"
                target_display = target if target.lower() != "all" else "all lights"
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Turned {state_word} {target_display}.",
                )
            else:
                error = result.get("error", "Unknown error")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't turn {action} the lights: {error}",
                )
        except Exception as e:
            logger.error(f"On/off failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Had trouble turning {action} the lights. Try again?",
            )

    async def _handle_brightness(self, provider, target: str, value: Optional[str]) -> AgentResult:
        """Set light brightness."""
        if not value:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="What brightness level? Please specify a value from 0 to 100.",
            )

        try:
            brightness = int(value.replace("%", "").strip())
            brightness = max(0, min(100, brightness))
        except ValueError:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I couldn't understand \"{value}\" as a brightness level. Use a number from 0 to 100.",
            )

        try:
            if target and target.lower() != "all":
                result = await provider.control_room(
                    room_name=target,
                    on=True,
                    brightness=brightness,
                )
            else:
                result = await provider.set_brightness(brightness=brightness)

            if result.get("success"):
                target_display = target if target.lower() != "all" else "all lights"
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Set {target_display} to {brightness}% brightness.",
                )
            else:
                error = result.get("error", "Unknown error")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't set brightness: {error}",
                )
        except Exception as e:
            logger.error(f"Brightness control failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Had trouble setting brightness. Try again?",
            )

    async def _handle_color(self, provider, target: str, value: Optional[str]) -> AgentResult:
        """Set light color."""
        if not value:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="What color would you like? Try red, blue, green, purple, etc.",
            )

        rgb = _color_name_to_rgb(value)
        if not rgb:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I don't recognize the color \"{value}\". "
                            "Try: red, blue, green, yellow, orange, purple, pink, or white.",
            )

        try:
            if target and target.lower() != "all":
                result = await provider.control_room(
                    room_name=target,
                    on=True,
                    color=rgb,
                )
            else:
                result = await provider.set_color(color=rgb)

            if result.get("success"):
                target_display = target if target.lower() != "all" else "all lights"
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Set {target_display} to {value}.",
                )
            else:
                error = result.get("error", "Unknown error")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't set color: {error}",
                )
        except Exception as e:
            logger.error(f"Color control failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Had trouble changing the color. Try again?",
            )

    async def _handle_color_temperature(self, provider, target: str, value: Optional[str]) -> AgentResult:
        """Set light color temperature."""
        if not value:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="What temperature? Try: warm, neutral, cool, or daylight.",
            )

        mirek = _temp_name_to_mirek(value)
        if mirek is None:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"I don't recognize \"{value}\" as a color temperature. "
                            "Try: warm, neutral, cool, or daylight.",
            )

        try:
            result = await provider.set_color_temperature(
                mirek=mirek,
                room_name=target if target.lower() != "all" else None,
            )

            if result.get("success"):
                target_display = target if target.lower() != "all" else "all lights"
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Set {target_display} to {value} white.",
                )
            else:
                error = result.get("error", "Unknown error")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't set color temperature: {error}",
                )
        except Exception as e:
            logger.error(f"Color temperature control failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Had trouble changing color temperature. Try again?",
            )

    async def _handle_scene(self, provider, value: Optional[str]) -> AgentResult:
        """Find and activate a scene."""
        if not value:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Which scene would you like to activate? E.g., movie, relax, energize.",
            )

        try:
            # List available scenes and find a match
            scenes_result = await provider.list_scenes()
            if not scenes_result.get("success"):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Couldn't retrieve your scenes. Is your Hue Bridge online?",
                )

            scenes = scenes_result.get("data", [])
            matched_scene = None
            for scene in scenes:
                scene_name = scene.get("name", "").lower()
                if value.lower() in scene_name or scene_name in value.lower():
                    matched_scene = scene
                    break

            if not matched_scene:
                scene_names = [s.get("name", "") for s in scenes[:10]]
                if scene_names:
                    names_str = ", ".join(scene_names)
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"No scene matching \"{value}\" found. Available scenes: {names_str}",
                    )
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No scene matching \"{value}\" found and no scenes available.",
                )

            result = await provider.activate_scene(scene_id=matched_scene.get("id"))

            if result.get("success"):
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Activated scene \"{matched_scene.get('name')}\".",
                )
            else:
                error = result.get("error", "Unknown error")
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't activate scene: {error}",
                )
        except Exception as e:
            logger.error(f"Scene activation failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Had trouble activating that scene. Try again?",
            )


# ------------------------------------------------------------------
# Helpers (module-level)
# ------------------------------------------------------------------

_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "white": (255, 255, 255),
    "warm white": (255, 214, 170),
    "cool white": (200, 220, 255),
}


def _color_name_to_rgb(name: str) -> Optional[Tuple[int, int, int]]:
    """Map common color names to (R, G, B) tuples."""
    return _COLOR_MAP.get(name.lower().strip())


_TEMP_MAP: Dict[str, int] = {
    "warm": 400,
    "warm white": 400,
    "neutral": 300,
    "cool": 200,
    "cool white": 200,
    "daylight": 153,
}


def _temp_name_to_mirek(name: str) -> Optional[int]:
    """Map temperature description to Hue mirek value."""
    return _TEMP_MAP.get(name.lower().strip())
