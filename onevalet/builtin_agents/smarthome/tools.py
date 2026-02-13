"""
SmartHome Domain Tools - Standalone API functions for SmartHomeAgent's mini ReAct loop.

Extracted from LightControlAgent and SpeakerControlAgent.
Each function takes (args: dict, context: DomainToolContext) -> str.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from onevalet.agents.domain_agent import DomainToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

async def _resolve_hue_provider(context: DomainToolContext):
    """Resolve Philips Hue credentials and return a ready provider, or None."""
    from onevalet.providers.email.resolver import AccountResolver
    from onevalet.providers.smarthome.philips_hue import PhilipsHueProvider

    resolver = AccountResolver()
    account = await resolver._resolve_account_for_service(
        context.tenant_id, "philips_hue", "primary"
    )
    if not account:
        return None

    provider = PhilipsHueProvider(credentials=account)
    if not await provider.ensure_valid_token():
        return None

    return provider


async def _resolve_sonos_provider(context: DomainToolContext):
    """Resolve Sonos credentials and return a ready provider, or None."""
    from onevalet.providers.email.resolver import AccountResolver
    from onevalet.providers.smarthome.sonos import SonosProvider

    resolver = AccountResolver()
    account = await resolver.credential_store.get(context.tenant_id, "sonos", "primary")
    if not account:
        return None

    provider = SonosProvider(account)
    if not await provider.ensure_valid_token():
        return None

    return provider


# =============================================================================
# Light Helpers
# =============================================================================

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


# =============================================================================
# control_lights
# =============================================================================

async def control_lights(args: dict, context: DomainToolContext) -> str:
    """Control Philips Hue lights: on/off, brightness, color, color_temperature, scene, status."""
    action = args.get("action", "").lower().strip()
    target = args.get("target", "all").strip()
    value = args.get("value", "").strip() if args.get("value") else None

    logger.info(f"Light control: action={action}, target={target}, value={value}")

    try:
        provider = await _resolve_hue_provider(context)
        if not provider:
            return "No Philips Hue account found or connection expired. Please connect your Hue Bridge in settings."

        # --- status ---
        if action == "status":
            lights = await provider.list_lights()
            if not lights.get("success"):
                return "Couldn't retrieve your lights. Is your Hue Bridge online?"

            light_list = lights.get("data", [])
            if not light_list:
                return "No lights found on your Hue Bridge."

            if target and target.lower() != "all":
                light_list = [
                    lt for lt in light_list
                    if target.lower() in lt.get("name", "").lower()
                    or target.lower() in lt.get("room", "").lower()
                ]

            if not light_list:
                return f'No lights found matching "{target}".'

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

            return "\n".join(lines)

        # --- on / off ---
        elif action in ("on", "off"):
            on = action == "on"
            if target and target.lower() != "all":
                result = await provider.control_room(room_name=target, on=on)
            else:
                result = await provider.turn_on() if on else await provider.turn_off()

            if result.get("success"):
                state_word = "on" if on else "off"
                target_display = target if target.lower() != "all" else "all lights"
                return f"Turned {state_word} {target_display}."
            else:
                return f"Couldn't turn {action} the lights: {result.get('error', 'Unknown error')}"

        # --- brightness ---
        elif action == "brightness":
            if not value:
                return "What brightness level? Please specify a value from 0 to 100."

            try:
                brightness = int(value.replace("%", "").strip())
                brightness = max(0, min(100, brightness))
            except ValueError:
                return f'I couldn\'t understand "{value}" as a brightness level. Use a number from 0 to 100.'

            if target and target.lower() != "all":
                result = await provider.control_room(room_name=target, on=True, brightness=brightness)
            else:
                result = await provider.set_brightness(brightness=brightness)

            if result.get("success"):
                target_display = target if target.lower() != "all" else "all lights"
                return f"Set {target_display} to {brightness}% brightness."
            else:
                return f"Couldn't set brightness: {result.get('error', 'Unknown error')}"

        # --- color ---
        elif action == "color":
            if not value:
                return "What color would you like? Try red, blue, green, purple, etc."

            rgb = _color_name_to_rgb(value)
            if not rgb:
                return (
                    f'I don\'t recognize the color "{value}". '
                    "Try: red, blue, green, yellow, orange, purple, pink, or white."
                )

            if target and target.lower() != "all":
                result = await provider.control_room(room_name=target, on=True, color=rgb)
            else:
                result = await provider.set_color(color=rgb)

            if result.get("success"):
                target_display = target if target.lower() != "all" else "all lights"
                return f"Set {target_display} to {value}."
            else:
                return f"Couldn't set color: {result.get('error', 'Unknown error')}"

        # --- color_temperature ---
        elif action == "color_temperature":
            if not value:
                return "What temperature? Try: warm, neutral, cool, or daylight."

            mirek = _temp_name_to_mirek(value)
            if mirek is None:
                return (
                    f'I don\'t recognize "{value}" as a color temperature. '
                    "Try: warm, neutral, cool, or daylight."
                )

            result = await provider.set_color_temperature(
                mirek=mirek,
                room_name=target if target.lower() != "all" else None,
            )

            if result.get("success"):
                target_display = target if target.lower() != "all" else "all lights"
                return f"Set {target_display} to {value} white."
            else:
                return f"Couldn't set color temperature: {result.get('error', 'Unknown error')}"

        # --- scene ---
        elif action == "scene":
            if not value:
                return "Which scene would you like to activate? E.g., movie, relax, energize."

            scenes_result = await provider.list_scenes()
            if not scenes_result.get("success"):
                return "Couldn't retrieve your scenes. Is your Hue Bridge online?"

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
                    return f'No scene matching "{value}" found. Available scenes: {", ".join(scene_names)}'
                return f'No scene matching "{value}" found and no scenes available.'

            result = await provider.activate_scene(scene_id=matched_scene.get("id"))
            if result.get("success"):
                return f'Activated scene "{matched_scene.get("name")}".'
            else:
                return f"Couldn't activate scene: {result.get('error', 'Unknown error')}"

        else:
            return (
                f'I\'m not sure how to handle the action "{action}". '
                "Try: on, off, brightness, color, color_temperature, scene, or status."
            )

    except ImportError:
        logger.error("PhilipsHueProvider not available")
        return "Philips Hue support is not available yet. Please check back later."
    except Exception as e:
        logger.error(f"Light control failed: {e}", exc_info=True)
        return "Something went wrong controlling your lights. Want me to try again?"


# =============================================================================
# control_speaker
# =============================================================================

def _find_group(groups: list, target: str | None) -> dict | None:
    """Find the matching speaker group by name, or return the first one."""
    if not groups:
        return None
    if not target:
        return groups[0]

    target_lower = target.lower()
    for group in groups:
        name = group.get("name", "").lower()
        if target_lower in name or name in target_lower:
            return group

    return None


async def control_speaker(args: dict, context: DomainToolContext) -> str:
    """Control Sonos speakers: play, pause, skip, volume, mute, unmute, status, play_favorite, favorites."""
    action = args.get("action", "").lower().strip()
    target = args.get("target")
    value = args.get("value")

    logger.info(f"Speaker control: action={action}, target={target}, value={value}")

    try:
        provider = await _resolve_sonos_provider(context)
        if not provider:
            return "No Sonos account found or connection expired. Please connect your Sonos in settings."

        groups = await provider.get_groups()
        if not groups:
            return "I couldn't find any Sonos speakers. Make sure they're powered on and connected."

        group = _find_group(groups, target)
        if not group:
            available = ", ".join(g.get("name", "Unknown") for g in groups)
            return f'I couldn\'t find a speaker called "{target}". Available: {available}'

        group_id = group.get("id")
        group_name = group.get("name", "your speaker")

        # --- play ---
        if action == "play":
            await provider.play(group_id)
            return f"Playing on {group_name}."

        # --- pause ---
        elif action == "pause":
            await provider.pause(group_id)
            return f"Paused {group_name}."

        # --- skip_next ---
        elif action == "skip_next":
            await provider.skip_to_next(group_id)
            return f"Skipped to the next track on {group_name}."

        # --- skip_previous ---
        elif action == "skip_previous":
            await provider.skip_to_previous(group_id)
            return f"Went back to the previous track on {group_name}."

        # --- volume ---
        elif action == "volume":
            if value and str(value).lower() == "up":
                status = await provider.get_playback_status(group_id)
                current = status.get("volume", 50)
                new_volume = min(100, current + 10)
                await provider.set_volume(group_id, new_volume)
                return f"Volume up to {new_volume}% on {group_name}."
            elif value and str(value).lower() == "down":
                status = await provider.get_playback_status(group_id)
                current = status.get("volume", 50)
                new_volume = max(0, current - 10)
                await provider.set_volume(group_id, new_volume)
                return f"Volume down to {new_volume}% on {group_name}."
            elif value and str(value).isdigit():
                level = max(0, min(100, int(value)))
                await provider.set_volume(group_id, level)
                return f"Volume set to {level}% on {group_name}."
            else:
                return 'What volume level? You can say a number (0-100), "up", or "down".'

        # --- mute ---
        elif action == "mute":
            await provider.set_mute(group_id, muted=True)
            return f"Muted {group_name}."

        # --- unmute ---
        elif action == "unmute":
            await provider.set_mute(group_id, muted=False)
            return f"Unmuted {group_name}."

        # --- status ---
        elif action == "status":
            status = await provider.get_playback_status(group_id)
            playback_state = status.get("playback_state", "unknown")
            track = status.get("track", {})
            title = track.get("name", "")
            artist = track.get("artist", "")
            album = track.get("album", "")
            volume = status.get("volume")

            parts = [f"Speaker: {group_name}"]
            if playback_state == "PLAYBACK_STATE_PLAYING":
                parts.append("Status: Playing")
            elif playback_state == "PLAYBACK_STATE_PAUSED":
                parts.append("Status: Paused")
            elif playback_state == "PLAYBACK_STATE_IDLE":
                parts.append("Status: Idle")
            else:
                parts.append(f"Status: {playback_state}")

            if title:
                track_info = title
                if artist:
                    track_info += f" by {artist}"
                if album:
                    track_info += f" ({album})"
                parts.append(f"Now playing: {track_info}")
            elif playback_state != "PLAYBACK_STATE_IDLE":
                parts.append("No track info available.")

            if volume is not None:
                parts.append(f"Volume: {volume}%")

            return "\n".join(parts)

        # --- play_favorite ---
        elif action == "play_favorite":
            if not value:
                return 'Which favorite would you like to play? Say "list my favorites" to see them.'

            favorites = await provider.get_favorites()
            if not favorites:
                return "No favorites found in your Sonos account."

            value_lower = str(value).lower()
            match = None
            for fav in favorites:
                fav_name = fav.get("name", "").lower()
                if value_lower in fav_name or fav_name in value_lower:
                    match = fav
                    break

            if not match:
                available = ", ".join(f.get("name", "Unknown") for f in favorites[:10])
                return f'I couldn\'t find a favorite matching "{value}". Your favorites: {available}'

            await provider.play_favorite(group_id, match.get("id"))
            return f'Playing "{match.get("name")}" on {group_name}.'

        # --- favorites ---
        elif action == "favorites":
            favorites = await provider.get_favorites()
            if not favorites:
                return "No favorites found in your Sonos account."

            parts = [f"Your Sonos favorites ({len(favorites)}):\n"]
            for i, fav in enumerate(favorites, 1):
                parts.append(f"{i}. {fav.get('name', 'Unknown')}")
            return "\n".join(parts)

        else:
            return (
                f'I\'m not sure how to do "{action}". '
                "Try play, pause, skip_next, skip_previous, volume, mute, unmute, status, play_favorite, or favorites."
            )

    except ImportError:
        logger.error("SonosProvider not available")
        return "Sonos support is not available yet. Please check back later."
    except Exception as e:
        logger.error(f"Speaker control failed: {e}", exc_info=True)
        return "Something went wrong controlling your speaker. Want me to try again?"
