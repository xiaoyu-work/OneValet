"""
Speaker Control Agent - Control Sonos speakers

This agent handles speaker commands:
- Play, pause, skip tracks
- Volume control (set, up, down, mute, unmute)
- Playback status
- Favorites management

This is an instant-control agent (no approval needed).
"""
import logging
import json
from typing import Dict, Any

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class SpeakerControlAgent(StandardAgent):
    """Control smart speakers (play, pause, volume, skip). Use when the user wants to control music playback."""

    action = InputField(
        prompt="What would you like to do?",
        description="Speaker action: play, pause, skip, volume, status, favorites",
    )
    target = InputField(
        prompt="Which speaker?",
        description="Speaker or room name",
        required=False,
    )
    value = InputField(
        prompt="What value?",
        description="Volume level, track name, or favorite name",
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
        """Extract speaker action, target, and value from natural language."""
        if not self.llm_client:
            return {"action": user_input}

        try:
            prompt = f"""Extract speaker control intent from the user's message.

User message: "{user_input}"

Possible actions:
- "play" : resume or start playback
- "pause" : pause playback
- "skip_next" : skip to next track
- "skip_previous" : go to previous track
- "volume" : set volume level
- "mute" : mute speaker
- "unmute" : unmute speaker
- "status" : check what's currently playing
- "play_favorite" : play a specific favorite/playlist
- "favorites" : list saved favorites

Extract:
1. action: one of the actions above
2. target: speaker or room name (null if not specified)
3. value: volume level (0-100), "up", "down", track/favorite name (null if not applicable)

Examples:
- "play some music" -> {{"action": "play", "target": null, "value": null}}
- "pause" -> {{"action": "pause", "target": null, "value": null}}
- "next song" -> {{"action": "skip_next", "target": null, "value": null}}
- "previous song" -> {{"action": "skip_previous", "target": null, "value": null}}
- "volume to 50" -> {{"action": "volume", "target": null, "value": "50"}}
- "turn it up" -> {{"action": "volume", "target": null, "value": "up"}}
- "turn it down" -> {{"action": "volume", "target": null, "value": "down"}}
- "mute the kitchen speaker" -> {{"action": "mute", "target": "kitchen", "value": null}}
- "unmute" -> {{"action": "unmute", "target": null, "value": null}}
- "what's playing?" -> {{"action": "status", "target": null, "value": null}}
- "play my jazz playlist" -> {{"action": "play_favorite", "target": null, "value": "jazz"}}
- "play jazz in the bedroom" -> {{"action": "play_favorite", "target": "bedroom", "value": "jazz"}}
- "list my favorites" -> {{"action": "favorites", "target": null, "value": null}}

Return ONLY the JSON object, no explanations.

JSON Output:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract speaker control intent from text and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format="json_object",
                enable_thinking=False,
            )

            extracted = json.loads(result.content.strip())

            fields = {}
            if extracted.get("action"):
                fields["action"] = extracted["action"]
            if extracted.get("target"):
                fields["target"] = extracted["target"]
            if extracted.get("value") is not None:
                fields["value"] = str(extracted["value"])

            if not fields.get("action"):
                fields["action"] = user_input

            logger.info(f"Extracted speaker fields: {fields}")
            return fields

        except Exception as e:
            logger.error(f"Field extraction failed: {e}", exc_info=True)
            return {"action": user_input}

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute speaker control action."""
        from onevalet.providers.email.resolver import AccountResolver
        from onevalet.providers.smarthome.sonos import SonosProvider

        fields = self.collected_fields
        action = fields.get("action", "")
        target = fields.get("target")
        value = fields.get("value")

        logger.info(f"Speaker action: {action}, target: {target}, value: {value}")

        try:
            # Resolve Sonos account
            resolver = AccountResolver()
            account = await resolver.credential_store.get(self.tenant_id, "sonos", "primary")

            if not account:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No Sonos account found. Please connect your Sonos in settings first.",
                )

            provider = SonosProvider(account)

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I lost access to your Sonos account. Could you reconnect it in settings?",
                )

            # Get default group/player if target not specified
            groups = await provider.get_groups()
            if not groups:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I couldn't find any Sonos speakers. Make sure they're powered on and connected.",
                )

            group = self._find_group(groups, target)
            if not group:
                available = ", ".join(g.get("name", "Unknown") for g in groups)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I couldn't find a speaker called \"{target}\". Available: {available}",
                )

            group_id = group.get("id")
            group_name = group.get("name", "your speaker")

            # Execute action
            if action == "play":
                await provider.play(group_id)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Playing on {group_name}.",
                )

            elif action == "pause":
                await provider.pause(group_id)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Paused {group_name}.",
                )

            elif action == "skip_next":
                await provider.skip_to_next(group_id)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Skipped to the next track on {group_name}.",
                )

            elif action == "skip_previous":
                await provider.skip_to_previous(group_id)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Went back to the previous track on {group_name}.",
                )

            elif action == "volume":
                response = await self._handle_volume(provider, group_id, group_name, value)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=response,
                )

            elif action == "mute":
                await provider.set_mute(group_id, muted=True)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Muted {group_name}.",
                )

            elif action == "unmute":
                await provider.set_mute(group_id, muted=False)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Unmuted {group_name}.",
                )

            elif action == "status":
                response = await self._handle_status(provider, group_id, group_name)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=response,
                )

            elif action == "play_favorite":
                response = await self._handle_play_favorite(provider, group_id, group_name, value)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=response,
                )

            elif action == "favorites":
                response = await self._handle_list_favorites(provider)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=response,
                )

            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I'm not sure how to do \"{action}\". "
                    "Try play, pause, skip, volume, mute, status, or favorites.",
                )

        except Exception as e:
            logger.error(f"Speaker control failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong controlling your speaker. Want me to try again?",
            )

    def _find_group(self, groups: list, target: str | None) -> dict | None:
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

    async def _handle_volume(
        self, provider, group_id: str, group_name: str, value: str | None
    ) -> str:
        """Handle volume set/up/down."""
        if value and value.lower() == "up":
            status = await provider.get_playback_status(group_id)
            current = status.get("volume", 50)
            new_volume = min(100, current + 10)
            await provider.set_volume(group_id, new_volume)
            return f"Volume up to {new_volume}% on {group_name}."

        elif value and value.lower() == "down":
            status = await provider.get_playback_status(group_id)
            current = status.get("volume", 50)
            new_volume = max(0, current - 10)
            await provider.set_volume(group_id, new_volume)
            return f"Volume down to {new_volume}% on {group_name}."

        elif value and value.isdigit():
            level = max(0, min(100, int(value)))
            await provider.set_volume(group_id, level)
            return f"Volume set to {level}% on {group_name}."

        else:
            return "What volume level? You can say a number (0-100), \"up\", or \"down\"."

    async def _handle_status(self, provider, group_id: str, group_name: str) -> str:
        """Get and format current playback status."""
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
        else:
            if playback_state != "PLAYBACK_STATE_IDLE":
                parts.append("No track info available.")

        if volume is not None:
            parts.append(f"Volume: {volume}%")

        return "\n".join(parts)

    async def _handle_play_favorite(
        self, provider, group_id: str, group_name: str, value: str | None
    ) -> str:
        """Find and play a matching favorite."""
        if not value:
            return "Which favorite would you like to play? Say \"list my favorites\" to see them."

        favorites = await provider.get_favorites()
        if not favorites:
            return "No favorites found in your Sonos account."

        # Find best match
        value_lower = value.lower()
        match = None
        for fav in favorites:
            fav_name = fav.get("name", "").lower()
            if value_lower in fav_name or fav_name in value_lower:
                match = fav
                break

        if not match:
            available = ", ".join(f.get("name", "Unknown") for f in favorites[:10])
            return f"I couldn't find a favorite matching \"{value}\". Your favorites: {available}"

        await provider.play_favorite(group_id, match.get("id"))
        return f"Playing \"{match.get('name')}\" on {group_name}."

    async def _handle_list_favorites(self, provider) -> str:
        """List all saved favorites."""
        favorites = await provider.get_favorites()
        if not favorites:
            return "No favorites found in your Sonos account."

        parts = [f"Your Sonos favorites ({len(favorites)}):\n"]
        for i, fav in enumerate(favorites, 1):
            name = fav.get("name", "Unknown")
            parts.append(f"{i}. {name}")

        return "\n".join(parts)
