"""Prompt assembly — builds the message list sent to the model.

Gathers the system prompt, the user's profile and long-term memories, the
session's recent turns, and the conversation history into the ordered message
list a request is sent with.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..memory.true_memory import format_true_memory_for_prompt
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)


class PromptBuilderMixin:
    """Mixin assembling the LLM message list for a request.

    Expects the following on ``self`` (provided by Orchestrator):
    - ``system_prompt``, ``system_prompt_mode``
    - ``momex``, ``session_memory``, ``memory_governance``
    - ``_agent_registry``, ``credential_store``
    """

    async def _build_llm_messages(
        self,
        context: Dict[str, Any],
        user_message: str,
        *,
        needs_memory: bool = True,
        include_planning: bool = False,
        approved_plan: str = "",
        pending_plan: str = "",
    ) -> List[Dict[str, Any]]:
        """Build the initial LLM message list.

        Contains:
        - System prompt + recalled memories
        - Conversation history (from Momex short-term memory)
        - Current user message

        Args:
            include_planning: If True, adds planning instructions to prompt.
            approved_plan: If set, injects the approved plan for execution.
            pending_plan: If set, injects a pending plan awaiting user response.
        """
        messages: List[Dict[str, Any]] = []

        # Dynamic system prompt: built from live agent registry
        agent_descriptions = ""
        if self._agent_registry:
            try:
                agent_descriptions = await self._agent_registry.get_agent_descriptions(
                    tenant_id=context.get("tenant_id"),
                    credential_store=self.credential_store,
                )
            except Exception as e:
                logger.warning(f"Failed to get agent descriptions: {e}")

        # Build system prompt with optional preamble override
        build_kwargs = dict(
            agent_descriptions=agent_descriptions,
            include_planning=include_planning,
            approved_plan=approved_plan,
            pending_plan=pending_plan,
        )
        if self.system_prompt and self.system_prompt_mode == "override":
            build_kwargs["preamble"] = self.system_prompt

        system_prompt = build_system_prompt(**build_kwargs)

        system_parts = [system_prompt]
        if self.system_prompt and self.system_prompt_mode != "override":
            system_parts.append(self.system_prompt)

        # Runtime context
        now = datetime.now(timezone.utc)

        # Add user location if available in metadata
        meta = context.get("metadata") or {}
        tz = meta.get("timezone")
        if tz and tz != "UTC":
            try:
                from zoneinfo import ZoneInfo

                user_tz = ZoneInfo(tz)
                user_now = now.astimezone(user_tz)
                context_lines = [f"Current time: {user_now.strftime('%Y-%m-%d %H:%M:%S')} ({tz})"]
            except Exception:
                context_lines = [f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"]
                context_lines.append(f"User timezone: {tz}")
        else:
            context_lines = [f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"]

        location = meta.get("location")
        if location and isinstance(location, dict):
            lat = location.get("lat")
            lng = location.get("lng")
            if lat is not None and lng is not None:
                place = location.get("place_name", "")
                try:
                    loc_str = f"User location: {float(lat):.4f}, {float(lng):.4f}"
                except (TypeError, ValueError):
                    loc_str = f"User location: {lat}, {lng}"
                if place:
                    loc_str += f" ({place})"
                context_lines.append(loc_str)

        system_parts.append("\n[Context]\n" + "\n".join(context_lines))

        # True Memory (canonical app-owned facts, passed by app layer)
        true_memory_text = format_true_memory_for_prompt(meta.get("true_memory"))
        if true_memory_text:
            system_parts.append("\n[True Memory]\n" + true_memory_text)

        # User profile (extracted from email, passed by app layer)
        profile_text = self._format_user_profile(meta.get("user_profile"))
        if profile_text:
            system_parts.append("\n[User Profile]\n" + profile_text)

        session_prompt = context.get(
            "session_memory_prompt"
        ) or self.session_memory.build_prompt_section(
            context.get("session_id", context.get("tenant_id", "")),
        )
        if session_prompt:
            system_parts.append("\n[Session Working Memory]\n" + session_prompt)

        # Relevant memories from Momex (auto-recall based on user message)
        if self.momex and needs_memory:
            try:
                recalled = await asyncio.wait_for(
                    self.momex.search(
                        tenant_id=context.get("tenant_id", ""),
                        query=user_message,
                        limit=10,
                    ),
                    timeout=5.0,
                )
                recalled = self.memory_governance.select_recalled_memories(
                    recalled,
                    true_memory=meta.get("true_memory"),
                )
                if recalled:
                    context["recalled_memories"] = recalled
                    memory_block = self.memory_governance.build_recalled_memory_block(recalled)
                    if memory_block:
                        system_parts.append("\n[Relevant Memories]\n" + memory_block)
            except asyncio.TimeoutError:
                logger.warning("MOMEX search timed out (5s), skipping memory recall")
            except Exception as e:
                logger.warning(f"Failed to auto-recall memories: {e}")

        # Episode prefetch via Momex (subkind="behavioral_pattern" or
        # "weekly_reflection" items tagged kind=episode). We dedupe against
        # the main "relevant memories" block since Momex is the shared index.
        episodes = context.get("recalled_episodes") or []
        if not episodes and needs_memory and self.momex is not None:
            try:
                from ..memory.lifecycle.episode_memory import EpisodeMemory

                tenant_id = context.get("tenant_id", "")
                if tenant_id and user_message:
                    episode_memory = EpisodeMemory(self.momex)
                    episodes = await asyncio.wait_for(
                        episode_memory.recall_episodes(
                            tenant_id,
                            user_message,
                            limit=5,
                        ),
                        timeout=3.0,
                    )
                    if episodes:
                        context["recalled_episodes"] = episodes
            except asyncio.TimeoutError:
                logger.debug("Episode prefetch timed out (3s), skipping")
            except Exception as e:
                logger.debug("Episode prefetch failed: %s", e)

        if episodes:
            lines: List[str] = []
            for ep in episodes[:5]:
                meta = ep.get("metadata") or {}
                date_s = meta.get("local_date") or meta.get("start_ts") or ep.get("timestamp") or ""
                title = meta.get("title") or meta.get("subkind") or "(episode)"
                summary = (ep.get("text") or ep.get("summary") or "").strip().replace("\n", " ")
                if len(summary) > 240:
                    summary = summary[:237] + "..."
                lines.append(f"- [{date_s}] {title}: {summary}")
            if lines:
                system_parts.append(
                    "\n[Recalled Episodes — past events relevant to the user's message]\n"
                    + "\n".join(lines)
                )

        messages.append(
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        )

        # Conversation history (from Momex short-term memory)
        history = context.get("conversation_history", [])
        if history:
            logger.info(
                f"[ReAct] history: {len(history)} messages, roles: {[m.get('role') for m in history[:6]]}..."
            )
            messages.extend(history)
        else:
            logger.info("[ReAct] history: 0 messages (clean session)")

        # Current user message
        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    @staticmethod
    def _format_user_profile(profile: Optional[Dict[str, Any]]) -> str:
        """Format user_profiles JSONB into concise text for system prompt.

        Returns empty string if profile is None or has no useful data.
        """
        if not profile:
            return ""

        lines: List[str] = []

        identity = profile.get("identity") or {}
        if identity.get("full_name"):
            lines.append(f"Name: {identity['full_name']}")
        if identity.get("birthday"):
            lines.append(f"Birthday: {identity['birthday']}")

        for addr in profile.get("addresses") or []:
            parts = [addr.get("street"), addr.get("city"), addr.get("state")]
            loc = ", ".join(p for p in parts if p)
            label = addr.get("label", "Address").title()
            if loc:
                line = f"{label}: {loc}"
            else:
                line = f"{label}:"
            lat, lng = addr.get("lat"), addr.get("lng")
            if lat is not None and lng is not None:
                line += f" (coordinates: {lat}, {lng})"
            if loc or (lat is not None and lng is not None):
                lines.append(line)

        work = profile.get("work") or {}
        for job in work.get("jobs") or []:
            if job.get("is_current"):
                parts = [job.get("title", ""), job.get("employer", "")]
                desc = " at ".join(p for p in parts if p)
                if desc:
                    lines.append(f"Work: {desc}")

        education = profile.get("education") or {}
        for school in education.get("schools") or []:
            parts = [school.get("degree"), school.get("major")]
            desc = " in ".join(p for p in parts if p)
            name = school.get("name", "")
            if name:
                line = f"Education: {name}"
                if desc:
                    line += f" ({desc})"
                lines.append(line)

        relationships = profile.get("relationships") or {}
        for person in relationships.get("family") or []:
            line = f"{person.get('relationship', 'Family')}: {person.get('name', '')}"
            if person.get("birthday"):
                line += f" (birthday: {person['birthday']})"
            lines.append(line)
        so = relationships.get("significant_other")
        if so and so.get("name"):
            line = f"Partner: {so['name']}"
            if so.get("birthday"):
                line += f" (birthday: {so['birthday']})"
            lines.append(line)

        lifestyle = profile.get("lifestyle") or {}
        for pet in lifestyle.get("pets") or []:
            if pet.get("name"):
                lines.append(f"Pet: {pet['name']} ({pet.get('type', '')})")
        for vehicle in lifestyle.get("vehicles") or []:
            if vehicle.get("is_current") and vehicle.get("make"):
                parts = [
                    str(vehicle.get("year", "")),
                    vehicle.get("make", ""),
                    vehicle.get("model", ""),
                ]
                lines.append(f"Vehicle: {' '.join(p for p in parts if p)}")

        travel = profile.get("travel") or {}
        for prog in travel.get("loyalty_programs") or []:
            name = prog.get("program", "")
            if name:
                line = f"Loyalty: {name}"
                if prog.get("status"):
                    line += f" ({prog['status']})"
                lines.append(line)

        return "\n".join(lines)
