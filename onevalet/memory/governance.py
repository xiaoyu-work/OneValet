"""Governance helpers for orchestrator-owned memory recall and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


_TRANSIENT_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|cool|great|sure|yes|no|bye)[!. ]*$",
    re.IGNORECASE,
)
_PERSISTENT_HINTS = (
    "remember",
    "prefer",
    "usually",
    "always",
    "never",
    "my birthday",
    "my address",
    "my email",
    "my phone",
    "i live",
    "i work",
    "timezone",
)
_FEEDBACK_HINTS = (
    "don't do that",
    "do not do that",
    "next time",
    "please stop",
    "please always",
    "i like it when",
    "i don't like",
)


def _compact(text: str, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


@dataclass
class MemoryWriteDecision:
    """Decision about whether a turn should be written to long-term memory."""

    should_store: bool
    reason: str
    tags: List[str] = field(default_factory=list)


class MemoryGovernance:
    """Apply explicit recall and storage rules around Momex usage."""

    def __init__(
        self,
        *,
        max_prompt_memories: int = 6,
        max_prompt_chars: int = 1400,
    ) -> None:
        self.max_prompt_memories = max_prompt_memories
        self.max_prompt_chars = max_prompt_chars

    def select_recalled_memories(self, recalled: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Deduplicate and cap recalled memories before prompt injection."""
        if not recalled:
            return []

        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        total_chars = 0
        sorted_items = sorted(
            recalled,
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        for item in sorted_items:
            text = _compact(str(item.get("text", "") or ""), 220)
            if not text or text in seen:
                continue
            projected = total_chars + len(text)
            if selected and projected > self.max_prompt_chars:
                break
            selected.append({**item, "text": text})
            seen.add(text)
            total_chars = projected
            if len(selected) >= self.max_prompt_memories:
                break
        return selected

    def build_recalled_memory_block(self, recalled: Optional[List[Dict[str, Any]]]) -> str:
        """Render recalled memories into a compact prompt section."""
        selected = self.select_recalled_memories(recalled)
        if not selected:
            return ""

        lines = []
        for item in selected:
            mem_type = str(item.get("type", "") or "").strip()
            text = str(item.get("text", "") or "").strip()
            if mem_type:
                lines.append(f"- [{mem_type}] {text}")
            else:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def decide_storage(
        self,
        *,
        user_message: str,
        assistant_message: str,
        result_status: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryWriteDecision:
        """Decide whether to write a turn to durable memory."""
        meta = metadata or {}
        user_text = " ".join((user_message or "").split())
        assistant_text = " ".join((assistant_message or "").split())
        status = str(result_status or "").lower()

        if meta.get("skip_memory_write"):
            return MemoryWriteDecision(False, "skip requested by metadata", ["explicit-skip"])
        if meta.get("force_memory_write"):
            return MemoryWriteDecision(True, "forced by metadata", ["explicit-force"])
        if status in {"waiting_for_input", "waiting_for_approval", "error"}:
            return MemoryWriteDecision(False, f"skip {status} turn", ["non-terminal"])
        if not user_text or not assistant_text:
            return MemoryWriteDecision(False, "missing user or assistant text", ["empty"])
        if len(user_text) <= 48 and _TRANSIENT_RE.match(user_text):
            return MemoryWriteDecision(False, "transient conversational turn", ["transient"])
        if assistant_text.lower().startswith("error:") or "something went wrong" in assistant_text.lower():
            return MemoryWriteDecision(False, "assistant error response", ["error"])

        lowered = user_text.lower()
        tags: List[str] = []
        score = 0

        if any(hint in lowered for hint in _PERSISTENT_HINTS):
            score += 2
            tags.append("persistent-signal")
        if any(hint in lowered for hint in _FEEDBACK_HINTS):
            score += 2
            tags.append("feedback-signal")
        if len(user_text) >= 32:
            score += 1
        if len(assistant_text) >= 64:
            score += 1

        should_store = score >= 2
        reason = "store durable turn" if should_store else "insufficient long-term memory signal"
        return MemoryWriteDecision(should_store, reason, tags)
