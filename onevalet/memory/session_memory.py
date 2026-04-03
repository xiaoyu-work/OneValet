"""Lightweight orchestrator-owned session working memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _dedupe_push(items: List[str], value: str, limit: int) -> None:
    value = value.strip()
    if not value:
        return
    if value in items:
        items.remove(value)
    items.append(value)
    if len(items) > limit:
        del items[:-limit]


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


@dataclass
class SessionWorkingMemory:
    """Compact, human-readable notes for an active session."""

    objective: str = ""
    constraints: List[str] = field(default_factory=list)
    recent_findings: List[str] = field(default_factory=list)
    pending_questions: List[str] = field(default_factory=list)
    pending_approvals: List[str] = field(default_factory=list)
    recent_tools: List[str] = field(default_factory=list)
    last_user_message: str = ""
    updated_at: str = ""


class SessionMemoryManager:
    """Stores lightweight working memory per session."""

    def __init__(
        self,
        *,
        max_constraints: int = 5,
        max_findings: int = 6,
        max_pending_items: int = 3,
        max_recent_tools: int = 5,
    ) -> None:
        self._sessions: Dict[str, SessionWorkingMemory] = {}
        self.max_constraints = max_constraints
        self.max_findings = max_findings
        self.max_pending_items = max_pending_items
        self.max_recent_tools = max_recent_tools

    def prepare_session(
        self,
        session_id: str,
        user_message: str,
        *,
        has_active_agents: bool = False,
    ) -> Dict[str, Any]:
        """Prime working memory before a new orchestrator turn."""
        state = self._sessions.setdefault(session_id, SessionWorkingMemory())
        user_summary = _truncate(user_message, 180)
        state.last_user_message = user_summary
        if user_summary and (not has_active_agents or not state.objective):
            state.objective = user_summary
        if not has_active_agents:
            state.pending_questions.clear()
            state.pending_approvals.clear()
        for constraint in self._extract_constraints(user_message):
            _dedupe_push(state.constraints, constraint, self.max_constraints)
        state.updated_at = self._now()
        return asdict(state)

    def update_from_result(
        self,
        session_id: str,
        *,
        user_message: str = "",
        assistant_message: str = "",
        result_status: str = "",
        tool_calls: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update working memory after a response is produced."""
        state = self._sessions.setdefault(session_id, SessionWorkingMemory())
        if user_message:
            state.last_user_message = _truncate(user_message, 180)
            if not state.objective:
                state.objective = state.last_user_message
            for constraint in self._extract_constraints(user_message):
                _dedupe_push(state.constraints, constraint, self.max_constraints)

        if tool_calls:
            tool_names = [self._tool_name(tc) for tc in tool_calls]
            state.recent_tools = [name for name in tool_names if name][-self.max_recent_tools :]

        if metadata:
            for finding in metadata.get("session_findings", []) or []:
                _dedupe_push(state.recent_findings, _truncate(str(finding), 240), self.max_findings)

        assistant_summary = _truncate(assistant_message, 240)
        status = str(result_status or "").lower()
        if assistant_summary:
            if status == "waiting_for_input":
                state.pending_questions.clear()
                _dedupe_push(
                    state.pending_questions,
                    assistant_summary,
                    self.max_pending_items,
                )
            elif status == "waiting_for_approval":
                state.pending_approvals.clear()
                _dedupe_push(
                    state.pending_approvals,
                    assistant_summary,
                    self.max_pending_items,
                )
            elif status == "completed":
                state.pending_questions.clear()
                state.pending_approvals.clear()
                _dedupe_push(
                    state.recent_findings,
                    assistant_summary,
                    self.max_findings,
                )
            elif status == "error":
                _dedupe_push(
                    state.recent_findings,
                    f"Error: {assistant_summary}",
                    self.max_findings,
                )

        state.updated_at = self._now()
        return asdict(state)

    def build_prompt_section(self, session_id: str) -> str:
        """Render working memory into a compact system-prompt section."""
        state = self._sessions.get(session_id)
        if not state:
            return ""

        lines: List[str] = []
        if state.objective:
            lines.append(f"Objective: {state.objective}")
        if state.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {item}" for item in state.constraints)
        if state.recent_findings:
            lines.append("Recent findings:")
            lines.extend(f"- {item}" for item in state.recent_findings)
        if state.pending_questions:
            lines.append("Pending user input:")
            lines.extend(f"- {item}" for item in state.pending_questions)
        if state.pending_approvals:
            lines.append("Pending approvals:")
            lines.extend(f"- {item}" for item in state.pending_approvals)
        if state.recent_tools:
            lines.append("Recent tools: " + ", ".join(state.recent_tools))
        return "\n".join(lines)

    def build_handoff_context(self, session_id: str) -> Dict[str, Any]:
        """Return session notes for agent handoff payloads."""
        state = self._sessions.get(session_id)
        if not state:
            return {}
        return asdict(state)

    @staticmethod
    def _tool_name(tool_call: Any) -> str:
        if hasattr(tool_call, "name"):
            return str(getattr(tool_call, "name") or "")
        if isinstance(tool_call, dict):
            return str(tool_call.get("name") or "")
        return ""

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _extract_constraints(user_message: str) -> List[str]:
        text = " ".join((user_message or "").split())
        lowered = text.lower()
        keywords = ("don't", "do not", "without", "only", "before", "after", "avoid")
        if not text or not any(k in lowered for k in keywords):
            return []
        return [_truncate(text, 180)]
