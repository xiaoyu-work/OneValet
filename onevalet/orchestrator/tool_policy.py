"""
Tool policy filter layer for the orchestrator.

Provides global and per-agent tool filtering with two layers:

1. **Global** -- deny-list and optional allow-list applied to every request.
2. **Agent-level** -- per-agent-type overrides (allow/deny sets).

Filter order: global deny -> global allow -> agent deny -> agent allow.

Usage::

    policy = ToolPolicyFilter()
    policy.set_global_deny({"dangerous_tool"})
    policy.set_agent_policy("email_agent", deny={"send_sms"})

    filtered = policy.filter_tools(all_schemas, agent_type="email_agent")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class AgentToolPolicy:
    """Per-agent tool policy override."""

    agent_type: str
    allow: Optional[Set[str]] = None  # whitelist (if set, only these)
    deny: Set[str] = field(default_factory=set)  # blacklist


class ToolPolicyFilter:
    """Two-layer tool policy filter (global + agent-level)."""

    def __init__(self) -> None:
        self._global_deny: Set[str] = set()
        self._global_allow: Optional[Set[str]] = None  # if set, only these tools allowed
        self._agent_policies: Dict[str, AgentToolPolicy] = {}

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def set_global_deny(self, tool_names: Set[str]) -> None:
        """Set the global deny-list (tools blocked for everyone)."""
        self._global_deny = set(tool_names)

    def set_global_allow(self, tool_names: Set[str]) -> None:
        """Set the global allow-list (if set, only these tools are permitted)."""
        self._global_allow = set(tool_names)

    def set_agent_policy(
        self,
        agent_type: str,
        allow: Optional[Set[str]] = None,
        deny: Optional[Set[str]] = None,
    ) -> None:
        """Set or update per-agent tool policy."""
        self._agent_policies[agent_type] = AgentToolPolicy(
            agent_type=agent_type,
            allow=set(allow) if allow is not None else None,
            deny=set(deny) if deny is not None else set(),
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_name(schema: Dict) -> Optional[str]:
        """Extract function name from an OpenAI-style tool schema."""
        func = schema.get("function")
        if isinstance(func, dict):
            name = func.get("name")
            return name if isinstance(name, str) else None
        return None

    def is_tool_allowed(self, tool_name: str, agent_type: Optional[str] = None) -> bool:
        """Check whether a single tool is allowed under current policies."""
        # Global deny
        if tool_name in self._global_deny:
            return False

        # Global allow
        if self._global_allow is not None and tool_name not in self._global_allow:
            return False

        # Agent-level
        if agent_type and agent_type in self._agent_policies:
            ap = self._agent_policies[agent_type]
            if tool_name in ap.deny:
                return False
            if ap.allow is not None and tool_name not in ap.allow:
                return False

        return True

    def filter_tools(
        self,
        tool_schemas: List[Dict],
        agent_type: Optional[str] = None,
    ) -> List[Dict]:
        """Filter tool schemas through global + agent policies.

        Args:
            tool_schemas: List of OpenAI-format tool schema dicts.
            agent_type: Optional agent type for agent-level filtering.

        Returns:
            Filtered list of tool schemas (order preserved).
        """
        result: List[Dict] = []
        for schema in tool_schemas:
            name = self._tool_name(schema)
            if name is None:
                result.append(schema)  # keep schemas we can't parse
                continue
            if self.is_tool_allowed(name, agent_type):
                result.append(schema)
        return result

    def get_filter_reason(
        self,
        tool_name: str,
        agent_type: Optional[str] = None,
    ) -> Optional[str]:
        """Return a human-readable reason why a tool was filtered, or None if allowed."""
        if tool_name in self._global_deny:
            return f"tool '{tool_name}' is in the global deny list"

        if self._global_allow is not None and tool_name not in self._global_allow:
            return f"tool '{tool_name}' is not in the global allow list"

        if agent_type and agent_type in self._agent_policies:
            ap = self._agent_policies[agent_type]
            if tool_name in ap.deny:
                return f"tool '{tool_name}' is denied for agent '{agent_type}'"
            if ap.allow is not None and tool_name not in ap.allow:
                return f"tool '{tool_name}' is not in the allow list for agent '{agent_type}'"

        return None
