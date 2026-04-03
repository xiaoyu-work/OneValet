"""Runtime execution policy checks for tools.

Provides a lightweight policy layer that complements schema-time filtering.
The orchestrator and StandardAgent can both use the same evaluator so tool
execution decisions are made with request context, not just registration-time
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from ..models import AgentTool


def _as_set(values: Optional[Iterable[Any]]) -> set[str]:
    if not values:
        return set()
    return {str(v) for v in values if v is not None}


@dataclass
class ExecutionPolicyDecision:
    """Result of evaluating a tool execution attempt."""

    allowed: bool = True
    reason: str = ""
    require_approval: bool = False
    tags: list[str] = field(default_factory=list)


class ExecutionPolicyEngine:
    """Evaluate runtime tool policies from request metadata and tool metadata."""

    def evaluate(
        self,
        tool: AgentTool,
        *,
        tenant_id: str = "",
        args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
    ) -> ExecutionPolicyDecision:
        del tenant_id, args, agent_type  # reserved for future richer policies

        security = self._merged_security_context(metadata, request_context)
        tags: list[str] = []

        denied_tools = _as_set(security.get("denied_tools"))
        if tool.name in denied_tools:
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool '{tool.name}' is denied by runtime policy",
                tags=["tool-deny"],
            )

        allowed_tools = _as_set(security.get("allowed_tools"))
        if allowed_tools and tool.name not in allowed_tools:
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool '{tool.name}' is not in the runtime allow list",
                tags=["tool-allow"],
            )

        denied_categories = _as_set(security.get("denied_categories"))
        if tool.category in denied_categories:
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool category '{tool.category}' is denied by runtime policy",
                tags=["category-deny"],
            )

        allowed_categories = _as_set(security.get("allowed_categories"))
        if allowed_categories and tool.category not in allowed_categories:
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool category '{tool.category}' is not in the runtime allow list",
                tags=["category-allow"],
            )

        if tool.requires_feature_flag:
            feature_flags = _as_set(security.get("feature_flags"))
            if tool.requires_feature_flag not in feature_flags:
                return ExecutionPolicyDecision(
                    allowed=False,
                    reason=(
                        f"tool '{tool.name}' requires feature flag "
                        f"'{tool.requires_feature_flag}'"
                    ),
                    tags=["feature-flag"],
                )

        if tool.enabled_tiers:
            user_tier = str(security.get("user_tier", "") or "")
            if user_tier not in set(tool.enabled_tiers):
                return ExecutionPolicyDecision(
                    allowed=False,
                    reason=f"tool '{tool.name}' is unavailable for tier '{user_tier or 'unknown'}'",
                    tags=["tier-gate"],
                )

        if security.get("read_only_mode") and not tool.read_only:
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool '{tool.name}' is blocked while read_only_mode is enabled",
                tags=["read-only-mode"],
            )

        if security.get("allow_write_actions") is False and (
            tool.mutates_user_data or tool.risk_level in ("write", "destructive")
        ):
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool '{tool.name}' is blocked because write actions are disabled",
                tags=["write-disabled"],
            )

        if security.get("allow_destructive_actions") is False and tool.risk_level == "destructive":
            return ExecutionPolicyDecision(
                allowed=False,
                reason=f"tool '{tool.name}' is blocked because destructive actions are disabled",
                tags=["destructive-disabled"],
            )

        allowed_risk_levels = _as_set(security.get("allowed_risk_levels"))
        if allowed_risk_levels and tool.risk_level not in allowed_risk_levels:
            return ExecutionPolicyDecision(
                allowed=False,
                reason=(
                    f"tool '{tool.name}' has risk level '{tool.risk_level}', "
                    "which is not allowed for this request"
                ),
                tags=["risk-level"],
            )

        approved_tools = _as_set(security.get("approved_tools"))
        approved_categories = _as_set(security.get("approved_categories"))
        require_approval = bool(
            tool.needs_approval
            and tool.name not in approved_tools
            and tool.category not in approved_categories
        )
        if require_approval:
            tags.append("approval-required")

        return ExecutionPolicyDecision(
            allowed=True,
            reason="allowed",
            require_approval=require_approval,
            tags=tags,
        )

    @staticmethod
    def _merged_security_context(
        metadata: Optional[Dict[str, Any]],
        request_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge security context from request metadata and tool metadata."""
        merged: Dict[str, Any] = {}

        meta = metadata or {}
        req = request_context or {}

        sources = [
            req.get("permissions"),
            (req.get("metadata") or {}).get("permissions"),
            meta.get("permissions"),
        ]
        for source in sources:
            if isinstance(source, dict):
                merged.update(source)

        return merged
