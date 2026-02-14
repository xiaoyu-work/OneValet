"""
Structured audit logging for orchestrator decisions.

Produces JSON log entries via Python's standard logging module under
the ``onevalet.audit`` logger name.  Each entry includes a timestamp,
event_type, optional tenant_id, and event-specific fields.

Usage::

    audit = AuditLogger()
    audit.log_policy_decision(
        intent="book_flight",
        must_use_tools=True,
        selected_tools=["flight_search"],
        reason_code="domain_action",
    )
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_audit_logger = logging.getLogger("onevalet.audit")


class AuditLogger:
    """Structured audit logger for key orchestrator decisions."""

    def __init__(self, tenant_id: Optional[str] = None) -> None:
        self._default_tenant_id = tenant_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, fields: Dict[str, Any]) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }
        entry.update(fields)
        _audit_logger.info(json.dumps(entry, default=str))

    def _tid(self, tenant_id: Optional[str] = None) -> str:
        return tenant_id or self._default_tenant_id or ""

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def log_policy_decision(
        self,
        intent: str,
        must_use_tools: bool,
        selected_tools: List[str],
        reason_code: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a tool-policy routing decision."""
        self._emit("policy_decision", {
            "tenant_id": self._tid(tenant_id),
            "intent": intent,
            "must_use_tools": must_use_tools,
            "selected_tools": selected_tools,
            "selected_tools_count": len(selected_tools),
            "reason_code": reason_code,
        })

    def log_route_decision(
        self,
        tenant_id: str,
        target_agent_id: Optional[str],
        waiting_agents_count: int,
        reason: str,
    ) -> None:
        """Log a WAITING-agent routing decision."""
        self._emit("route_decision", {
            "tenant_id": tenant_id,
            "target_agent_id": target_agent_id,
            "waiting_agents_count": waiting_agents_count,
            "reason": reason,
        })

    def log_tool_execution(
        self,
        tool_name: str,
        args_summary: Dict[str, Any],
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a tool execution result."""
        fields: Dict[str, Any] = {
            "tenant_id": self._tid(tenant_id),
            "tool_name": tool_name,
            "args_summary": args_summary,
            "success": success,
            "duration_ms": duration_ms,
        }
        if error is not None:
            fields["error"] = error
        self._emit("tool_execution", fields)

    def log_approval_decision(
        self,
        agent_name: str,
        tool_name: str,
        risk_level: str,
        decision: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log an approval decision."""
        self._emit("approval_decision", {
            "tenant_id": self._tid(tenant_id),
            "agent_name": agent_name,
            "tool_name": tool_name,
            "risk_level": risk_level,
            "decision": decision,
        })

    def log_react_turn(
        self,
        turn: int,
        tool_calls: List[str],
        final_answer: bool,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a ReAct turn summary."""
        self._emit("react_turn", {
            "tenant_id": self._tid(tenant_id),
            "turn": turn,
            "tool_calls": tool_calls,
            "tool_calls_count": len(tool_calls),
            "final_answer": final_answer,
        })
