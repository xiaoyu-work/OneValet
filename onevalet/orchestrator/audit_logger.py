"""
Structured audit logging for orchestrator decisions.

Produces JSON log entries via Python's standard logging module under
the ``onevalet.audit`` logger name.  Each entry includes a timestamp,
event_type, optional tenant_id, and event-specific fields.

When a request_id is set via ``start_request()``, all subsequent log
entries automatically include it — enabling end-to-end tracing of a
single user request across intent analysis, routing, tool execution,
memory extraction, and response delivery.

Usage::

    audit = AuditLogger()
    audit.start_request(tenant_id="user-123", message="check my email")
    audit.log_phase("intent_analysis", {"domains": ["communication"]})
    audit.log_tool_execution(...)
    audit.end_request(status="completed", token_usage={...})
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_audit_logger = logging.getLogger("onevalet.audit")


class AuditLogger:
    """Structured audit logger with per-request tracing."""

    def __init__(self, tenant_id: Optional[str] = None) -> None:
        self._default_tenant_id = tenant_id
        self._request_id: Optional[str] = None
        self._request_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Request tracing
    # ------------------------------------------------------------------

    def start_request(
        self,
        tenant_id: str,
        message: str,
        request_id: Optional[str] = None,
    ) -> str:
        """Begin tracing a new request. Returns the request_id."""
        self._request_id = request_id or uuid.uuid4().hex[:12]
        self._request_start = time.monotonic()
        self._default_tenant_id = tenant_id
        self._emit("request_start", {
            "tenant_id": tenant_id,
            "message_preview": message[:120] if message else "",
        })
        return self._request_id

    def end_request(
        self,
        status: str = "completed",
        token_usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End tracing the current request."""
        fields: Dict[str, Any] = {"status": status}
        if self._request_start is not None:
            fields["total_ms"] = int((time.monotonic() - self._request_start) * 1000)
        if token_usage:
            fields["token_usage"] = token_usage
        if error:
            fields["error"] = error
        self._emit("request_end", fields)
        self._request_id = None
        self._request_start = None

    def log_phase(
        self,
        phase: str,
        details: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Log a named phase within the current request.

        Useful for: intent_analysis, domain_filtering, tool_loading,
        memory_recall, true_memory_extraction, post_process, etc.
        """
        fields: Dict[str, Any] = {
            "tenant_id": self._tid(tenant_id),
            "phase": phase,
        }
        if self._request_start is not None:
            fields["elapsed_ms"] = int((time.monotonic() - self._request_start) * 1000)
        if details:
            fields.update(details)
        self._emit("phase", fields)

    @property
    def request_id(self) -> Optional[str]:
        return self._request_id

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, fields: Dict[str, Any]) -> None:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }
        if self._request_id:
            entry["request_id"] = self._request_id
        entry.update(fields)
        _audit_logger.info(json.dumps(entry, default=str))

    def _tid(self, tenant_id: Optional[str] = None) -> str:
        return tenant_id or self._default_tenant_id or ""

    # ------------------------------------------------------------------
    # public API (existing)
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
