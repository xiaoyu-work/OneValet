"""Reminder Guard — post-process hook that detects unfulfilled reminder commitments.

When the AI response says "I'll set a reminder" or "I'll remember to follow up"
but no cron tool was actually called during the turn, this hook appends a
self-correction note so the user doesn't receive a false promise.

Usage:
    orchestrator = Orchestrator(
        ...,
        post_process_hooks=[reminder_guard_hook],
    )
"""

import logging
import re
from typing import Any, Dict

from ..result import AgentResult

logger = logging.getLogger(__name__)

UNSCHEDULED_REMINDER_NOTE = (
    "Note: I did not schedule a reminder in this turn, so this will not trigger automatically."
)

# Patterns that indicate the AI committed to scheduling a reminder/alert
# but may not have actually called the cron tool.
REMINDER_COMMITMENT_PATTERNS = [
    re.compile(
        r"\b(?:i\s*[''\u2019]?ll|i will)\s+(?:make sure to\s+)?"
        r"(?:remember|remind|ping|follow up|follow-up|check back|circle back)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s*[''\u2019]?ll|i will)\s+(?:set|create|schedule)\s+(?:a\s+)?reminder\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breminder\s+(?:has been|is)\s+(?:set|created|scheduled)\b",
        re.IGNORECASE,
    ),
]

# Tool names that directly create/modify reminders or cron jobs.
# `set_reminder` is the tool exposed by TodoAgent; `cron_add/update/run`
# are lower-level tools exposed by CronAgent or other direct callers.
CRON_TOOL_NAMES = {"set_reminder", "cron_add", "cron_update", "cron_run"}

# Sub-agents that encapsulate reminder/cron tools inside their own
# ReAct loop. When the orchestrator delegates to one of these, the
# nested `set_reminder` / `cron_add` call is NOT surfaced into the
# top-level `context["tool_calls"]` list — only the agent name is.
# We treat a successful (or pending-approval) delegation to one of
# these as evidence that a reminder was scheduled (or is about to be),
# to avoid the guard falsely appending the "not scheduled" note when
# the work is actually delegated and in-flight.
REMINDER_CAPABLE_AGENTS = {"TodoAgent", "CronAgent"}

# result_status values that indicate the delegated agent either
# completed its work or is blocking on user approval — both of
# which mean we should NOT claim "no reminder was scheduled".
# Values are normalized to uppercase before comparison because
# ToolCallRecord.result_status is written in uppercase by react_loop.
_REMINDER_SUCCESS_STATUSES = {"COMPLETED", "WAITING_FOR_APPROVAL"}


def _has_unbacked_reminder_commitment(text: str) -> bool:
    """Check if the response text contains a reminder commitment."""
    if not text.strip():
        return False
    # Don't double-append
    if UNSCHEDULED_REMINDER_NOTE.lower() in text.lower():
        return False
    return any(p.search(text) for p in REMINDER_COMMITMENT_PATTERNS)


def _tc_field(tc: Any, field: str) -> Any:
    """Read a field from a ToolCallRecord or dict-shaped tool call entry."""
    if isinstance(tc, dict):
        return tc.get(field)
    return getattr(tc, field, None)


def _reminder_was_scheduled(context: Dict[str, Any]) -> bool:
    """Return True if this turn actually scheduled (or is scheduling) a reminder.

    Two kinds of evidence are accepted:

    1. A direct call to one of the known cron/reminder tools
       (``CRON_TOOL_NAMES``) in the top-level ``context['tool_calls']``.

    2. A delegation to a reminder-capable sub-agent
       (``REMINDER_CAPABLE_AGENTS``) that either completed successfully or
       is waiting for user approval. This covers the common case where
       ``TodoAgent`` runs ``set_reminder`` inside its own ReAct loop — the
       nested tool name never reaches the orchestrator's tool_calls list.
    """
    tool_calls = context.get("tool_calls", [])
    if not tool_calls:
        return False

    for tc in tool_calls:
        name = _tc_field(tc, "name") or ""
        if name in CRON_TOOL_NAMES:
            return True
        if name in REMINDER_CAPABLE_AGENTS:
            status = _tc_field(tc, "result_status")
            status_norm = str(status).upper() if status is not None else ""
            success = _tc_field(tc, "success")
            # Default `success` to True when the field is absent (dict-shaped
            # entries may omit it). A record with success=False is explicitly
            # failing and should NOT suppress the guard.
            if success is False:
                continue
            if status_norm in _REMINDER_SUCCESS_STATUSES:
                return True
    return False


async def reminder_guard_hook(
    result: AgentResult,
    context: Dict[str, Any],
) -> AgentResult:
    """Post-process hook: detect unfulfilled reminder commitments.

    If the AI's response claims it set/created a reminder but no cron tool
    was called, append a self-correction note to the response.
    """
    if not result.raw_message:
        return result

    if not _has_unbacked_reminder_commitment(result.raw_message):
        return result

    if _reminder_was_scheduled(context):
        return result

    # AI promised a reminder but didn't actually create one
    logger.warning("Reminder guard triggered: response claims reminder but no cron tool was called")
    result.raw_message = f"{result.raw_message.rstrip()}\n\n{UNSCHEDULED_REMINDER_NOTE}"
    return result
