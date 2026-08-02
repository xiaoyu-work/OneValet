"""Whether a human is present to answer the agent mid-run.

Interactive chat has someone waiting on the other end, so an agent can pause
and ask for approval. Cron jobs, proactive triggers, and pipeline steps do
not -- nothing is watching, and a run that pauses there simply stops, with
the user never learning it was attempted.

Callers declare where a request came from in ``metadata["source"]``. This
maps that to the one question the loop actually needs answered.
"""

from typing import Any, Dict, Optional

#: Sources that run with nobody watching. Anything not listed is treated as
#: attended, so a new interactive surface does not silently lose approvals.
UNATTENDED_SOURCES = frozenset({"cron", "trigger", "pipeline", "webhook", "system"})


def is_attended(metadata: Optional[Dict[str, Any]]) -> bool:
    """True when a human can answer a question raised mid-run.

    Defaults to True for unknown or missing sources: wrongly assuming someone
    is present costs a stalled run that a user can still retry, while wrongly
    assuming nobody is present would let an agent take an unapproved action.
    """
    source = (metadata or {}).get("source")
    if not source:
        return True
    return str(source).lower() not in UNATTENDED_SOURCES
