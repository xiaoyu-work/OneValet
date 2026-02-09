"""Event trigger matching — source + type + filter matching."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def matches_event(
    trigger_params: Dict[str, Any],
    event_source: str,
    event_type: str,
    event_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if an incoming event matches a trigger's filter criteria.

    Args:
        trigger_params: Trigger params with "source", "event_type", and optional "filters"
        event_source: Incoming event source (e.g. "email", "calendar")
        event_type: Incoming event type (e.g. "new_email", "event_created")
        event_data: Incoming event payload

    Returns:
        True if the event matches
    """
    # Source match
    expected_source = trigger_params.get("source", "")
    if expected_source and expected_source != event_source:
        return False

    # Type match
    expected_type = trigger_params.get("event_type", "")
    if expected_type and expected_type != event_type:
        return False

    # Filter match
    filters = trigger_params.get("filters", {})
    if filters and event_data:
        for key, expected_value in filters.items():
            actual_value = event_data.get(key)
            if actual_value is None:
                return False
            # String contains match
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                if expected_value.lower() not in actual_value.lower():
                    return False
            elif actual_value != expected_value:
                return False

    return True
