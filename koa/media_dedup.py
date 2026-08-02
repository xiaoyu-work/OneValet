"""Collecting media without collecting it twice.

The same image or card can arrive from more than one place in a run --
an agent forwards its inner tool's output, and the tool that produced it
also reports it -- so what reaches the user has to be de-duplicated by
content rather than by identity.

Both the orchestrator's ReAct loop and StandardAgent had a verbatim copy
of this. They are one copy now.
"""

import json
from typing import Any, Dict, List


def stable_json(value: Any) -> str:
    """A form of a value that compares equal whenever the value does."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def media_key(item: Dict[str, Any]) -> str:
    """Identity of a media item by what it holds, not by object identity.

    Card payloads are parsed when they arrive as strings: the same cards
    serialised twice can differ in key order, and would otherwise be sent
    to the user as two different attachments.
    """
    if item.get("type") == "inline_cards":
        data = item.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
        return f"inline_cards:{stable_json(data)}"
    return stable_json(
        {
            "type": item.get("type"),
            "data": item.get("data"),
            "media_type": item.get("media_type"),
            "metadata": item.get("metadata"),
        }
    )


def append_unique_media(target: List[Dict[str, Any]], media: List[Dict[str, Any]]) -> None:
    """Add media not already present, preserving arrival order."""
    seen = {media_key(item) for item in target}
    for item in media:
        key = media_key(item)
        if key in seen:
            continue
        seen.add(key)
        target.append(item)
