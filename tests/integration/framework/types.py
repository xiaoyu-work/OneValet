"""Protocol definitions for the multi-turn conversation testing framework.

These protocols decouple the framework from concrete OneValet types,
making it possible to extract this module as a standalone package later.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class MessageHandler(Protocol):
    """Anything that can handle a user message and return a result.

    The returned object must have at least:
      - ``status``: comparable with ``==`` (str or enum)
      - ``raw_message``: str
    """

    async def handle_message(
        self,
        user_id: str,
        message: str,
        images: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any: ...


@runtime_checkable
class ToolRecorder(Protocol):
    """Anything that records tool calls as a list of dicts.

    Each dict must have at least:
      - ``tool_name``: str
      - ``arguments``: dict
    """

    @property
    def tool_calls(self) -> List[Dict[str, Any]]: ...
