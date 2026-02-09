"""
OneValet Momex Integration - Direct wrapper around typeagent-py memory system.

Provides both long-term structured memory and short-term conversation history
via the Momex (typeagent-py) library.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MomexMemory:
    """
    Thin wrapper around typeagent-py's Memory + ShortTermMemory API.

    Provides:
    - get_history(): Load conversation history (short-term)
    - save_history(): Save conversation history
    - search(): Search long-term memories
    - add(): Add messages for knowledge extraction

    Args:
        collection: Collection identifier for multi-tenant isolation
            (e.g. "user:xiaoyuzhang", "team:eng:alice")
        config: MomexConfig from typeagent-py (optional, uses env defaults)

    Example:
        momex = MomexMemory(
            collection="user:xiaoyuzhang",
        )

        # Load conversation history
        history = await momex.get_history(
            tenant_id="user123",
            session_id="session456",
            limit=50,
        )

        # Search long-term memories
        results = await momex.search(
            tenant_id="user123",
            query="user's email preferences",
            limit=5,
        )

        # Save history + extract knowledge
        await momex.save_history(tenant_id, session_id, messages)
        await momex.add(tenant_id, messages, infer=True)
    """

    def __init__(
        self,
        collection: str = "default",
        config: Optional[Any] = None,
    ):
        self.collection = collection
        self._config = config
        self._memory = None  # Lazy-initialized typeagent Memory
        self._short_term = None  # Lazy-initialized ShortTermMemory

    def _ensure_initialized(self) -> None:
        """Lazy-initialize momex clients."""
        if self._memory is not None:
            return

        try:
            from momex import Memory, ShortTermMemory
        except ImportError:
            raise ImportError(
                "momex is required for memory. "
                "Install with: pip install momex"
            )

        if self._config:
            self._memory = Memory(
                collection=self.collection,
                config=self._config,
            )
            self._short_term = ShortTermMemory(
                collection=self.collection,
                config=self._config,
            )
        else:
            self._memory = Memory(collection=self.collection)
            self._short_term = ShortTermMemory(collection=self.collection)

    async def get_history(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Load conversation history from short-term memory.

        Args:
            tenant_id: User/tenant identifier
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        self._ensure_initialized()
        try:
            messages = await self._short_term.get_session_messages(
                user_id=tenant_id,
                session_id=session_id,
                limit=limit,
            )
            return messages if messages else []
        except Exception as e:
            logger.warning(f"Failed to load history for {tenant_id}/{session_id}: {e}")
            return []

    async def save_history(
        self,
        tenant_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Save conversation history to short-term memory.

        Args:
            tenant_id: User/tenant identifier
            session_id: Session identifier
            messages: List of message dicts to save
        """
        self._ensure_initialized()
        try:
            await self._short_term.add_messages(
                user_id=tenant_id,
                session_id=session_id,
                messages=messages,
            )
        except Exception as e:
            logger.warning(f"Failed to save history for {tenant_id}/{session_id}: {e}")

    async def search(
        self,
        tenant_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search long-term memories.

        Args:
            tenant_id: User/tenant identifier
            query: Search query
            limit: Maximum results

        Returns:
            List of memory result dicts
        """
        self._ensure_initialized()
        try:
            results = await self._memory.search(
                query=query,
                user_id=tenant_id,
                limit=limit,
            )
            return results if results else []
        except Exception as e:
            logger.warning(f"Failed to search memories for {tenant_id}: {e}")
            return []

    async def add(
        self,
        tenant_id: str,
        messages: List[Dict[str, Any]],
        infer: bool = True,
    ) -> None:
        """Add messages for long-term knowledge extraction.

        Momex internally handles entity extraction, contradiction detection,
        and index updates when infer=True.

        Args:
            tenant_id: User/tenant identifier
            messages: Conversation messages to process
            infer: Whether to run knowledge extraction (default True)
        """
        self._ensure_initialized()
        try:
            await self._memory.add_messages(
                messages=messages,
                user_id=tenant_id,
                infer=infer,
            )
        except Exception as e:
            logger.warning(f"Failed to add memories for {tenant_id}: {e}")
