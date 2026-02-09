"""
FlowAgent Memory Manager - Mem0-based long-term memory

This module provides:
- MemoryManager: Main class wrapping mem0 for agent memory
- Auto-recall and auto-store functionality
"""

from typing import Dict, Any, List, Optional, Protocol

from .models import MemoryConfig, RecallResult, StoreResult


class MemoryManager:
    """
    Memory manager using mem0 as backend.

    Supports both:
    - Platform version (with API key)
    - Self-hosted version (with local vector store)

    Example (Platform):
        config = MemoryConfig(
            enabled=True,
            use_platform=True,
            api_key="your-mem0-api-key"
        )
        manager = MemoryManager(config)

    Example (Self-hosted):
        config = MemoryConfig(
            enabled=True,
            use_platform=False,
            vector_store_provider="qdrant",
            vector_store_config={"host": "localhost", "port": 6333}
        )
        manager = MemoryManager(config)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._memory = None
        self._client = None

        if self.config.enabled:
            self._init_mem0()

    def _init_mem0(self) -> None:
        """Initialize mem0 client"""
        if self.config.use_platform:
            # Platform version
            from mem0 import MemoryClient
            self._client = MemoryClient(api_key=self.config.api_key)
        else:
            # Self-hosted version
            from mem0 import Memory
            mem0_config = self.config.to_mem0_config()
            self._memory = Memory.from_config(mem0_config)

    def add(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        agent_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoreResult:
        """
        Add memories from conversation messages.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            user_id: User identifier
            agent_type: Optional agent type for filtering
            metadata: Additional metadata

        Returns:
            StoreResult
        """
        if not self.config.enabled:
            return StoreResult()

        result = StoreResult()
        meta = metadata or {}
        if agent_type:
            meta["agent_type"] = agent_type

        try:
            if self._client:
                # Platform version
                response = self._client.add(messages, user_id=user_id, metadata=meta)
            else:
                # Self-hosted version
                response = self._memory.add(messages, user_id=user_id, metadata=meta)

            # Extract memory IDs from response
            if isinstance(response, dict) and "results" in response:
                for item in response["results"]:
                    if "id" in item:
                        result.memory_ids.append(item["id"])
                        result.stored_count += 1
            elif isinstance(response, list):
                for item in response:
                    if isinstance(item, dict) and "id" in item:
                        result.memory_ids.append(item["id"])
                        result.stored_count += 1

        except Exception as e:
            result.errors.append(str(e))

        return result

    def add_text(
        self,
        text: str,
        user_id: str,
        agent_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoreResult:
        """
        Add a single text memory.

        Args:
            text: Text to remember
            user_id: User identifier
            agent_type: Optional agent type
            metadata: Additional metadata

        Returns:
            StoreResult
        """
        if not self.config.enabled:
            return StoreResult()

        result = StoreResult()
        meta = metadata or {}
        if agent_type:
            meta["agent_type"] = agent_type

        try:
            if self._client:
                response = self._client.add(text, user_id=user_id, metadata=meta)
            else:
                response = self._memory.add(text, user_id=user_id, metadata=meta)

            if isinstance(response, dict) and "results" in response:
                for item in response["results"]:
                    if "id" in item:
                        result.memory_ids.append(item["id"])
                        result.stored_count += 1

        except Exception as e:
            result.errors.append(str(e))

        return result

    def search(
        self,
        query: str,
        user_id: str,
        agent_type: Optional[str] = None,
        limit: int = 10
    ) -> RecallResult:
        """
        Search memories.

        Args:
            query: Search query
            user_id: User identifier
            agent_type: Optional agent type filter
            limit: Maximum results

        Returns:
            RecallResult with matching memories
        """
        if not self.config.enabled:
            return RecallResult(query=query)

        result = RecallResult(query=query)
        filters = {"user_id": user_id}
        if agent_type:
            filters["agent_type"] = agent_type

        try:
            if self._client:
                # mem0 platform API uses filters parameter
                response = self._client.search(query, filters={"user_id": user_id}, limit=limit)
            else:
                response = self._memory.search(query, user_id=user_id, limit=limit)

            # Extract memories from response
            if isinstance(response, dict) and "results" in response:
                result.memories = response["results"]
            elif isinstance(response, list):
                result.memories = response

        except Exception as e:
            pass  # Return empty result on error

        return result

    def get_all(
        self,
        user_id: str,
        agent_type: Optional[str] = None
    ) -> RecallResult:
        """
        Get all memories for a user.

        Args:
            user_id: User identifier
            agent_type: Optional agent type filter

        Returns:
            RecallResult with all memories
        """
        if not self.config.enabled:
            return RecallResult()

        result = RecallResult()

        try:
            if self._client:
                response = self._client.get_all(user_id=user_id)
            else:
                response = self._memory.get_all(user_id=user_id)

            if isinstance(response, dict) and "results" in response:
                memories = response["results"]
            elif isinstance(response, list):
                memories = response
            else:
                memories = []

            # Filter by agent_type if specified
            if agent_type:
                memories = [
                    m for m in memories
                    if m.get("metadata", {}).get("agent_type") == agent_type
                ]

            result.memories = memories

        except Exception as e:
            pass

        return result

    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted
        """
        if not self.config.enabled:
            return False

        try:
            if self._client:
                self._client.delete(memory_id)
            else:
                self._memory.delete(memory_id)
            return True
        except Exception:
            return False

    def delete_all(self, user_id: str) -> bool:
        """
        Delete all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            True if deleted
        """
        if not self.config.enabled:
            return False

        try:
            if self._client:
                self._client.delete_all(user_id=user_id)
            else:
                self._memory.delete_all(user_id=user_id)
            return True
        except Exception:
            return False

    def auto_recall(
        self,
        user_id: str,
        field_names: List[str],
        agent_type: Optional[str] = None
    ) -> RecallResult:
        """
        Auto-recall fields by searching for each field name.

        Args:
            user_id: User identifier
            field_names: Field names to recall
            agent_type: Optional agent type filter

        Returns:
            RecallResult with recalled fields
        """
        if not self.config.enabled or not self.config.auto_recall:
            return RecallResult()

        result = RecallResult()

        # Build search query from field names
        fields_to_search = [
            f for f in field_names
            if self.config.should_remember(f)
        ]

        if not fields_to_search:
            return result

        # Search for user's preferences and information
        query = f"user information about: {', '.join(fields_to_search)}"
        search_result = self.search(query, user_id=user_id, agent_type=agent_type)

        result.memories = search_result.memories
        result.query = query

        # Try to extract field values from memories
        for memory in result.memories:
            memory_text = memory.get("memory", "").lower()
            for field in fields_to_search:
                # Simple extraction - look for patterns like "email is xxx" or "email: xxx"
                if field.lower() in memory_text:
                    result.recalled_fields[field] = memory.get("memory")
                    break

        return result

    def auto_store(
        self,
        user_id: str,
        agent_type: str,
        fields: Dict[str, Any]
    ) -> StoreResult:
        """
        Auto-store collected fields as memories.

        Args:
            user_id: User identifier
            agent_type: Agent type
            fields: Fields to store

        Returns:
            StoreResult
        """
        if not self.config.enabled or not self.config.auto_store:
            return StoreResult()

        result = StoreResult()

        # Filter fields
        fields_to_store = {
            k: v for k, v in fields.items()
            if v is not None and self.config.should_remember(k)
        }

        if not fields_to_store:
            return result

        # Build memory text from fields
        memory_parts = []
        for key, value in fields_to_store.items():
            memory_parts.append(f"User's {key} is {value}")

        memory_text = ". ".join(memory_parts)

        # Store as single memory
        store_result = self.add_text(
            text=memory_text,
            user_id=user_id,
            agent_type=agent_type,
            metadata={"fields": list(fields_to_store.keys())}
        )

        return store_result


class MemoryMixin:
    """
    Mixin class to add mem0 memory support to agents.

    Usage:
        class MyAgent(MemoryMixin, StandardAgent):
            def __init__(self, memory_manager: MemoryManager):
                self.set_memory_manager(memory_manager)

            async def on_waiting_for_input(self, msg):
                # Auto-recall
                recalled = await self._auto_recall_fields(["email", "name"])
                # Use recalled.memories or recalled.recalled_fields

            async def on_completed(self, msg):
                # Auto-store
                await self._auto_store_fields()
    """

    _memory_manager: Optional[MemoryManager] = None

    def set_memory_manager(self, manager: MemoryManager) -> None:
        """Set the memory manager"""
        self._memory_manager = manager

    def _auto_recall_fields(
        self,
        field_names: Optional[List[str]] = None
    ) -> RecallResult:
        """
        Auto-recall fields from memory.

        Args:
            field_names: Fields to recall

        Returns:
            RecallResult
        """
        if not self._memory_manager:
            return RecallResult()

        user_id = getattr(self, 'user_id', None)
        if not user_id:
            return RecallResult()

        if field_names is None:
            required_fields = getattr(self, 'required_fields', [])
            field_names = [
                f.name if hasattr(f, 'name') else str(f)
                for f in required_fields
            ]

        if not field_names:
            return RecallResult()

        agent_type = getattr(self, 'agent_type', None)

        return self._memory_manager.auto_recall(
            user_id=user_id,
            field_names=field_names,
            agent_type=agent_type
        )

    def _auto_store_fields(self) -> StoreResult:
        """
        Auto-store collected fields.

        Returns:
            StoreResult
        """
        if not self._memory_manager:
            return StoreResult()

        user_id = getattr(self, 'user_id', None)
        if not user_id:
            return StoreResult()

        agent_type = getattr(self, 'agent_type', type(self).__name__)
        collected_fields = getattr(self, 'collected_fields', {})

        return self._memory_manager.auto_store(
            user_id=user_id,
            agent_type=agent_type,
            fields=collected_fields
        )


# Global memory manager
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> Optional[MemoryManager]:
    """Get global memory manager"""
    return _global_memory_manager


def configure_memory(config: MemoryConfig) -> MemoryManager:
    """
    Configure global memory manager.

    Args:
        config: Memory configuration

    Returns:
        MemoryManager instance
    """
    global _global_memory_manager
    _global_memory_manager = MemoryManager(config)
    return _global_memory_manager
