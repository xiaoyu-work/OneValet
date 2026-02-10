"""
OneValet Momex Integration - Wrapper around momex memory system.

Provides long-term structured memory and short-term conversation history.
Multi-tenant isolation via collection naming: "tenant:{tenant_id}".
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MomexMemory:
    """
    Wrapper around momex Memory + ShortTermMemory.

    Multi-tenant isolation: each tenant_id gets its own collection,
    so memories are never shared across tenants.

    Args:
        llm_provider: LLM provider for knowledge extraction (openai/anthropic/azure/deepseek/qwen)
        llm_model: Model name
        llm_api_key: API key
        llm_api_base: Base URL (for Azure or custom endpoints)
        database_url: PostgreSQL DSN (reuses OneValet's database)
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "",
        llm_api_key: str = "",
        llm_api_base: str = "",
        database_url: str = "",
        embedding_api_key: str = "",
        embedding_api_base: str = "",
    ):
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._llm_api_base = llm_api_base
        self._database_url = database_url
        self._embedding_api_key = embedding_api_key
        self._embedding_api_base = embedding_api_base
        self._config = None

        # Cache: tenant_id -> Memory instance
        self._memories: Dict[str, Any] = {}
        # Cache: (tenant_id, session_id) -> ShortTermMemory instance
        self._short_terms: Dict[tuple, Any] = {}

    def _get_config(self):
        """Build MomexConfig from OneValet's LLM config (lazy, once)."""
        if self._config is not None:
            return self._config

        from momex import MomexConfig
        from momex.config import LLMConfig, EmbeddingConfig, StorageConfig

        llm = LLMConfig(
            provider=self._llm_provider,
            model=self._llm_model,
            api_key=self._llm_api_key,
            api_base=self._llm_api_base,
        )

        # Embedding always uses OpenAI (or Azure OpenAI)
        embedding = None
        if self._embedding_api_key:
            embedding_provider = "azure" if self._embedding_api_base else "openai"
            embedding = EmbeddingConfig(
                provider=embedding_provider,
                api_key=self._embedding_api_key,
                api_base=self._embedding_api_base,
            )

        storage = StorageConfig()
        if self._database_url:
            # Detect pgbouncer (Supabase pooler uses port 6543 or "pooler" in URL)
            is_pgbouncer = "pooler" in self._database_url or ":6543" in self._database_url
            storage = StorageConfig(
                backend="postgres",
                postgres_url=self._database_url,
                postgres_pgbouncer=is_pgbouncer,
            )

        self._config = MomexConfig(llm=llm, embedding=embedding, storage=storage)
        return self._config

    def _get_memory(self, tenant_id: str):
        """Get or create a Memory instance for a tenant."""
        if tenant_id not in self._memories:
            from momex import Memory
            collection = f"tenant:{tenant_id}"
            self._memories[tenant_id] = Memory(
                collection=collection,
                config=self._get_config(),
            )
        return self._memories[tenant_id]

    def _get_short_term(self, tenant_id: str, session_id: str):
        """Get or create a ShortTermMemory instance for a tenant+session."""
        key = (tenant_id, session_id)
        if key not in self._short_terms:
            from momex import ShortTermMemory
            collection = f"tenant:{tenant_id}"
            self._short_terms[key] = ShortTermMemory(
                collection=collection,
                config=self._get_config(),
                session_id=session_id,
            )
        return self._short_terms[key]

    def get_history(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Load conversation history from short-term memory.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        try:
            stm = self._get_short_term(tenant_id, session_id)
            messages = stm.get(limit=limit)
            return [{"role": m.role, "content": m.content} for m in messages]
        except Exception as e:
            logger.warning(f"Failed to load history for {tenant_id}/{session_id}: {e}")
            return []

    def save_message(
        self,
        tenant_id: str,
        session_id: str,
        content: str,
        role: str = "user",
    ) -> None:
        """Save a single message to short-term memory."""
        try:
            stm = self._get_short_term(tenant_id, session_id)
            stm.add(content=content, role=role)
        except Exception as e:
            logger.warning(f"Failed to save message for {tenant_id}/{session_id}: {e}")

    async def search(
        self,
        tenant_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search long-term memories.

        Returns:
            List of dicts with 'text', 'type', and 'score' keys.
        """
        try:
            memory = self._get_memory(tenant_id)
            results = await memory.search(query_text=query, limit=limit)
            return [
                {"text": item.text, "type": item.type, "score": item.score}
                for item in results
            ]
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

        Momex handles entity extraction, contradiction detection,
        and index updates when infer=True.
        """
        try:
            memory = self._get_memory(tenant_id)
            await memory.add(messages=messages, infer=infer)
        except Exception as e:
            logger.warning(f"Failed to add memories for {tenant_id}: {e}")
