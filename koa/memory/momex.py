"""
Koa Momex Integration - Wrapper around momex memory system.

Provides long-term structured memory (RAG) with multi-tenant isolation
via collection naming: "tenant:{tenant_id}".

Conversation history is managed by the app layer.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MomexMemory:
    """
    Wrapper around momex Memory for long-term knowledge (RAG).

    Multi-tenant isolation: each tenant_id gets its own collection,
    so memories are never shared across tenants.

    Conversation history is managed by the app layer, not here.

    Args:
        llm_provider: LLM provider for knowledge extraction (openai/anthropic/azure/deepseek/qwen)
        llm_model: Model name
        llm_api_key: API key
        llm_api_base: Base URL (for Azure or custom endpoints)
        database_url: PostgreSQL DSN (reuses Koa's database)
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "",
        llm_api_key: str = "",
        llm_api_base: str = "",
        database_url: str = "",
        embedding_provider: str = "",
        embedding_model: str = "",
        embedding_api_key: str = "",
        embedding_api_base: str = "",
        embedding_api_version: str = "",
    ):
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._llm_api_base = llm_api_base
        self._database_url = database_url
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key
        self._embedding_api_base = embedding_api_base
        self._embedding_api_version = embedding_api_version
        self._config = None

        # Cache: tenant_id -> Memory instance
        self._memories: Dict[str, Any] = {}

    def _get_config(self):
        """Build MomexConfig from Koa's LLM config (lazy, once)."""
        if self._config is not None:
            return self._config

        from momex import MomexConfig
        from momex.config import EmbeddingConfig, LLMConfig, StorageConfig

        llm = LLMConfig(
            provider=self._llm_provider,
            model=self._llm_model,
            api_key=self._llm_api_key,
            api_base=self._llm_api_base,
        )

        # Embedding config (independent provider/endpoint support)
        embedding = None
        if self._embedding_api_key:
            provider = self._embedding_provider or (
                "azure" if self._embedding_api_base else "openai"
            )
            embedding_kwargs = {
                "provider": provider,
                "api_key": self._embedding_api_key,
            }
            if self._embedding_api_base:
                embedding_kwargs["api_base"] = self._embedding_api_base
            if self._embedding_model:
                embedding_kwargs["model"] = self._embedding_model
            if self._embedding_api_version:
                embedding_kwargs["api_version"] = self._embedding_api_version
            embedding = EmbeddingConfig(**embedding_kwargs)

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
                {
                    "text": item.text,
                    "type": item.type,
                    "score": item.score,
                    "timestamp": item.timestamp,
                }
                for item in results
            ]
        except Exception as e:
            import traceback

            logger.warning(
                f"Failed to search memories for {tenant_id}: {e}\n{traceback.format_exc()}"
            )
            return []

    async def add(
        self,
        tenant_id: str,
        messages: List[Dict[str, Any]],
        infer: bool = True,
        *,
        moderator: Any = None,
    ) -> None:
        """Add messages for long-term knowledge extraction.

        Momex handles entity extraction, contradiction detection,
        and index updates when infer=True.

        When ``moderator`` is provided (``koa.memory.governance.ContentModerator``),
        each message's textual content is checked and messages that fail
        moderation are silently dropped.  This is the primary defense against
        memory-poisoning attacks: malicious content in a single user turn
        cannot corrupt the long-term store.
        """
        # Moderate before persisting.  We do this at the wrapper layer rather
        # than relying on the orchestrator's call-site so every caller of
        # MomexMemory.add benefits.
        if moderator is not None and messages:
            filtered: List[Dict[str, Any]] = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = " ".join(
                        str(p.get("text") or "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") in (None, "text")
                    )
                else:
                    text = ""
                try:
                    verdict = await moderator.check(text)
                except Exception as exc:
                    logger.warning("Moderator raised; failing open: %s", exc)
                    verdict = None
                if verdict is not None and not getattr(verdict, "allowed", True):
                    logger.info(
                        "[MomexMemory] Dropped message from memory write: %s",
                        getattr(verdict, "reason", ""),
                    )
                    continue
                filtered.append(msg)
            messages = filtered
            if not messages:
                return
        try:
            memory = self._get_memory(tenant_id)
            await memory.add(messages=messages, infer=infer)
        except Exception as e:
            import traceback

            logger.warning(f"Failed to add memories for {tenant_id}: {e}\n{traceback.format_exc()}")

    async def delete_for_tenant(self, tenant_id: str) -> int:
        """Delete every memory for a tenant (GDPR Article 17 "right to erasure").

        Returns the number of rows affected.  Raises ``NotImplementedError``
        if the underlying momex backend does not expose a delete-all API;
        in that case callers must perform the deletion at the database layer.
        """
        try:
            memory = self._get_memory(tenant_id)
        except Exception as exc:
            logger.warning("delete_for_tenant: unable to load memory: %s", exc)
            return 0
        # momex.Memory may expose delete_all / reset.  Probe both.
        for method_name in ("delete_all", "reset", "clear"):
            method = getattr(memory, method_name, None)
            if callable(method):
                try:
                    result = method()
                    if hasattr(result, "__await__"):
                        result = await result
                    # Forget the cached Memory instance so a fresh one is
                    # created for subsequent writes.
                    self._memories.pop(tenant_id, None)
                    return int(result) if isinstance(result, int) else -1
                except Exception as exc:
                    logger.warning("delete_for_tenant via %s failed: %s", method_name, exc)
                    break
        raise NotImplementedError(
            "Underlying momex backend does not support delete-all. "
            "Perform the deletion at the database layer "
            "(e.g. DELETE FROM momex_memories WHERE collection = ?)."
        )

    async def forget_older_than(self, tenant_id: str, older_than_days: float) -> int:
        """Prune memories older than ``older_than_days`` days.

        Raises ``NotImplementedError`` if the backend cannot express TTL
        deletion natively.
        """
        try:
            memory = self._get_memory(tenant_id)
        except Exception as exc:
            logger.warning("forget_older_than: unable to load memory: %s", exc)
            return 0
        method = getattr(memory, "prune_older_than", None)
        if callable(method):
            try:
                result = method(older_than_days)
                if hasattr(result, "__await__"):
                    result = await result
                return int(result) if isinstance(result, int) else -1
            except Exception as exc:
                logger.warning("forget_older_than failed: %s", exc)
                raise
        raise NotImplementedError("Underlying momex backend does not support age-based pruning.")
