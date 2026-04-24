"""Async embedding helper — thin wrapper over litellm.aembedding.

Used by the episode store / weekly reflector to generate embeddings for
pgvector kNN search.  We deliberately keep this independent of Momex's
internal embedding pipeline because:

  * Momex embeds into its own collection (typeagent) — callers can't reuse
    that output without going through Memory's API.
  * Our pgvector column lives in koi-backend's Postgres, not Momex's store.

Configuration is env-driven so the wrapper matches whatever provider the
rest of the stack is using:

  * ``KOI_EMBEDDING_PROVIDER``  — e.g. ``openai`` (default), ``azure``
  * ``KOI_EMBEDDING_MODEL``     — default ``text-embedding-3-small``
  * ``KOI_EMBEDDING_API_KEY``   — falls back to ``OPENAI_API_KEY`` /
                                  ``AZURE_OPENAI_API_KEY`` per provider
  * ``KOI_EMBEDDING_API_BASE``  — optional (needed for Azure)
  * ``KOI_EMBEDDING_API_VERSION`` — optional (needed for Azure)

``build_embedder()`` returns ``None`` if no API key is configured so
callers can gracefully fall back to keyword search.
"""
from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

Embedder = Callable[[str], Awaitable[Optional[List[float]]]]


def _resolve_api_key(provider: str) -> Optional[str]:
    key = os.getenv("KOI_EMBEDDING_API_KEY")
    if key:
        return key
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return None


def build_embedder() -> Optional[Embedder]:
    """Return an async embedder callable, or None if not configured."""
    provider = (os.getenv("KOI_EMBEDDING_PROVIDER") or "openai").lower()
    model = os.getenv("KOI_EMBEDDING_MODEL") or "text-embedding-3-small"
    api_key = _resolve_api_key(provider)
    if not api_key:
        return None

    api_base = os.getenv("KOI_EMBEDDING_API_BASE")
    api_version = os.getenv("KOI_EMBEDDING_API_VERSION")

    # litellm model string convention: "openai/<model>", "azure/<deployment>".
    if "/" not in model:
        model = f"{provider}/{model}"

    async def _embed(text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        try:
            # Import lazily to avoid hard dep + reduce startup cost.
            from litellm import aembedding  # type: ignore

            kwargs = {
                "model": model,
                "input": [text[:8000]],  # guard against oversize input
                "api_key": api_key,
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_version:
                kwargs["api_version"] = api_version
            resp = await aembedding(**kwargs)
            data = getattr(resp, "data", None) or (
                resp.get("data") if isinstance(resp, dict) else None
            )
            if not data:
                return None
            first = data[0]
            vec = first.get("embedding") if isinstance(first, dict) else getattr(first, "embedding", None)
            if vec is None:
                return None
            return list(vec)
        except Exception as e:
            logger.warning("embedding call failed: %s", e)
            return None

    return _embed


# Process-wide singleton so we don't re-create the closure on every request.
_singleton: Optional[Embedder] = None
_checked = False


def get_embedder() -> Optional[Embedder]:
    """Lazy singleton accessor for the shared embedder."""
    global _singleton, _checked
    if not _checked:
        _singleton = build_embedder()
        _checked = True
    return _singleton
