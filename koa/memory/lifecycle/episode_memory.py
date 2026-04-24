"""EpisodeMemory — thin adapter over MomexMemory for episodic storage.

Why not a dedicated ``episodes`` table? Momex (TypeAgent-backed) already
gives us tenant isolation, vector search, entity extraction, and
contradiction handling. An "episode" (a weekly reflection, a daily log,
a behavioural pattern summary) is just a Momex entry with a metadata
convention:

    metadata = {
        "kind": "episode",
        "subkind": "daily_log" | "weekly_reflection" | "behavioral_pattern",
        "start_ts": <iso>, "end_ts": <iso>,
        "source": "sensing" | "reflection" | ...,
        ...                # free-form extras (importance, mood, etc.)
    }

Writes use ``infer=False`` so Momex treats the summary text as an atomic
unit — we've already done the reflection, so we don't want Momex to
shred it into fragment-level facts.

Reads fall back to in-Python filtering on ``kind``/``subkind`` since
Momex's public search surface doesn't accept metadata predicates. That's
acceptable for our recall sizes (tens to low hundreds per user).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EPISODE_KIND = "episode"


def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).isoformat()
    if isinstance(ts, str):
        return ts
    return str(ts)


def _extract_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Momex search results expose ``text/type/score/timestamp`` and an
    opaque JSON blob for the original message. Best-effort dig for metadata.
    """
    for key in ("metadata", "meta", "extra"):
        v = item.get(key)
        if isinstance(v, dict):
            return v
    return {}


class EpisodeMemory:
    """High-level episode read/write over MomexMemory.

    The caller is responsible for passing a live ``MomexMemory`` instance
    (usually ``app.momex``). This wrapper is stateless.
    """

    def __init__(self, momex: Any):
        self._momex = momex

    async def write_episode(
        self,
        tenant_id: str,
        *,
        summary: str,
        subkind: str,
        start_ts: Any = None,
        end_ts: Any = None,
        source: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist an episode as a Momex system message with kind=episode.

        Idempotency: Momex itself de-duplicates on identical content + tenant
        under its own rules, which is sufficient for weekly/daily reruns.
        """
        if not summary or not subkind:
            return
        if self._momex is None:
            logger.debug("EpisodeMemory.write skipped: Momex not configured")
            return

        metadata: Dict[str, Any] = {
            "kind": EPISODE_KIND,
            "subkind": subkind,
        }
        start_iso = _iso(start_ts)
        end_iso = _iso(end_ts)
        if start_iso:
            metadata["start_ts"] = start_iso
        if end_iso:
            metadata["end_ts"] = end_iso
        if source:
            metadata["source"] = source
        if extras:
            for k, v in extras.items():
                metadata.setdefault(k, v)

        message = {
            "role": "system",
            "content": summary,
            "metadata": metadata,
        }
        try:
            await self._momex.add(
                tenant_id=tenant_id,
                messages=[message],
                infer=False,
            )
        except Exception as exc:
            logger.warning("EpisodeMemory.write_episode failed: %s", exc)

    async def recall_episodes(
        self,
        tenant_id: str,
        query: str,
        *,
        limit: int = 5,
        subkind: Optional[str] = None,
        # ``oversample`` controls how many Momex results we fetch before
        # filtering client-side for the episode kind. Tuned so that even
        # tenants with a lot of conversational memory still produce enough
        # episode hits to satisfy ``limit``.
        oversample: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return up to ``limit`` episode entries best matching ``query``.

        Results are plain dicts (not typed) so the caller can render freely.
        """
        if self._momex is None or not query:
            return []
        try:
            fetched = await self._momex.search(
                tenant_id=tenant_id,
                query=query,
                limit=max(limit * oversample, limit),
            )
        except Exception as exc:
            logger.warning("EpisodeMemory.recall_episodes search failed: %s", exc)
            return []

        out: List[Dict[str, Any]] = []
        for item in fetched or []:
            meta = _extract_metadata(item)
            if meta.get("kind") != EPISODE_KIND:
                continue
            if subkind and meta.get("subkind") != subkind:
                continue
            out.append(
                {
                    "text": item.get("text"),
                    "score": item.get("score"),
                    "timestamp": item.get("timestamp"),
                    "metadata": meta,
                }
            )
            if len(out) >= limit:
                break
        return out

    async def recall_recent_episodes(
        self,
        tenant_id: str,
        *,
        subkind: str,
        limit: int = 7,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper: recall the most recent ``subkind`` episodes.

        Uses a deliberately-generic query ("recent {subkind}") so Momex's
        vector search surfaces temporally-recent items. The reflector uses
        this to stitch a week's worth of daily_log episodes together.
        """
        return await self.recall_episodes(
            tenant_id,
            f"recent {subkind}",
            limit=limit,
            subkind=subkind,
            oversample=10,
        )


__all__ = ["EpisodeMemory", "EPISODE_KIND"]
