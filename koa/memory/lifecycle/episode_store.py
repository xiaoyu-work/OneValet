"""EpisodeStore — CRUD for ``episodes`` and ``entities`` + ``episode_entities``.

An *episode* is a narrative event ("Jay ran the 5K with Mia on Saturday").
Distinct from Momex (semantic facts) because episodes are indexed temporally
and emotionally, not just by semantic similarity.

The store handles:
  * embedding generation (delegated to the caller via an optional callback
    so we don't pin a provider here — different deployments may use
    OpenAI / Azure / Cohere / a local model).
  * upsert with conflict-safe dedupe on (user_id, local_date, title).
  * entity resolution (matches against entities.name; inserts new ones).
  * relationship linking via episode_entities.

Reading:
  * ``list_recent`` for chronological browsing.
  * ``find_similar`` — pgvector kNN if enabled, falls back to keyword.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

Embedder = Callable[[str], Awaitable[Optional[List[float]]]]


@dataclass
class EpisodeDraft:
    title: str
    summary: str
    local_date: date
    kind: str = "event"                 # event | milestone | reflection | routine_break
    mood: Optional[str] = None
    location: Optional[str] = None
    importance: int = 3                 # 1..5
    source: Optional[str] = None        # "weekly_reflection" | "proactive" | ...
    entities: List[str] = field(default_factory=list)  # display names
    metadata: Dict[str, Any] = field(default_factory=dict)


async def create_episode(
    db,
    user_id: str,
    draft: EpisodeDraft,
    embedder: Optional[Embedder] = None,
) -> Optional[str]:
    """Insert (or upsert) an episode.  Returns its id.

    The table has a uniqueness constraint we rely on to make this call
    idempotent when the WeeklyReflector reruns: (user_id, local_date, title).
    """
    embedding = await embedder(draft.summary or draft.title) if embedder else None

    try:
        row = await db.fetchrow(
            """INSERT INTO episodes
                 (user_id, local_date, title, summary, kind, mood, location,
                  importance, source, metadata, embedding, status)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::vector, 'active')
               ON CONFLICT (user_id, local_date, title) DO UPDATE SET
                  summary = EXCLUDED.summary,
                  kind = EXCLUDED.kind,
                  mood = EXCLUDED.mood,
                  location = EXCLUDED.location,
                  importance = EXCLUDED.importance,
                  source = EXCLUDED.source,
                  metadata = EXCLUDED.metadata,
                  embedding = COALESCE(EXCLUDED.embedding, episodes.embedding),
                  updated_at = NOW()
               RETURNING id""",
            user_id,
            draft.local_date,
            draft.title,
            draft.summary,
            draft.kind,
            draft.mood,
            draft.location,
            draft.importance,
            draft.source,
            json.dumps(draft.metadata),
            _to_vector(embedding),
        )
    except Exception as e:
        # pgvector might be unavailable (bytea fallback) — retry without embedding.
        logger.debug("create_episode vector insert failed (%s); retrying without embedding", e)
        try:
            row = await db.fetchrow(
                """INSERT INTO episodes
                     (user_id, local_date, title, summary, kind, mood, location,
                      importance, source, metadata, status)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, 'active')
                   ON CONFLICT (user_id, local_date, title) DO UPDATE SET
                      summary = EXCLUDED.summary,
                      kind = EXCLUDED.kind,
                      mood = EXCLUDED.mood,
                      location = EXCLUDED.location,
                      importance = EXCLUDED.importance,
                      source = EXCLUDED.source,
                      metadata = EXCLUDED.metadata,
                      updated_at = NOW()
                   RETURNING id""",
                user_id,
                draft.local_date,
                draft.title,
                draft.summary,
                draft.kind,
                draft.mood,
                draft.location,
                draft.importance,
                draft.source,
                json.dumps(draft.metadata),
            )
        except Exception as e2:
            logger.error("create_episode failed: %s", e2)
            return None

    episode_id = str(row["id"])

    if draft.entities:
        await _link_entities(db, user_id, episode_id, draft.entities, embedder)

    return episode_id


async def list_recent(
    db, user_id: str, limit: int = 50, before: Optional[date] = None,
) -> List[Dict[str, Any]]:
    sql = (
        """SELECT id, local_date, title, summary, kind, mood, location, importance
           FROM episodes
           WHERE user_id = $1 AND status = 'active'"""
    )
    args: List[Any] = [user_id]
    if before:
        sql += " AND local_date < $2"
        args.append(before)
    sql += " ORDER BY local_date DESC LIMIT " + str(int(limit))
    try:
        rows = await db.fetch(sql, *args)
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("list_recent failed: %s", e)
        return []


async def find_similar(
    db,
    user_id: str,
    query: str,
    k: int = 5,
    embedder: Optional[Embedder] = None,
) -> List[Dict[str, Any]]:
    """kNN via pgvector when available; keyword LIKE fallback otherwise."""
    embedding = await embedder(query) if embedder else None
    if embedding is not None:
        try:
            rows = await db.fetch(
                """SELECT id, local_date, title, summary, kind, mood, location, importance,
                          1 - (embedding <=> $2::vector) AS similarity
                   FROM episodes
                   WHERE user_id = $1 AND status = 'active' AND embedding IS NOT NULL
                   ORDER BY embedding <=> $2::vector
                   LIMIT $3""",
                user_id, _to_vector(embedding), k,
            )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug("vector search failed, falling back: %s", e)

    try:
        rows = await db.fetch(
            """SELECT id, local_date, title, summary, kind, mood, location, importance
               FROM episodes
               WHERE user_id = $1 AND status = 'active'
                 AND (title ILIKE $2 OR summary ILIKE $2)
               ORDER BY local_date DESC LIMIT $3""",
            user_id, f"%{query}%", k,
        )
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("keyword search failed: %s", e)
        return []


async def _link_entities(db, user_id: str, episode_id: str, names: Iterable[str], embedder: Optional[Embedder]):
    """Resolve each name to an entity row (inserting if needed) and create
    the episode_entities link."""
    for raw in names:
        name = (raw or "").strip()
        if not name:
            continue
        try:
            row = await db.fetchrow(
                """INSERT INTO entities (user_id, name, entity_type)
                   VALUES ($1, $2, 'unknown')
                   ON CONFLICT (user_id, name) DO UPDATE SET updated_at = NOW()
                   RETURNING id""",
                user_id, name,
            )
            if not row:
                continue
            entity_id = row["id"]
            await db.execute(
                """INSERT INTO episode_entities (episode_id, entity_id)
                   VALUES ($1, $2)
                   ON CONFLICT DO NOTHING""",
                episode_id, entity_id,
            )
        except Exception as e:
            logger.debug("entity link failed for %s: %s", name, e)


def _to_vector(embedding: Optional[List[float]]) -> Optional[str]:
    """Serialize an embedding for pgvector over asyncpg.

    pgvector's text input format is ``'[1.0,2.0,3.0]'`` — identical to a
    JSON array with square brackets and no spaces.  Sending the value as a
    string plus an explicit ``$N::vector`` cast in the SQL lets us avoid
    registering a custom asyncpg type codec at connection time (which would
    require touching every connection acquire in the pool).

    If pgvector is unavailable and the column degraded to bytea per the
    migration's DO-block fallback, the ``::vector`` cast raises and the
    caller's except clause routes to keyword search.
    """
    if embedding is None:
        return None
    return "[" + ",".join(format(float(x), ".7g") for x in embedding) + "]"
