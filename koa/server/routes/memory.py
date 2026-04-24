"""Memory read endpoints — episode listing for backend/UI use.

Episodes live in Momex (see ``koa.memory.lifecycle.episode_memory``). This
module exposes a thin read surface so koi-backend can list/recall them on
behalf of the user-facing clients without needing to speak Momex directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from ...errors import E, KoaError
from ..app import require_app, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_momex(app):
    momex = getattr(app, "momex", None)
    if momex is None:
        raise KoaError(
            E.SERVICE_UNAVAILABLE, "Momex memory not initialised",
            details={"service": "momex"},
        )
    return momex


@router.get("/api/memory/episodes", dependencies=[Depends(verify_api_key)])
async def list_episodes(
    user_id: str = Query(..., description="Tenant / user id"),
    query: str = Query("", description="Free-text query; empty → most recent"),
    subkind: Optional[str] = Query(
        None,
        description="daily_log | weekly_reflection | behavioral_pattern",
    ),
    limit: int = Query(10, ge=1, le=50),
) -> Dict[str, Any]:
    """Recall episodes for a user via Momex vector search + kind filter."""
    from ...memory.lifecycle.episode_memory import EpisodeMemory

    momex = _require_momex(require_app())
    em = EpisodeMemory(momex)

    if not query:
        # "Most recent" semantics — let Momex surface whatever is freshest in
        # the embedding space under a generic temporal probe.
        items = await em.recall_recent_episodes(
            user_id, subkind=subkind or "daily_log", limit=limit,
        )
    else:
        items = await em.recall_episodes(
            user_id, query, limit=limit, subkind=subkind,
        )

    return {"user_id": user_id, "count": len(items), "episodes": items}
