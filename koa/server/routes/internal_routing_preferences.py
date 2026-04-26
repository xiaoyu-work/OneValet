"""Internal routing-preferences CRUD — service-key protected.

Stores the user's preferred provider per surface (calendar, todo, ...).
The agent's ``resolve_surface_target`` reads this to pick a default
provider when the user hasn't explicitly named one. Previously this
data was supposed to live in koi-backend, but the endpoint was never
built — every read 404'd and the resolver always fell back to "local".
Now it lives in koa, alongside the data it routes to.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..app import require_app, verify_service_key

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_db(app):
    db = getattr(app, "database", None)
    if db is None:
        raise HTTPException(503, "Database not initialised")
    return db


def _row_to_pref(row: dict) -> Dict[str, Any]:
    return {
        "tenant_id": row["tenant_id"],
        "surface": row["surface"],
        "default_provider": row["default_provider"],
        "default_account": row.get("default_account"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


@router.get(
    "/api/internal/routing-preferences/{surface}",
    dependencies=[Depends(verify_service_key)],
)
async def get_routing_preference(
    surface: str,
    tenant_id: str = Query(...),
) -> Dict[str, Any]:
    db = _require_db(require_app())

    row = await db.fetchrow(
        "SELECT tenant_id, surface, default_provider, default_account, "
        "       created_at, updated_at "
        "FROM tenant_default.user_routing_preferences "
        "WHERE tenant_id = $1 AND surface = $2",
        tenant_id,
        surface,
    )
    if row is None:
        raise HTTPException(404, "No preference set for this surface")
    return {"preference": _row_to_pref(dict(row))}


class PreferenceUpsert(BaseModel):
    tenant_id: str
    surface: str
    default_provider: str
    default_account: Optional[str] = None


@router.post(
    "/api/internal/routing-preferences",
    dependencies=[Depends(verify_service_key)],
)
async def upsert_routing_preference(req: PreferenceUpsert) -> Dict[str, Any]:
    db = _require_db(require_app())

    row = await db.fetchrow(
        """
        INSERT INTO tenant_default.user_routing_preferences
            (tenant_id, surface, default_provider, default_account, created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW())
        ON CONFLICT (tenant_id, surface) DO UPDATE SET
            default_provider = EXCLUDED.default_provider,
            default_account = EXCLUDED.default_account,
            updated_at = NOW()
        RETURNING tenant_id, surface, default_provider, default_account,
                  created_at, updated_at
        """,
        req.tenant_id,
        req.surface,
        req.default_provider,
        req.default_account,
    )
    return {"preference": _row_to_pref(dict(row))}
