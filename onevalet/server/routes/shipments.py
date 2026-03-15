"""Internal shipment API routes (service-to-service)."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...errors import OneValetError, E
from ..app import require_app, verify_service_key
from onevalet.builtin_agents.shipment.shipment_repo import ShipmentRepository
from onevalet.providers.shipment import TrackingProvider

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_repo() -> ShipmentRepository:
    app = require_app()
    await app._ensure_initialized()
    return ShipmentRepository(app._database)


class UpsertShipmentRequest(BaseModel):
    tenant_id: str
    tracking_number: str
    carrier: str
    tracking_url: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    last_update: Optional[str] = None
    estimated_delivery: Optional[str] = None
    tracking_history: Optional[list] = None
    delivered_notified: Optional[bool] = None
    is_active: Optional[bool] = None


class ArchiveByTrackingRequest(BaseModel):
    tenant_id: str
    tracking_number: str


# --- Internal Shipment APIs (service-to-service) ---


@router.get("/api/internal/shipments/list")
async def internal_list_shipments(
    request: Request,
    tenant_id: str,
    active_only: bool = True,
):
    """List all shipments for a tenant. Internal use only."""
    verify_service_key(request)
    repo = await _get_repo()
    query = "SELECT * FROM shipments WHERE tenant_id = $1"
    params = [tenant_id]
    if active_only:
        query += " AND is_active = TRUE"
    query += " ORDER BY updated_at DESC"
    rows = await repo.db.fetch(query, *params)
    return [dict(r) for r in rows]


@router.get("/api/internal/shipments")
async def internal_get_shipment(
    request: Request,
    tenant_id: str,
    tracking_number: str,
    carrier: Optional[str] = None,
):
    """Get a shipment by tenant_id + tracking_number. Internal use only."""
    verify_service_key(request)
    repo = await _get_repo()
    rows = await repo.db.fetch(
        "SELECT * FROM shipments WHERE tenant_id = $1 AND tracking_number = $2"
        + (" AND carrier = $3" if carrier else "")
        + " LIMIT 1",
        tenant_id,
        tracking_number,
        *([carrier] if carrier else []),
    )
    if not rows:
        raise OneValetError(E.NOT_FOUND, "Shipment not found",
                            details={"resource": "shipment"})
    return dict(rows[0])


@router.get("/api/internal/shipments/by-tracking")
async def internal_get_shipment_by_tracking(
    request: Request,
    tracking_number: str,
):
    """Get a shipment by tracking_number only (cross-tenant, for webhooks). Internal use only."""
    verify_service_key(request)
    repo = await _get_repo()
    rows = await repo.db.fetch(
        "SELECT * FROM shipments WHERE tracking_number = $1 AND is_active = TRUE LIMIT 1",
        tracking_number.upper(),
    )
    if not rows:
        raise OneValetError(E.NOT_FOUND, "Shipment not found",
                            details={"resource": "shipment"})
    return dict(rows[0])


@router.put("/api/internal/shipments")
async def internal_upsert_shipment(
    request: Request,
    body: UpsertShipmentRequest,
):
    """Upsert a shipment. Internal use only."""
    verify_service_key(request)
    repo = await _get_repo()
    kwargs = {}
    for field in (
        "tracking_url", "status", "description", "last_update",
        "estimated_delivery", "tracking_history", "delivered_notified", "is_active",
    ):
        val = getattr(body, field)
        if val is not None:
            kwargs[field] = val

    result = await repo.upsert_shipment(
        tenant_id=body.tenant_id,
        tracking_number=body.tracking_number,
        carrier=body.carrier,
        **kwargs,
    )
    if not result:
        raise OneValetError(E.INTERNAL_ERROR, "Failed to upsert shipment")
    return result


@router.put("/api/internal/shipments/archive")
async def internal_archive_shipment(
    request: Request,
    shipment_id: str,
):
    """Archive a shipment by ID. Internal use only."""
    verify_service_key(request)
    repo = await _get_repo()
    result = await repo.archive_shipment(shipment_id)
    if not result:
        raise OneValetError(E.NOT_FOUND, "Shipment not found",
                            details={"resource": "shipment"})
    return result


@router.put("/api/internal/shipments/archive-by-tracking")
async def internal_archive_by_tracking(
    request: Request,
    body: ArchiveByTrackingRequest,
):
    """Archive a shipment by tenant_id + tracking_number. Internal use only."""
    verify_service_key(request)
    repo = await _get_repo()
    result = await repo.archive_shipment_by_tracking(body.tenant_id, body.tracking_number)
    if not result:
        raise OneValetError(E.NOT_FOUND, "Shipment not found",
                            details={"resource": "shipment"})
    return result


@router.post("/api/internal/shipments/refresh")
async def internal_refresh_shipments(
    request: Request,
    tenant_id: str,
    timezone: Optional[str] = None,
):
    """Refresh all active non-delivered shipments from 17Track, then return updated list."""
    verify_service_key(request)
    repo = await _get_repo()
    provider = TrackingProvider()

    # Persist timezone for background poller
    if timezone:
        try:
            app = require_app()
            await app._database.execute(
                """
                INSERT INTO tenant_profiles (tenant_id, profile, extracted_at, updated_at)
                VALUES ($1, jsonb_build_object('timezone', $2), NOW(), NOW())
                ON CONFLICT (tenant_id)
                DO UPDATE SET profile = tenant_profiles.profile || jsonb_build_object('timezone', $2),
                              updated_at = NOW()
                """,
                tenant_id, timezone,
            )
        except Exception as e:
            logger.debug(f"Failed to persist timezone for {tenant_id}: {e}")

    if not provider.api_key:
        # No API key — just return cached data
        rows = await repo.db.fetch(
            "SELECT * FROM shipments WHERE tenant_id = $1 AND is_active = TRUE "
            "ORDER BY updated_at DESC",
            tenant_id,
        )
        return [dict(r) for r in rows]

    rows = await repo.db.fetch(
        "SELECT * FROM shipments WHERE tenant_id = $1 AND is_active = TRUE "
        "ORDER BY updated_at DESC",
        tenant_id,
    )
    shipments = [dict(r) for r in rows]

    if not shipments:
        return []

    async def refresh_one(shipment: dict) -> dict:
        """Refresh a single shipment from 17Track if not already delivered."""
        if (shipment.get("status") or "").lower() == "delivered":
            return shipment
        tn = shipment["tracking_number"]
        carrier = shipment.get("carrier", "")
        try:
            result = await provider.track(tn, carrier)
            if result.get("success"):
                updated = await repo.upsert_shipment(
                    tenant_id=tenant_id,
                    tracking_number=tn,
                    carrier=carrier or result.get("carrier", ""),
                    tracking_url=result.get("tracking_url"),
                    status=result.get("status", "unknown"),
                    description=shipment.get("description"),
                    last_update=result.get("last_update"),
                    estimated_delivery=result.get("estimated_delivery"),
                    tracking_history=result.get("events", []),
                )
                if result.get("status") == "delivered":
                    await repo.archive_shipment_by_tracking(tenant_id, tn)
                    if updated:
                        updated["is_active"] = False
                return updated or shipment
        except Exception as e:
            logger.warning(f"Failed to refresh {tn}: {e}")
        return shipment

    updated = await asyncio.gather(*[refresh_one(s) for s in shipments])
    # Filter out archived (delivered) ones
    return [s for s in updated if s.get("is_active", True)]
