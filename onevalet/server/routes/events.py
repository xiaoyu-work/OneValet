"""Email event ingestion routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..app import require_app, verify_api_key
from ..models import EmailEventRequest

router = APIRouter()


@router.post("/api/events/email", dependencies=[Depends(verify_api_key)])
async def ingest_email_event(req: EmailEventRequest):
    """Ingest an email event and publish to the EventBus."""
    from ...triggers.event_bus import Event

    app = require_app()
    if app.event_bus is None:
        raise HTTPException(503, "EventBus not available")

    request_data = req.model_dump()
    event = Event(
        source="email",
        event_type="received",
        data=request_data,
        tenant_id=req.tenant_id,
    )
    await app.ingest_event(event)
    return {"status": "ok"}
