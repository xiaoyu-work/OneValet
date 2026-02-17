"""Email event ingestion routes."""

from fastapi import APIRouter, HTTPException

from ..app import get_app_instance
from ..models import EmailEventRequest

router = APIRouter()


@router.post("/api/events/email")
async def ingest_email_event(req: EmailEventRequest):
    """Ingest an email event and publish to the EventBus."""
    from ...triggers.event_bus import Event

    _app = get_app_instance()
    event_bus = getattr(_app, "_event_bus", None) if _app else None
    if event_bus is None:
        raise HTTPException(503, "EventBus not available")

    request_data = req.model_dump()
    event = Event(
        source="email",
        event_type="received",
        data=request_data,
        tenant_id=req.tenant_id,
    )
    await event_bus.publish(event)
    return {"status": "ok"}
