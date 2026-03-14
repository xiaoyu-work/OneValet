"""Email event ingestion routes."""

from fastapi import APIRouter, Depends

from ...errors import OneValetError, E
from ..app import require_app, verify_api_key
from ..models import EmailEventRequest

router = APIRouter()


@router.post("/api/events/email", dependencies=[Depends(verify_api_key)])
async def ingest_email_event(req: EmailEventRequest):
    """Ingest an email event and evaluate importance."""
    app = require_app()
    if app.email_handler is None:
        raise OneValetError(E.SERVICE_UNAVAILABLE, "Email handler not available",
                            details={"service": "events"})

    await app.email_handler.handle_email(req.tenant_id, req.model_dump())
    return {"status": "ok"}
