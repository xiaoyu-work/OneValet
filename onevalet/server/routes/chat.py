"""Chat, streaming, health, and session routes."""

import dataclasses
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..app import require_app
from ..models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    app = require_app()
    result = await app.chat(
        message_or_tenant_id=req.tenant_id,
        message=req.message,
        metadata=req.metadata,
    )
    return ChatResponse(
        response=result.raw_message or "",
        status=result.status.value if result.status else "completed",
    )


@router.post("/stream")
async def stream(req: ChatRequest):
    app = require_app()

    async def event_generator():
        async for event in app.stream(
            message_or_tenant_id=req.tenant_id,
            message=req.message,
            metadata=req.metadata,
        ):
            def _default(obj):
                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    result = {}
                    for f in dataclasses.fields(obj):
                        val = getattr(obj, f.name)
                        try:
                            json.dumps(val)
                            result[f.name] = val
                        except (TypeError, ValueError):
                            result[f.name] = str(val)
                    return result
                try:
                    return str(obj)
                except Exception:
                    return "<non-serializable>"

            data = json.dumps({
                "type": event.type.value if event.type else "unknown",
                "data": event.data,
            }, ensure_ascii=False, default=_default)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/api/clear-session")
async def clear_session(tenant_id: str = "default"):
    """Clear conversation history for a tenant."""
    app = require_app()
    await app._ensure_initialized()
    app._momex.clear_history(tenant_id=tenant_id, session_id=tenant_id)
    return {"status": "ok", "message": "Session history cleared"}
