"""Chat, streaming, health, and session routes."""

import dataclasses
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..app import require_app, verify_api_key
from ..models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    app = require_app()
    images = [img.model_dump() for img in req.images] if req.images else None
    result = await app.handle_message(
        tenant_id=req.tenant_id,
        message=req.message,
        images=images,
        metadata=req.metadata,
    )
    return ChatResponse(
        response=result.raw_message or "",
        status=result.status.value if result.status else "completed",
    )


@router.post("/stream", dependencies=[Depends(verify_api_key)])
async def stream(req: ChatRequest):
    app = require_app()

    images = [img.model_dump() for img in req.images] if req.images else None

    async def event_generator():
        async for event in app.stream_message(
            tenant_id=req.tenant_id,
            message=req.message,
            images=images,
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


@router.post("/api/clear-session", dependencies=[Depends(verify_api_key)])
async def clear_session(tenant_id: str = "default"):
    """Clear conversation history for a tenant."""
    app = require_app()
    await app.clear_session(tenant_id)
    return {"status": "ok", "message": "Session history cleared"}
