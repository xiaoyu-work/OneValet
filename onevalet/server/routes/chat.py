"""Chat, streaming, health, and session routes."""

import asyncio
import dataclasses
import json
import logging
import os

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..app import require_app, verify_api_key
from ..models import ChatRequest, ChatResponse
from ..streaming.models import EventType

logger = logging.getLogger(__name__)

router = APIRouter()

_KOIAI_CALLBACK_URL = os.getenv("KOIAI_CALLBACK_URL", "")


async def _post_stream_result(
    tenant_id: str, final_response: str, tool_calls: list,
) -> None:
    """POST the stream result back to koiai so it can persist chat history."""
    if not _KOIAI_CALLBACK_URL:
        return
    payload = {
        "tenant_id": tenant_id,
        "response": final_response,
        "tool_calls": tool_calls,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                _KOIAI_CALLBACK_URL, json=payload,
                headers={"X-Service-Key": os.getenv("ONEVALET_SERVICE_KEY", "")},
            )
            if resp.status_code != 200:
                logger.warning(f"Stream result callback failed: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Stream result callback error: {e}")


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(req: ChatRequest):
    app = require_app()
    images = [img.model_dump() for img in req.images] if req.images else None
    metadata = dict(req.metadata or {})
    if req.conversation_history is not None:
        metadata["conversation_history"] = req.conversation_history
    result = await app.handle_message(
        tenant_id=req.tenant_id,
        message=req.message,
        images=images,
        metadata=metadata,
    )
    return ChatResponse(
        response=result.raw_message or "",
        status=result.status.value if result.status else "completed",
    )


@router.post("/stream", dependencies=[Depends(verify_api_key)])
async def stream(req: ChatRequest):
    app = require_app()

    images = [img.model_dump() for img in req.images] if req.images else None
    metadata = dict(req.metadata or {})
    if req.conversation_history is not None:
        metadata["conversation_history"] = req.conversation_history

    # Use a queue so the orchestrator runs to completion in a background task
    # even if the client disconnects mid-stream.
    _SENTINEL = object()
    queue: asyncio.Queue = asyncio.Queue()
    execution_end_data_holder: list = []  # mutable container for closure

    async def _run_orchestrator():
        try:
            async for event in app.stream_message(
                tenant_id=req.tenant_id,
                message=req.message,
                images=images,
                metadata=metadata,
            ):
                if event.type == EventType.EXECUTION_END:
                    execution_end_data_holder.append(event.data)
                await queue.put(event)
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
        finally:
            await queue.put(_SENTINEL)
            # Fire callback after orchestrator completes
            if execution_end_data_holder and _KOIAI_CALLBACK_URL:
                ed = execution_end_data_holder[0]
                await _post_stream_result(
                    tenant_id=req.tenant_id,
                    final_response=ed.get("final_response", ""),
                    tool_calls=ed.get("tool_calls", []),
                )

    async def event_generator():
        task = asyncio.create_task(_run_orchestrator())
        try:
            while True:
                event = await queue.get()
                if event is _SENTINEL:
                    break

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
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected — let the orchestrator task keep running
            pass

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
