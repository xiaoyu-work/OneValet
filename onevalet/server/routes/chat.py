"""Chat, streaming, health, and session routes."""

import asyncio
import dataclasses
import json
import logging
import os

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from ..app import require_app, verify_api_key
from ..models import ChatRequest, ChatResponse
from onevalet.streaming.models import EventType

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
                # ed is an AgentResult dataclass, not a dict
                final_resp = getattr(ed, "raw_message", "") or ""
                tool_calls = getattr(ed, "metadata", {}).get("tool_calls", []) if hasattr(ed, "metadata") else []
                await _post_stream_result(
                    tenant_id=req.tenant_id,
                    final_response=final_resp,
                    tool_calls=tool_calls,
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
    """Basic health check."""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready():
    """Readiness check - verifies all components are available."""
    checks = {}

    try:
        app = require_app()
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "checks": {"app": "not_configured"}},
        )

    # Database check
    if app.database:
        try:
            await app.database.fetchval("SELECT 1")
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {e}"
    else:
        checks["database"] = "not_configured"

    # LLM check (just verify client exists, don't make API call)
    checks["llm"] = "ok" if app.orchestrator and app.orchestrator.llm_client else "not_configured"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )


@router.post("/api/clear-session", dependencies=[Depends(verify_api_key)])
async def clear_session(tenant_id: str = "default"):
    """Clear conversation history for a tenant."""
    app = require_app()
    await app.clear_session(tenant_id)
    return {"status": "ok", "message": "Session history cleared"}


@router.get("/api/actions", dependencies=[Depends(verify_api_key)])
async def get_actions(tenant_id: str, limit: int = 50, offset: int = 0):
    """Get paginated action history for a tenant."""
    app = require_app()
    db = app.database
    if not db:
        return {"actions": [], "total": 0, "has_more": False}

    try:
        count = await db.fetchval(
            "SELECT COUNT(*) FROM tool_call_history WHERE tenant_id = $1",
            tenant_id,
        )
        rows = await db.fetch(
            """
            SELECT id, tool_name, agent_name, summary, args_summary,
                   success, result_status, duration_ms, created_at
            FROM tool_call_history
            WHERE tenant_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            tenant_id, limit, offset,
        )
        actions = []
        for r in rows:
            actions.append({
                "id": str(r["id"]),
                "tool_name": r["tool_name"],
                "agent_name": r["agent_name"],
                "summary": r["summary"],
                "args_summary": r["args_summary"],
                "success": r["success"],
                "result_status": r["result_status"],
                "duration_ms": r["duration_ms"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            })
        return {
            "actions": actions,
            "total": count or 0,
            "has_more": (count or 0) > offset + limit,
        }
    except Exception:
        return {"actions": [], "total": 0, "has_more": False}
