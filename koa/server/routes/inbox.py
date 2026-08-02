"""Inbox routes — what the assistant is waiting on, and answering it."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...errors import E, KoaError
from ..app import require_app, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class AnswerRequest(BaseModel):
    resolution: str
    resolved_by: str = ""


@router.get("/api/inbox/{tenant_id}", dependencies=[Depends(verify_api_key)])
async def list_pending(tenant_id: str, limit: int = 50):
    """Everything still waiting on this person."""
    app = require_app()
    await app._ensure_initialized()

    orch = app.orchestrator
    if orch is None or not orch.inbox.enabled:
        return {"asks": []}

    asks = await orch.inbox.pending(tenant_id, limit=limit)
    return {
        "asks": [
            {
                "id": a.id,
                "kind": a.kind,
                "title": a.title,
                "body": a.body,
                "options": a.options,
                "run_id": a.run_id,
            }
            for a in asks
        ]
    }


@router.post("/api/inbox/{ask_id}/answer", dependencies=[Depends(verify_api_key)])
async def answer(ask_id: str, req: AnswerRequest):
    """Answer an ask and let the run it blocked continue.

    Answering twice is not an error -- the first answer wins and later ones are
    reported as already resolved, so a user tapping approve on two devices sees
    a sensible result on both rather than the action happening twice.
    """
    app = require_app()
    await app._ensure_initialized()

    orch = app.orchestrator
    if orch is None or not orch.inbox.enabled:
        raise KoaError(E.SERVICE_UNAVAILABLE, "Inbox not available", details={"service": "inbox"})

    try:
        run_id = await orch.answer_ask(ask_id, req.resolution, resolved_by=req.resolved_by)
    except ValueError as e:
        raise KoaError(E.VALIDATION_ERROR, str(e), details={"ask_id": ask_id}) from e
    if run_id is None:
        return {"status": "already_resolved", "resumed": False}

    # Resume in the background: the caller should not wait out a full agent
    # run just to have their tap acknowledged.
    async def _resume():
        try:
            async for _ in orch.resume_run(run_id):
                pass
        except Exception as e:
            logger.error(f"[Inbox] Resuming run {run_id} failed: {e}", exc_info=True)

    orch.task_registry.create_task(_resume(), name=f"resume:{run_id}")
    return {"status": "resolved", "resumed": True, "run_id": run_id}


@router.get("/api/runs/{tenant_id}/resumable", dependencies=[Depends(verify_api_key)])
async def list_resumable(tenant_id: str, limit: int = 20):
    """Runs that stopped before finishing -- waiting on someone, or interrupted."""
    app = require_app()
    await app._ensure_initialized()

    orch = app.orchestrator
    if orch is None:
        return {"runs": []}
    return {"runs": await orch.list_resumable_runs(tenant_id, limit=limit)}
