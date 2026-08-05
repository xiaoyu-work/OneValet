"""Inbox routes — what the assistant is waiting on, and answering it."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...errors import E, KoaError
from ..app import require_app, verify_api_key

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
                "expires_at": a.expires_at.isoformat() if a.expires_at else None,
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

    # Continue in the background: the caller should not wait out a full agent
    # run to have their tap acknowledged. The run may still be finishing, so
    # this waits for it rather than giving up on the first refusal.
    scheduled = orch.schedule_resume(run_id)
    return {
        "status": "resolved",
        "continuation": "scheduled" if scheduled else "deferred",
        "run_id": run_id,
    }


@router.get("/api/runs/{tenant_id}/resumable", dependencies=[Depends(verify_api_key)])
async def list_resumable(tenant_id: str, limit: int = 20):
    """Runs that stopped before finishing -- waiting on someone, or interrupted."""
    app = require_app()
    await app._ensure_initialized()

    orch = app.orchestrator
    if orch is None:
        return {"runs": []}
    return {"runs": await orch.list_resumable_runs(tenant_id, limit=limit)}


@router.post("/api/runs/{run_id}/resume", dependencies=[Depends(verify_api_key)])
async def resume(run_id: str):
    """Continue a run that stopped before finishing.

    Answering an ask normally wakes its run on its own. This is for the runs
    that answering cannot reach: one whose decision was recorded but never
    acted on has no open question left to answer, so without this it would
    stay listed as resumable and never move.
    """
    app = require_app()
    await app._ensure_initialized()

    orch = app.orchestrator
    if orch is None:
        raise KoaError(
            E.SERVICE_UNAVAILABLE, "Orchestrator not available", details={"run_id": run_id}
        )

    scheduled = orch.schedule_resume(run_id)
    return {
        "status": "resuming" if scheduled else "deferred",
        "run_id": run_id,
    }
