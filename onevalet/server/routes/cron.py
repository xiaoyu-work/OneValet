"""Cron job CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..app import require_app, verify_api_key
from ..models import CronJobCreateRequest, CronJobUpdateRequest

router = APIRouter()


def _require_cron_service(app):
    if not app.cron_service:
        raise HTTPException(503, "CronService not available")
    return app.cron_service


@router.get("/api/cron/status", dependencies=[Depends(verify_api_key)])
async def cron_status():
    """Get cron scheduler status."""
    app = require_app()
    return await app.cron_status()


@router.get("/api/cron/jobs", dependencies=[Depends(verify_api_key)])
async def list_cron_jobs(
    tenant_id: str = "default",
    include_disabled: bool = False,
):
    """List cron jobs for a tenant."""
    app = require_app()
    service = _require_cron_service(app)
    jobs = service.list_jobs(user_id=tenant_id, include_disabled=include_disabled)
    return [j.to_dict() for j in jobs]


@router.post("/api/cron/jobs", dependencies=[Depends(verify_api_key)])
async def create_cron_job(req: CronJobCreateRequest):
    """Create a new cron job."""
    from ...triggers.cron.models import (
        AtSchedule,
        EverySchedule,
        CronScheduleSpec,
        SessionTarget,
        WakeMode,
        SystemEventPayload,
        AgentTurnPayload,
        DeliveryMode,
        DeliveryConfig,
        CronJobCreate,
    )

    app = require_app()
    service = _require_cron_service(app)

    # Build schedule
    if req.schedule_type == "at":
        schedule = AtSchedule(at=req.schedule_value)
    elif req.schedule_type == "every":
        try:
            schedule = EverySchedule(every_ms=int(float(req.schedule_value) * 1000))
        except ValueError:
            raise HTTPException(400, f"Invalid interval: {req.schedule_value}")
    elif req.schedule_type == "cron":
        schedule = CronScheduleSpec(expr=req.schedule_value, tz=req.timezone or None)
    else:
        raise HTTPException(400, f"Unknown schedule_type: {req.schedule_type}")

    # Build payload
    target = SessionTarget(req.session_target)
    if target == SessionTarget.MAIN:
        payload = SystemEventPayload(text=req.instruction)
    else:
        payload = AgentTurnPayload(message=req.instruction)

    # Build delivery
    delivery = None
    if req.delivery_mode != "none" or req.conditional:
        mode = DeliveryMode(req.delivery_mode) if req.delivery_mode != "none" else DeliveryMode.ANNOUNCE
        delivery = DeliveryConfig(
            mode=mode,
            channel=req.delivery_channel,
            webhook_url=req.webhook_url,
            conditional=req.conditional,
        )

    input_data = CronJobCreate(
        name=req.name,
        user_id=req.tenant_id,
        schedule=schedule,
        session_target=target,
        wake_mode=WakeMode(req.wake_mode),
        payload=payload,
        delivery=delivery,
        delete_after_run=req.delete_after_run,
    )

    job = await service.add(input_data)
    return job.to_dict()


@router.get("/api/cron/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_cron_job(job_id: str):
    """Get a cron job by ID."""
    app = require_app()
    service = _require_cron_service(app)
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Cron job not found")
    return job.to_dict()


@router.put("/api/cron/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def update_cron_job(job_id: str, req: CronJobUpdateRequest):
    """Update a cron job."""
    from ...triggers.cron.models import (
        AtSchedule,
        EverySchedule,
        CronScheduleSpec,
        SystemEventPayload,
        AgentTurnPayload,
        CronJobPatch,
    )

    app = require_app()
    service = _require_cron_service(app)

    job = service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Cron job not found")

    patch = CronJobPatch()

    if req.enabled is not None:
        patch.enabled = req.enabled
    if req.name is not None:
        patch.name = req.name

    if req.schedule_type and req.schedule_value:
        if req.schedule_type == "at":
            patch.schedule = AtSchedule(at=req.schedule_value)
        elif req.schedule_type == "every":
            patch.schedule = EverySchedule(every_ms=int(float(req.schedule_value) * 1000))
        elif req.schedule_type == "cron":
            patch.schedule = CronScheduleSpec(expr=req.schedule_value, tz=req.timezone)

    if req.instruction is not None:
        if isinstance(job.payload, SystemEventPayload):
            patch.payload = SystemEventPayload(text=req.instruction)
        else:
            patch.payload = AgentTurnPayload(message=req.instruction)

    try:
        updated = await service.update(job_id, patch)
        return updated.to_dict()
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.delete("/api/cron/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def delete_cron_job(job_id: str):
    """Delete a cron job."""
    app = require_app()
    service = _require_cron_service(app)
    removed = await service.remove(job_id)
    if not removed:
        raise HTTPException(404, "Cron job not found")
    return {"deleted": True}


@router.post("/api/cron/jobs/{job_id}/run", dependencies=[Depends(verify_api_key)])
async def run_cron_job(job_id: str, mode: str = "force"):
    """Manually trigger a cron job."""
    app = require_app()
    service = _require_cron_service(app)
    try:
        entry = await service.run(job_id, mode=mode)
        return entry.to_dict()
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/api/cron/jobs/{job_id}/runs", dependencies=[Depends(verify_api_key)])
async def get_cron_job_runs(job_id: str, limit: int = 20):
    """Get run history for a cron job."""
    app = require_app()
    service = _require_cron_service(app)
    runs = await service.get_runs(job_id, limit=limit)
    return [r.to_dict() for r in runs]
