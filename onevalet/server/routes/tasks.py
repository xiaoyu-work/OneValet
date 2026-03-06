"""Trigger task CRUD routes."""

from fastapi import APIRouter, Depends

from ...errors import OneValetError, E
from ..app import require_app, verify_api_key
from ..models import TaskCreateRequest, TaskUpdateRequest

router = APIRouter()


def _require_trigger_engine(app):
    if not app.trigger_engine:
        raise OneValetError(E.SERVICE_UNAVAILABLE, "TriggerEngine not available",
                            details={"service": "triggers"})
    return app.trigger_engine


@router.get("/api/tasks", dependencies=[Depends(verify_api_key)])
async def list_tasks(tenant_id: str = "default"):
    """List trigger tasks for a tenant."""
    app = require_app()
    tasks = await app.list_tasks(tenant_id)
    if not tasks and not app.trigger_engine:
        raise OneValetError(E.SERVICE_UNAVAILABLE, "TriggerEngine not available",
                            details={"service": "triggers"})
    return [t.to_dict() for t in tasks]


@router.post("/api/tasks", dependencies=[Depends(verify_api_key)])
async def create_task(req: TaskCreateRequest):
    """Create a new trigger task."""
    from ...triggers import TriggerConfig, TriggerType, ActionConfig

    app = require_app()
    _require_trigger_engine(app)

    trigger = TriggerConfig(
        type=TriggerType(req.trigger_type),
        params=req.trigger_params,
    )
    action = ActionConfig(
        executor=req.executor,
        instruction=req.instruction,
        config=req.action_config or {},
    )
    task = await app.create_task(
        user_id=req.tenant_id,
        trigger=trigger,
        action=action,
        name=req.name,
        description=req.description,
        max_runs=req.max_runs,
        metadata=req.metadata,
    )
    return task.to_dict()


@router.put("/api/tasks/{task_id}", dependencies=[Depends(verify_api_key)])
async def update_task(task_id: str, req: TaskUpdateRequest):
    """Update a trigger task status."""
    from ...triggers import TaskStatus

    app = require_app()
    _require_trigger_engine(app)

    if not req.status:
        raise OneValetError(E.VALIDATION_ERROR, "No updates specified")

    task = await app.update_task(task_id, TaskStatus(req.status))
    if not task:
        raise OneValetError(E.NOT_FOUND, "Task not found", details={"resource": "task"})
    return task.to_dict()


@router.delete("/api/tasks/{task_id}", dependencies=[Depends(verify_api_key)])
async def delete_task(task_id: str):
    """Delete a trigger task."""
    app = require_app()
    try:
        deleted = await app.delete_task(task_id)
    except RuntimeError:
        raise OneValetError(E.SERVICE_UNAVAILABLE, "TriggerEngine not available",
                            details={"service": "triggers"})
    if not deleted:
        raise OneValetError(E.NOT_FOUND, "Task not found", details={"resource": "task"})
    return {"deleted": True}
