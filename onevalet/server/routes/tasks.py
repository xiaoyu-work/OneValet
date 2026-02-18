"""Trigger task CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..app import require_app, verify_api_key
from ..models import TaskCreateRequest, TaskUpdateRequest

router = APIRouter()


@router.get("/api/tasks", dependencies=[Depends(verify_api_key)])
async def list_tasks(tenant_id: str = "default"):
    """List trigger tasks for a tenant."""
    app = require_app()
    tasks = await app.list_tasks(tenant_id)
    if not tasks and not app.trigger_engine:
        raise HTTPException(503, "TriggerEngine not available")
    return [t.to_dict() for t in tasks]


@router.post("/api/tasks", dependencies=[Depends(verify_api_key)])
async def create_task(req: TaskCreateRequest):
    """Create a new trigger task."""
    from ...triggers import TriggerConfig, TriggerType, ActionConfig

    app = require_app()
    if not app.trigger_engine:
        raise HTTPException(503, "TriggerEngine not available")

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
    if not app.trigger_engine:
        raise HTTPException(503, "TriggerEngine not available")

    if req.status:
        task = await app.update_task(task_id, TaskStatus(req.status))
        if not task:
            raise HTTPException(404, "Task not found")
        return task.to_dict()
    raise HTTPException(400, "No updates specified")


@router.delete("/api/tasks/{task_id}", dependencies=[Depends(verify_api_key)])
async def delete_task(task_id: str):
    """Delete a trigger task."""
    app = require_app()
    try:
        deleted = await app.delete_task(task_id)
    except RuntimeError:
        raise HTTPException(503, "TriggerEngine not available")
    if not deleted:
        raise HTTPException(404, "Task not found")
    return {"deleted": True}
