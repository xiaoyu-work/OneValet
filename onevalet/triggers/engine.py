"""OneValet TriggerEngine — manages trigger tasks and dispatches execution."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ActionConfig,
    ActionResult,
    Task,
    TaskStatus,
    TriggerConfig,
    TriggerContext,
    TriggerType,
)

logger = logging.getLogger(__name__)


class TriggerEngine:
    """
    Proactive trigger system. Manages scheduled, event-driven, and condition-based tasks.

    Args:
        executors: Dict of executor_name -> executor instance (must have async execute(context) -> ActionResult)
        notifications: List of notification channel instances (must have async send(tenant_id, message, metadata))
        check_interval: Seconds between evaluation loop iterations (default 10)
    """

    def __init__(
        self,
        executors: Optional[Dict[str, Any]] = None,
        notifications: Optional[List[Any]] = None,
        check_interval: int = 10,
    ):
        self._executors: Dict[str, Any] = executors or {}
        self._notifications: List[Any] = notifications or []
        self._tasks: Dict[str, Task] = {}  # task_id -> Task
        self._check_interval = check_interval
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._cron_service: Optional[Any] = None
        # Trigger evaluators (lazy-imported)
        self._schedule_evaluator = None
        self._event_evaluator = None
        self._condition_evaluator = None

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    async def create_task(
        self,
        user_id: str,
        trigger: TriggerConfig,
        action: Optional[ActionConfig] = None,
        name: str = "",
        description: str = "",
        max_runs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create a new trigger task."""
        task = Task(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            description=description,
            trigger=trigger,
            action=action or ActionConfig(),
            max_runs=max_runs,
            metadata=metadata or {},
        )
        # Compute next_run_at for schedule triggers
        task.next_run_at = self._compute_next_run(task)
        self._tasks[task.id] = task
        logger.info(f"Created trigger task {task.id} ({task.trigger.type.value}) for user {user_id}")
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    async def list_tasks(
        self, user_id: Optional[str] = None, status: Optional[TaskStatus] = None
    ) -> List[Task]:
        tasks = list(self._tasks.values())
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    async def update_task_status(self, task_id: str, status: TaskStatus) -> Optional[Task]:
        task = self._tasks.get(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.now()
        return task

    async def delete_task(self, task_id: str) -> bool:
        return self._tasks.pop(task_id, None) is not None

    async def list_pending_approvals(self, user_id: str) -> List[Task]:
        """List all tasks in PENDING_APPROVAL state for a user."""
        return [
            t
            for t in self._tasks.values()
            if t.user_id == user_id and t.status == TaskStatus.PENDING_APPROVAL
        ]

    # ------------------------------------------------------------------
    # Executor registry
    # ------------------------------------------------------------------

    def register_executor(self, name: str, executor: Any) -> None:
        """Register a custom executor."""
        self._executors[name] = executor
        logger.info(f"Registered executor: {name}")

    def get_executor(self, name: str) -> Optional[Any]:
        return self._executors.get(name)

    def set_cron_service(self, cron_service: Any) -> None:
        """Attach a CronService for timer-based cron job scheduling."""
        self._cron_service = cron_service

    @property
    def cron_service(self) -> Optional[Any]:
        return self._cron_service

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the trigger evaluation loop and cron service."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._evaluation_loop())
        if self._cron_service:
            await self._cron_service.start()
        logger.info("TriggerEngine started")

    async def stop(self) -> None:
        """Stop the trigger evaluation loop and cron service."""
        self._running = False
        if self._cron_service:
            await self._cron_service.stop()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("TriggerEngine stopped")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    async def _evaluation_loop(self) -> None:
        """Main loop: periodically check all active tasks for triggers that should fire."""
        while self._running:
            try:
                await self._evaluate_all()
            except Exception as e:
                logger.error(f"Trigger evaluation error: {e}")
            await asyncio.sleep(self._check_interval)

    async def _evaluate_all(self) -> None:
        """Evaluate all active tasks."""
        now = datetime.now()
        for task in list(self._tasks.values()):
            if task.status != TaskStatus.ACTIVE:
                continue

            should_fire = False

            if task.trigger.type == TriggerType.SCHEDULE:
                should_fire = self._should_fire_schedule(task, now)
            elif task.trigger.type == TriggerType.CONDITION:
                should_fire = await self._should_fire_condition(task)
            # Event triggers are handled reactively via EventBus, not polling

            if should_fire:
                await self._fire_task(task, now)

    def _should_fire_schedule(self, task: Task, now: datetime) -> bool:
        """Check if a schedule trigger should fire."""
        if task.next_run_at and now >= task.next_run_at:
            return True
        return False

    async def _should_fire_condition(self, task: Task) -> bool:
        """Check if a condition trigger should fire (placeholder — override in subclass or extend)."""
        # Poll interval check
        poll_interval = task.trigger.params.get("poll_interval_minutes", 30)
        if task.last_run_at:
            elapsed = (datetime.now() - task.last_run_at).total_seconds() / 60
            if elapsed < poll_interval:
                return False
        # Condition evaluation is application-specific
        # Return False by default; condition evaluator can be plugged in
        if self._condition_evaluator:
            return await self._condition_evaluator(task)
        return False

    async def _fire_task(self, task: Task, now: datetime) -> None:
        """Execute a triggered task."""
        # Check max_runs
        if task.max_runs is not None and task.run_count >= task.max_runs:
            task.status = TaskStatus.COMPLETED
            task.updated_at = now
            logger.info(f"Task {task.id} completed (max_runs={task.max_runs} reached)")
            return

        context = TriggerContext(
            task=task,
            trigger_type=task.trigger.type.value,
            fired_at=now,
        )

        # Find executor
        executor = self._executors.get(task.action.executor)
        if not executor:
            logger.error(f"Executor '{task.action.executor}' not found for task {task.id}")
            return

        # Execute
        try:
            result = await executor.execute(context)
            task.run_count += 1
            task.last_run_at = now
            task.next_run_at = self._compute_next_run(task)
            task.updated_at = now

            # Handle approval pending
            if result.metadata.get("pending_approval"):
                task.status = TaskStatus.PENDING_APPROVAL
                task.metadata["pending_agent_ids"] = result.metadata.get("agent_ids", [])

            # Notify user
            if result.output:
                await self._notify(
                    task.user_id,
                    result.output,
                    {
                        "task_id": task.id,
                        "task_name": task.name,
                        "trigger_type": task.trigger.type.value,
                    },
                )

            logger.info(f"Task {task.id} fired successfully (run {task.run_count})")

        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")

    async def _notify(self, user_id: str, message: str, metadata: Dict[str, Any]) -> None:
        """Send notification through all configured channels."""
        for channel in self._notifications:
            try:
                await channel.send(user_id, message, metadata)
            except Exception as e:
                logger.warning(f"Notification failed via {type(channel).__name__}: {e}")

    def _compute_next_run(self, task: Task) -> Optional[datetime]:
        """Compute next run time for schedule triggers."""
        if task.trigger.type != TriggerType.SCHEDULE:
            return None

        params = task.trigger.params
        now = datetime.now()

        # One-time: run_at
        if "run_at" in params:
            run_at = params["run_at"]
            if isinstance(run_at, str):
                run_at = datetime.fromisoformat(run_at)
            return run_at if run_at > now else None

        # Interval: interval_minutes
        if "interval_minutes" in params:
            base = task.last_run_at or now
            return base + timedelta(minutes=params["interval_minutes"])

        # Cron: cron expression
        if "cron" in params:
            try:
                from croniter import croniter

                cron = croniter(params["cron"], now)
                return cron.get_next(datetime)
            except ImportError:
                logger.warning("croniter not installed, cron triggers disabled")
                return None
            except Exception as e:
                logger.warning(f"Invalid cron expression '{params['cron']}': {e}")
                return None

        return None

    # ------------------------------------------------------------------
    # TTL cleanup for pending approvals
    # ------------------------------------------------------------------

    async def cleanup_expired_approvals(self, pool_manager: Any) -> int:
        """Scan PENDING_APPROVAL tasks, expire those whose agents were TTL-removed from Pool."""
        count = 0
        for task in list(self._tasks.values()):
            if task.status != TaskStatus.PENDING_APPROVAL:
                continue
            agent_ids = task.metadata.get("pending_agent_ids", [])
            if not agent_ids:
                continue
            # Check if any agent still exists in pool
            all_expired = True
            for aid in agent_ids:
                agent = await pool_manager.get_agent(task.user_id, aid)
                if agent is not None:
                    all_expired = False
                    break
            if all_expired:
                task.status = TaskStatus.EXPIRED
                task.updated_at = datetime.now()
                count += 1
                logger.info(f"Task {task.id} expired (agents TTL-removed from pool)")
        return count
