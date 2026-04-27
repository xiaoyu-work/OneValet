"""P1-8: TaskRegistry graceful shutdown cancels outstanding tasks."""

import asyncio

import pytest

from koa.observability.task_registry import TaskRegistry


@pytest.mark.asyncio
async def test_cancel_all_cancels_running_tasks():
    reg = TaskRegistry(name="test")

    started = asyncio.Event()

    async def long_running():
        started.set()
        await asyncio.sleep(30)

    reg.create_task(long_running(), name="long")
    await started.wait()
    # cancel_all blocks until tasks wind down.
    await reg.cancel_all(timeout=2.0)


@pytest.mark.asyncio
async def test_task_exceptions_are_logged_not_silent(caplog):
    reg = TaskRegistry(name="test")

    async def boom():
        raise RuntimeError("kaboom")

    reg.create_task(boom(), name="boom")
    await asyncio.sleep(0.05)
    # Should have recorded the exception via the done-callback.
    assert any("kaboom" in r.getMessage() or "boom" in r.getMessage() for r in caplog.records)
