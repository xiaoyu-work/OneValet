"""P1-10: Approval store TTL + batch ordering."""

import pytest

from koa.orchestrator.approval import (
    ApprovalRequest,
    InMemoryApprovalStore,
    collect_batch_approvals,
)


@pytest.mark.asyncio
async def test_approval_expires_after_timeout():
    store = InMemoryApprovalStore()
    req = ApprovalRequest(agent_name="A", action_summary="go", timeout_minutes=1)
    # Monkey-patch deadline to the past.
    await store.put(req)
    store._entries[req.request_id].deadline = 0.0
    assert await store.get(req.request_id) is None


@pytest.mark.asyncio
async def test_expire_overdue_returns_expired():
    store = InMemoryApprovalStore()
    req = ApprovalRequest(agent_name="A", action_summary="x")
    await store.put(req)
    store._entries[req.request_id].deadline = 0.0
    expired = await store.expire_overdue()
    assert len(expired) == 1
    assert expired[0].request_id == req.request_id


def test_batch_orders_destructive_first():
    a = ApprovalRequest(agent_name="A", action_summary="read", risk_level="read")
    b = ApprovalRequest(agent_name="B", action_summary="write", risk_level="write")
    c = ApprovalRequest(agent_name="C", action_summary="delete", risk_level="destructive")
    ordered = collect_batch_approvals([a, b, c])
    assert [r.risk_level for r in ordered] == ["destructive", "write", "read"]
