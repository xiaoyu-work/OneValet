from types import SimpleNamespace

import pytest

from koa.orchestrator.agent_tool import _has_tier_access, execute_agent_tool


def test_has_tier_access_defaults_missing_permissions_to_free():
    assert _has_tier_access({}, "free") is True
    assert _has_tier_access({}, "pro") is False
    assert _has_tier_access({"user_tier": "pro"}, "pro") is True


@pytest.mark.asyncio
async def test_execute_agent_tool_blocks_required_pro_agent_for_free_user():
    metadata = SimpleNamespace(extra={"required_tier": "pro"})

    class Registry:
        def get_agent_metadata(self, agent_type):
            assert agent_type == "ShippingAgent"
            return metadata

    orchestrator = SimpleNamespace(
        _agent_registry=Registry(),
        database=None,
        trigger_engine=None,
    )

    result = await execute_agent_tool(
        orchestrator,
        agent_type="ShippingAgent",
        tenant_id="tenant-123",
        tool_call_args={},
        task_instruction="Track my package",
        request_context={"metadata": {"permissions": {"user_tier": "free"}}},
    )

    assert result.completed is True
    assert "Pro feature" in result.result_text
    assert result.metadata["error"]["code"] == "tier_upgrade_required"
