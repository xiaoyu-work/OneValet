"""Tests for per-tenant deny in ToolPolicyFilter."""

import pytest

from onevalet.orchestrator.tool_policy import ToolPolicyFilter


def _make_schema(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {}},
    }


class TestTenantDeny:

    def test_tenant_deny_blocks_tool(self):
        policy = ToolPolicyFilter()
        policy.set_tenant_deny("t1", {"SmartHomeAgent"})

        schemas = [_make_schema("EmailAgent"), _make_schema("SmartHomeAgent")]
        filtered = policy.filter_tools(schemas, tenant_id="t1")

        names = [s["function"]["name"] for s in filtered]
        assert names == ["EmailAgent"]

    def test_tenant_deny_no_affect_other_tenant(self):
        policy = ToolPolicyFilter()
        policy.set_tenant_deny("t1", {"SmartHomeAgent"})

        schemas = [_make_schema("SmartHomeAgent")]
        # t2 should not be affected
        filtered = policy.filter_tools(schemas, tenant_id="t2")
        assert len(filtered) == 1

    def test_clear_tenant_deny(self):
        policy = ToolPolicyFilter()
        policy.set_tenant_deny("t1", {"EmailAgent"})
        policy.clear_tenant_deny("t1")

        schemas = [_make_schema("EmailAgent")]
        filtered = policy.filter_tools(schemas, tenant_id="t1")
        assert len(filtered) == 1

    def test_tenant_deny_combined_with_global(self):
        policy = ToolPolicyFilter()
        policy.set_global_deny({"GlobalBlocked"})
        policy.set_tenant_deny("t1", {"TenantBlocked"})

        schemas = [
            _make_schema("GlobalBlocked"),
            _make_schema("TenantBlocked"),
            _make_schema("Allowed"),
        ]
        filtered = policy.filter_tools(schemas, tenant_id="t1")
        names = [s["function"]["name"] for s in filtered]
        assert names == ["Allowed"]

    def test_no_tenant_id_skips_tenant_filter(self):
        policy = ToolPolicyFilter()
        policy.set_tenant_deny("t1", {"EmailAgent"})

        schemas = [_make_schema("EmailAgent")]
        # No tenant_id -> tenant deny not applied
        filtered = policy.filter_tools(schemas)
        assert len(filtered) == 1

    def test_is_tool_allowed_with_tenant(self):
        policy = ToolPolicyFilter()
        policy.set_tenant_deny("t1", {"SmartHomeAgent"})

        assert policy.is_tool_allowed("SmartHomeAgent", tenant_id="t1") is False
        assert policy.is_tool_allowed("SmartHomeAgent", tenant_id="t2") is True
        assert policy.is_tool_allowed("SmartHomeAgent") is True

    def test_get_filter_reason_tenant(self):
        policy = ToolPolicyFilter()
        policy.set_tenant_deny("t1", {"SmartHomeAgent"})

        reason = policy.get_filter_reason("SmartHomeAgent", tenant_id="t1")
        assert reason is not None
        assert "tenant" in reason
        assert "t1" in reason

        assert policy.get_filter_reason("SmartHomeAgent", tenant_id="t2") is None
