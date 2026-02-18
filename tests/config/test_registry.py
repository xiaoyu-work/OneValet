"""Tests for tenant-aware credential filtering in AgentRegistry."""

import pytest
from unittest.mock import AsyncMock

from onevalet.agents.decorator import AGENT_REGISTRY, valet
from onevalet.config.registry import AgentRegistry


# ── Fixtures ──


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save/restore the global AGENT_REGISTRY around each test."""
    original = dict(AGENT_REGISTRY)
    yield
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(original)


def _make_credential_store(services: list[str]) -> AsyncMock:
    """Create a mock CredentialStore that returns the given services."""
    store = AsyncMock()
    store.list = AsyncMock(return_value=[
        {"service": s, "account_name": "primary", "credentials": {}}
        for s in services
    ])
    return store


def _register_test_agents():
    """Register a few test agents with different requires_service."""

    @valet(requires_service=["gmail", "outlook"])
    class _EmailAgent:
        """Email agent"""

    @valet(requires_service=["philips_hue", "sonos"])
    class _SmartHomeAgent:
        """Smart home agent"""

    @valet()
    class _MapsAgent:
        """Maps agent (no requires_service)"""

    return _EmailAgent, _SmartHomeAgent, _MapsAgent


# ── Tests ──


class TestCredentialFiltering:

    @pytest.mark.asyncio
    async def test_filters_by_credential(self):
        _register_test_agents()
        registry = AgentRegistry()
        store = _make_credential_store(["gmail"])  # only gmail

        schemas = await registry.get_all_agent_tool_schemas(
            tenant_id="t1", credential_store=store,
        )
        names = [s["function"]["name"] for s in schemas]

        assert "_EmailAgent" in names       # gmail matches
        assert "_SmartHomeAgent" not in names  # no hue/sonos
        assert "_MapsAgent" in names         # no requires_service

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Tenant has gmail but not outlook — EmailAgent still available."""
        _register_test_agents()
        registry = AgentRegistry()
        store = _make_credential_store(["gmail"])

        schemas = await registry.get_all_agent_tool_schemas(
            tenant_id="t1", credential_store=store,
        )
        names = [s["function"]["name"] for s in schemas]
        assert "_EmailAgent" in names  # gmail is sufficient (any-of)

    @pytest.mark.asyncio
    async def test_no_tenant_returns_all(self):
        _register_test_agents()
        registry = AgentRegistry()

        schemas = await registry.get_all_agent_tool_schemas()
        names = [s["function"]["name"] for s in schemas]

        assert "_EmailAgent" in names
        assert "_SmartHomeAgent" in names
        assert "_MapsAgent" in names

    @pytest.mark.asyncio
    async def test_agent_without_requires_service_always_included(self):
        _register_test_agents()
        registry = AgentRegistry()
        store = _make_credential_store([])  # no services at all

        schemas = await registry.get_all_agent_tool_schemas(
            tenant_id="t1", credential_store=store,
        )
        names = [s["function"]["name"] for s in schemas]

        assert "_MapsAgent" in names         # always available
        assert "_EmailAgent" not in names    # no gmail/outlook
        assert "_SmartHomeAgent" not in names  # no hue/sonos

    @pytest.mark.asyncio
    async def test_get_agent_descriptions_also_filters(self):
        _register_test_agents()
        registry = AgentRegistry()
        store = _make_credential_store(["gmail"])

        desc = await registry.get_agent_descriptions(
            tenant_id="t1", credential_store=store,
        )

        assert "_EmailAgent" in desc
        assert "_SmartHomeAgent" not in desc
        assert "_MapsAgent" in desc
