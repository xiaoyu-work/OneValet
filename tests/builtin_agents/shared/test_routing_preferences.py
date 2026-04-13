from unittest.mock import AsyncMock, patch

import pytest

from koa.models import AgentToolContext
from koa.builtin_agents.shared.routing_preferences import (
    ResolvedSurfaceTarget,
    resolve_surface_target,
    set_routing_preference,
    wrap_routing_error,
)


class DummyClient:
    def __init__(self, preference=None):
        self.preference = preference

    async def get_routing_preference(self, tenant_id, surface):
        return self.preference


class TestResolveSurfaceTarget:
    @pytest.mark.asyncio
    async def test_prefers_explicit_over_saved_preference(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="calendar",
            backend_client=DummyClient(
                {
                    "default_provider": "local",
                    "default_account": None,
                }
            ),
            explicit_provider="google",
            explicit_account="primary",
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="calendar",
            provider="google",
            account="primary",
            source="explicit",
        )

    @pytest.mark.asyncio
    async def test_uses_saved_preference_when_no_explicit_target(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="todo",
            backend_client=DummyClient(
                {
                    "default_provider": "google",
                    "default_account": "work",
                }
            ),
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="todo",
            provider="google",
            account="work",
            source="saved",
        )

    @pytest.mark.asyncio
    async def test_explicit_account_overrides_saved_account_on_saved_provider(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="todo",
            backend_client=DummyClient(
                {
                    "default_provider": "google",
                    "default_account": "primary",
                }
            ),
            explicit_account="work",
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="todo",
            provider="google",
            account="work",
            source="explicit",
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_local_default_when_no_preference_saved(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="reminder",
            backend_client=DummyClient(None),
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="reminder",
            provider="local",
            account=None,
            source="default",
        )


class TestSetRoutingPreferenceTool:
    @pytest.mark.asyncio
    async def test_calls_backend_client_from_context(self):
        ctx = AgentToolContext(
            tenant_id="user-1",
            metadata={"koiai_url": "https://koiai.example", "service_key": "svc-key"},
        )
        backend_client = AsyncMock()
        backend_client.set_routing_preference.return_value = {
            "surface": "calendar",
            "default_provider": "google",
            "default_account": "primary",
        }

        with patch(
            "koa.builtin_agents.shared.routing_preferences.LocalBackendClient.from_context",
            return_value=backend_client,
        ):
            result = await set_routing_preference.executor(
                {
                    "surface": "calendar",
                    "provider": "google",
                    "account": "primary",
                },
                ctx,
            )

        backend_client.set_routing_preference.assert_awaited_once_with(
            "user-1",
            "calendar",
            "google",
            "primary",
        )
        assert result == "Okay — I'll use google by default for calendar."


class TestWrapRoutingError:
    def test_not_connected_error_is_actionable(self):
        result = wrap_routing_error("calendar", "google", "not_connected")
        assert "couldn't use google" in result.lower()
        assert "connect google in settings" in result.lower()
        assert "save it locally" in result.lower()

    def test_auth_expired_error_prompts_reconnect(self):
        result = wrap_routing_error("todo", "google", "auth_expired")
        assert "connection expired" in result.lower()
        assert "reconnect it in settings" in result.lower()

    def test_default_error_offers_local_fallback(self):
        result = wrap_routing_error("reminder", "local", "write_failed")
        assert "couldn't finish that reminder action" in result.lower()
        assert "save it locally" in result.lower()
