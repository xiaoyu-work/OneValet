from unittest.mock import AsyncMock

import pytest

from koa.builtin_agents.shared.routing_preferences import (
    ResolvedSurfaceTarget,
    resolve_surface_target,
    set_routing_preference,
    wrap_routing_error,
)
from koa.models import AgentToolContext


def _make_db(preference_row=None):
    db = AsyncMock()
    db.fetchrow.return_value = preference_row
    return db


class TestResolveSurfaceTarget:
    @pytest.mark.asyncio
    async def test_prefers_explicit_over_saved_preference(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="calendar",
            db=_make_db({"default_provider": "local", "default_account": None}),
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
            db=_make_db({"default_provider": "google", "default_account": "work"}),
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
            db=_make_db({"default_provider": "google", "default_account": "primary"}),
            explicit_account="work",
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="todo",
            provider="google",
            account="work",
            source="explicit",
        )

    @pytest.mark.asyncio
    async def test_explicit_account_no_provider_no_preference_falls_back_to_local(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="calendar",
            db=_make_db(None),
            explicit_account="work",
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="calendar",
            provider="local",
            account="work",
            source="explicit",
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_local_default_when_no_preference_saved(self):
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="reminder",
            db=_make_db(None),
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="reminder",
            provider="local",
            account=None,
            source="default",
        )

    @pytest.mark.asyncio
    async def test_no_db_falls_back_to_local_default(self):
        """If db is unavailable, resolver shouldn't crash — fall back to local."""
        resolved = await resolve_surface_target(
            tenant_id="user-1",
            surface="calendar",
            db=None,
        )

        assert resolved == ResolvedSurfaceTarget(
            surface="calendar",
            provider="local",
            account=None,
            source="default",
        )


class TestSetRoutingPreferenceTool:
    @pytest.mark.asyncio
    async def test_upserts_preference_via_db_from_context_hints(self):
        db = AsyncMock()
        db.fetchrow.return_value = {
            "default_provider": "google",
            "default_account": "primary",
        }
        ctx = AgentToolContext(
            tenant_id="user-1",
            context_hints={"db": db},
        )

        result = await set_routing_preference.executor(
            {
                "surface": "calendar",
                "provider": "google",
                "account": "primary",
            },
            ctx,
        )

        db.fetchrow.assert_awaited_once()
        call_args = db.fetchrow.call_args
        # positional args: sql, tenant_id, surface, provider, account
        assert call_args.args[1:] == ("user-1", "calendar", "google", "primary")
        assert result == "Okay — I'll use google by default for calendar."

    @pytest.mark.asyncio
    async def test_returns_friendly_error_when_db_missing(self):
        ctx = AgentToolContext(tenant_id="user-1", context_hints={})
        result = await set_routing_preference.executor(
            {"surface": "calendar", "provider": "google"},
            ctx,
        )
        assert "couldn't save" in result.lower()


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

    def test_unsupported_provider_error_suggests_supported_alternative(self):
        result = wrap_routing_error("calendar", "myspace", "unsupported_provider")
        assert "don't support myspace" in result.lower()
        assert "use local instead" in result.lower()

    def test_default_error_offers_local_fallback(self):
        result = wrap_routing_error("reminder", "local", "write_failed")
        assert "couldn't finish that reminder action" in result.lower()
        assert "save it locally" in result.lower()

    def test_read_failed_error_does_not_suggest_save_locally(self):
        result = wrap_routing_error("calendar", "google", "read_failed")
        assert "couldn't retrieve your calendar data" in result.lower()
        assert "save it locally" not in result.lower()
        assert "try again" in result.lower()
