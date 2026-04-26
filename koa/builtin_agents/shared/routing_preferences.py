"""Routing-preferences resolver and set_routing_preference tool.

The resolver decides which provider (local, google, ...) and account
should handle a calendar/todo request when the user hasn't named one
explicitly. It reads from koa's own ``user_routing_preferences`` table
via the agent's db handle (``context.context_hints["db"]``), so the AI
engine no longer reverse-calls into the app gateway.

If no preference is set, the resolver falls back to ``provider="local"``
— the AI agent's data lives in koa's ``local_calendar_events`` table
(populated by CalendarSyncService and direct user creates), so this
default lets the agent answer "what's on my calendar" without any
external integrations needing to be wired up.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

from koa.models import AgentToolContext
from koa.tool_decorator import tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSurfaceTarget:
    surface: str
    provider: str
    account: Optional[str]
    source: Literal["explicit", "saved", "default"]


def _db_from_context(context: AgentToolContext) -> Any:
    """Return the koa Postgres handle out of the standard context_hints slot.

    Keeps the resolver decoupled from the wider Koa app object — tests can
    just pass a mock with `.fetchrow(...)` / `.execute(...)`.
    """
    if context.context_hints is None:
        return None
    return context.context_hints.get("db")


async def _fetch_preference(db: Any, tenant_id: str, surface: str) -> Optional[dict]:
    if db is None:
        return None
    try:
        row = await db.fetchrow(
            "SELECT default_provider, default_account "
            "FROM tenant_default.user_routing_preferences "
            "WHERE tenant_id = $1 AND surface = $2",
            tenant_id,
            surface,
        )
    except Exception as e:
        logger.warning(
            f"routing_preferences: failed to read preference for "
            f"tenant={tenant_id} surface={surface}: {e}"
        )
        return None
    if row is None:
        return None
    return {
        "default_provider": row["default_provider"],
        "default_account": row.get("default_account"),
    }


async def _upsert_preference(
    db: Any,
    tenant_id: str,
    surface: str,
    provider: str,
    account: Optional[str],
) -> dict:
    row = await db.fetchrow(
        """
        INSERT INTO tenant_default.user_routing_preferences
            (tenant_id, surface, default_provider, default_account, created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW())
        ON CONFLICT (tenant_id, surface) DO UPDATE SET
            default_provider = EXCLUDED.default_provider,
            default_account = EXCLUDED.default_account,
            updated_at = NOW()
        RETURNING default_provider, default_account
        """,
        tenant_id,
        surface,
        provider,
        account,
    )
    return {
        "default_provider": row["default_provider"],
        "default_account": row.get("default_account"),
    }


async def resolve_surface_target(
    tenant_id: str,
    surface: str,
    db: Any,
    explicit_provider: str | None = None,
    explicit_account: str | None = None,
) -> ResolvedSurfaceTarget:
    """Pick the provider+account for a request on (tenant, surface)."""
    if explicit_provider:
        return ResolvedSurfaceTarget(
            surface=surface,
            provider=explicit_provider,
            account=explicit_account,
            source="explicit",
        )

    preference = await _fetch_preference(db, tenant_id, surface)

    if explicit_account:
        return ResolvedSurfaceTarget(
            surface=surface,
            provider=(preference or {}).get("default_provider", "local"),
            account=explicit_account,
            source="explicit",
        )

    if preference:
        return ResolvedSurfaceTarget(
            surface=surface,
            provider=preference["default_provider"],
            account=preference.get("default_account"),
            source="saved",
        )

    return ResolvedSurfaceTarget(
        surface=surface,
        provider="local",
        account=None,
        source="default",
    )


def wrap_routing_error(surface: str, provider: str, reason: str) -> str:
    if reason == "not_connected":
        return (
            f"I couldn't use {provider} for this {surface} because it isn't connected. "
            f"Please connect {provider} in settings, or tell me to save it locally."
        )
    if reason == "auth_expired":
        return (
            f"I couldn't use {provider} for this {surface} because the connection expired. "
            f"Please reconnect it in settings and try again."
        )
    if reason == "unsupported_provider":
        return (
            f"I don't support {provider} for {surface} yet. "
            f"Tell me to use local instead, or connect a supported account in Settings."
        )
    if reason == "read_failed":
        return (
            f"I couldn't retrieve your {surface} data right now. "
            f"Please try again in a moment."
        )
    return (
        f"I couldn't finish that {surface} action right now. "
        f"Please try again, or tell me to save it locally."
    )


@tool(category="productivity", risk_level="write")
async def set_routing_preference(
    surface: str,
    provider: str,
    account: str | None = None,
    *,
    context: AgentToolContext,
) -> str:
    """Save the default destination for future calendar, todo, or reminder requests."""
    db = _db_from_context(context)
    if db is None:
        return (
            "I couldn't save that preference right now — the storage backend isn't "
            "available. Please try again."
        )

    saved = await _upsert_preference(
        db,
        context.tenant_id,
        surface,
        provider,
        account,
    )
    provider_name = saved.get("default_provider", provider)
    return f"Okay — I'll use {provider_name} by default for {surface}."
