from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from koa.models import AgentToolContext
from koa.providers.local_backend import LocalBackendClient
from koa.tool_decorator import tool


@dataclass(frozen=True)
class ResolvedSurfaceTarget:
    surface: str
    provider: str
    account: Optional[str]
    source: Literal["explicit", "saved", "default"]


async def resolve_surface_target(
    tenant_id: str,
    surface: str,
    backend_client: LocalBackendClient,
    explicit_provider: str | None = None,
    explicit_account: str | None = None,
) -> ResolvedSurfaceTarget:
    if explicit_provider:
        return ResolvedSurfaceTarget(
            surface=surface,
            provider=explicit_provider,
            account=explicit_account,
            source="explicit",
        )

    preference = await backend_client.get_routing_preference(tenant_id, surface)
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
    client = LocalBackendClient.from_context(context)
    saved = await client.set_routing_preference(
        context.tenant_id,
        surface,
        provider,
        account,
    )

    provider_name = saved.get("default_provider", provider)
    return f"Okay — I'll use {provider_name} by default for {surface}."
