"""Credential management routes (public and internal)."""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..app import require_app, sanitize_credential, verify_api_key, verify_service_key
from ..models import CredentialSaveRequest

router = APIRouter()

# Issue #4: Known service allowlist for credential validation
_KNOWN_SERVICES = {
    "gmail", "google_calendar", "google_tasks", "google_drive",
    "outlook", "outlook_calendar", "microsoft_todo", "onedrive",
    "todoist", "notion", "philips_hue", "sonos", "dropbox",
    "amadeus", "weather_api", "google_api", "google_oauth_app",
    "microsoft_oauth_app", "composio", "todoist_oauth_app",
    "hue_oauth_app", "sonos_oauth_app", "dropbox_oauth_app",
}

# Valid account_name: alphanumeric, underscores, hyphens, 1-64 chars
_ACCOUNT_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_service(service: str) -> None:
    """Validate that a service name is in the allowlist."""
    if service not in _KNOWN_SERVICES:
        raise HTTPException(
            400,
            f"Unknown service: {service}. "
            f"Allowed services: {', '.join(sorted(_KNOWN_SERVICES))}",
        )


def _validate_account_name(account_name: str) -> None:
    """Validate account_name format."""
    if not _ACCOUNT_NAME_RE.match(account_name):
        raise HTTPException(
            400,
            "Invalid account_name. Must be 1-64 characters, "
            "alphanumeric, underscores, or hyphens only.",
        )


@router.get("/api/credentials", dependencies=[Depends(verify_api_key)])
async def list_credentials(tenant_id: str = "default", service: Optional[str] = None):
    app = require_app()
    entries = await app.list_credentials(tenant_id, service=service)
    return [sanitize_credential(e) for e in entries]


@router.post("/api/credentials/{service}", dependencies=[Depends(verify_api_key)])
async def save_credential(service: str, req: CredentialSaveRequest, tenant_id: str = "default"):
    _validate_service(service)
    _validate_account_name(req.account_name)
    app = require_app()
    await app.save_credential(
        tenant_id=tenant_id,
        service=service,
        credentials=req.credentials,
        account_name=req.account_name,
    )
    return {"saved": True}


@router.delete("/api/credentials/{service}/{account_name}", dependencies=[Depends(verify_api_key)])
async def delete_credential(service: str, account_name: str, tenant_id: str = "default"):
    _validate_service(service)
    _validate_account_name(account_name)
    app = require_app()
    deleted = await app.delete_credential(
        tenant_id=tenant_id,
        service=service,
        account_name=account_name,
    )
    return {"deleted": deleted}


# --- Internal Credential APIs (service-to-service) ---


@router.get("/api/internal/credentials/by-email")
async def internal_credentials_by_email(
    request: Request, email: str, service: Optional[str] = None,
):
    """Lookup credentials by email. Returns full tokens. Internal use only."""
    verify_service_key(request)
    app = require_app()
    result = await app.find_credential_by_email(email, service)
    if not result:
        raise HTTPException(404, "No credentials found for email")
    return result


@router.get("/api/internal/credentials")
async def internal_credentials_get(
    request: Request, tenant_id: str, service: str, account_name: str = "primary",
):
    """Get full credentials including tokens. Internal use only."""
    verify_service_key(request)
    app = require_app()
    creds = await app.get_credential(tenant_id, service, account_name)
    if not creds:
        raise HTTPException(404, "Credentials not found")
    return {"tenant_id": tenant_id, "service": service, "account_name": account_name, "credentials": creds}


@router.put("/api/internal/credentials")
async def internal_credentials_update(
    request: Request, tenant_id: str, service: str,
    account_name: str = "primary",
):
    """Update credentials (e.g. after token refresh). Internal use only."""
    verify_service_key(request)
    app = require_app()
    body = await request.json()
    await app.save_credential_raw(tenant_id, service, body, account_name)
    return {"updated": True}
