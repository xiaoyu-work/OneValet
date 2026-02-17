"""Credential management routes (public and internal)."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from ..app import require_app, sanitize_credential, verify_service_key
from ..models import CredentialSaveRequest

router = APIRouter()


@router.get("/api/credentials")
async def list_credentials(tenant_id: str = "default", service: Optional[str] = None):
    app = require_app()
    await app._ensure_initialized()
    entries = await app._credential_store.list(tenant_id, service=service)
    return [sanitize_credential(e) for e in entries]


@router.post("/api/credentials/{service}")
async def save_credential(service: str, req: CredentialSaveRequest, tenant_id: str = "default"):
    app = require_app()
    await app._ensure_initialized()
    await app._credential_store.save(
        tenant_id=tenant_id,
        service=service,
        credentials=req.credentials,
        account_name=req.account_name,
    )
    # Reload API keys / OAuth app credentials into env vars immediately
    await app._load_api_keys_to_env()
    return {"saved": True}


@router.delete("/api/credentials/{service}/{account_name}")
async def delete_credential(service: str, account_name: str, tenant_id: str = "default"):
    app = require_app()
    await app._ensure_initialized()
    deleted = await app._credential_store.delete(
        tenant_id=tenant_id,
        service=service,
        account_name=account_name,
    )
    return {"deleted": deleted}


# ─── Internal Credential APIs (service-to-service) ───


@router.get("/api/internal/credentials/by-email")
async def internal_credentials_by_email(
    request: Request, email: str, service: Optional[str] = None,
):
    """Lookup credentials by email. Returns full tokens. Internal use only."""
    verify_service_key(request)
    app = require_app()
    await app._ensure_initialized()
    result = await app._credential_store.find_by_email(email, service)
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
    await app._ensure_initialized()
    creds = await app._credential_store.get(tenant_id, service, account_name)
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
    await app._ensure_initialized()
    body = await request.json()
    await app._credential_store.save(tenant_id, service, body, account_name)
    return {"updated": True}
