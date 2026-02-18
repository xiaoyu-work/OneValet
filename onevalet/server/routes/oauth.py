"""OAuth provider authorize + callback routes."""

import logging
import os
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..app import (
    get_base_url,
    oauth_success_html,
    oauth_success_redirect,
    require_app,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Google OAuth ---


@router.get("/api/oauth/google/authorize")
async def google_oauth_authorize(
    request: Request,
    tenant_id: str = "default",
    redirect_after: Optional[str] = None,
    account_name: str = "primary",
):
    """Initiate Google OAuth flow. Returns authorization URL."""
    from ...oauth.google_oauth import GoogleOAuth

    app = require_app()

    state = await app.save_oauth_state(
        tenant_id=tenant_id, service="google",
        redirect_after=redirect_after, account_name=account_name,
    )
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/google/callback"

    try:
        url = GoogleOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/oauth/google/callback")
async def google_oauth_callback(request: Request, code: str, state: str):
    """Google OAuth callback -- exchange code for tokens and store credentials."""
    from ...oauth.google_oauth import GoogleOAuth

    app = require_app()

    state_data = await app.consume_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    tenant_id = state_data["tenant_id"]
    account_name = state_data["account_name"]
    redirect_after = state_data["redirect_after"]

    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/google/callback"

    try:
        tokens = await GoogleOAuth.exchange_code(code=code, redirect_uri=redirect_uri)
        email = await GoogleOAuth.fetch_user_email(tokens["access_token"])

        credentials = {
            "provider": "google",
            "email": email,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
            "scopes": tokens.get("scope", "").split(),
        }

        for svc in ("gmail", "google_calendar", "google_tasks", "google_drive"):
            await app.save_credential_raw(
                tenant_id=tenant_id, service=svc,
                credentials=credentials, account_name=account_name,
            )

        if redirect_after:
            return oauth_success_redirect(redirect_after, "google", email)
        return oauth_success_html("google", email, "Gmail, Google Calendar, Tasks, Drive")
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(f"<h2>OAuth Error</h2><p>{e}</p>", status_code=500)


# --- Microsoft OAuth ---


@router.get("/api/oauth/microsoft/authorize")
async def microsoft_oauth_authorize(
    request: Request,
    tenant_id: str = "default",
    redirect_after: Optional[str] = None,
    account_name: str = "primary",
):
    """Initiate Microsoft OAuth flow. Returns authorization URL."""
    from ...oauth.microsoft_oauth import MicrosoftOAuth

    app = require_app()

    state = await app.save_oauth_state(
        tenant_id=tenant_id, service="microsoft",
        redirect_after=redirect_after, account_name=account_name,
    )
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/microsoft/callback"

    try:
        url = MicrosoftOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/oauth/microsoft/callback")
async def microsoft_oauth_callback(request: Request, code: str, state: str):
    """Microsoft OAuth callback -- exchange code for tokens and store credentials."""
    from ...oauth.microsoft_oauth import MicrosoftOAuth

    app = require_app()

    state_data = await app.consume_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    tenant_id = state_data["tenant_id"]
    account_name = state_data["account_name"]
    redirect_after = state_data["redirect_after"]

    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/microsoft/callback"

    try:
        tokens = await MicrosoftOAuth.exchange_code(code=code, redirect_uri=redirect_uri)
        email = await MicrosoftOAuth.fetch_user_email(tokens["access_token"])

        credentials = {
            "provider": "microsoft",
            "email": email,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
            "scopes": tokens.get("scope", "").split(),
        }

        for svc in ("outlook", "outlook_calendar", "microsoft_todo", "onedrive"):
            await app.save_credential_raw(
                tenant_id=tenant_id, service=svc,
                credentials=credentials, account_name=account_name,
            )

        if redirect_after:
            return oauth_success_redirect(redirect_after, "microsoft", email)
        return oauth_success_html("microsoft", email, "Outlook, Calendar, To Do &amp; OneDrive")
    except Exception as e:
        logger.error(f"Microsoft OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(f"<h2>OAuth Error</h2><p>{e}</p>", status_code=500)


# --- Todoist OAuth ---


@router.get("/api/oauth/todoist/authorize")
async def todoist_oauth_authorize(
    request: Request,
    tenant_id: str = "default",
    redirect_after: Optional[str] = None,
    account_name: str = "primary",
):
    """Initiate Todoist OAuth flow. Returns authorization URL."""
    app = require_app()

    client_id = os.getenv("TODOIST_CLIENT_ID")
    if not client_id:
        raise HTTPException(400, "Todoist OAuth not configured. Set TODOIST_CLIENT_ID in Settings > OAuth Apps.")

    state = await app.save_oauth_state(
        tenant_id=tenant_id, service="todoist",
        redirect_after=redirect_after, account_name=account_name,
    )
    params = {
        "client_id": client_id,
        "scope": "data:read_write",
        "state": state,
    }
    url = f"https://todoist.com/oauth/authorize?{urlencode(params)}"
    return {"authorize_url": url}


@router.get("/api/oauth/todoist/callback")
async def todoist_oauth_callback(request: Request, code: str, state: str):
    """Todoist OAuth callback -- exchange code for token and store credentials."""
    app = require_app()

    state_data = await app.consume_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    tenant_id = state_data["tenant_id"]
    account_name = state_data["account_name"]
    redirect_after = state_data["redirect_after"]

    client_id = os.getenv("TODOIST_CLIENT_ID")
    client_secret = os.getenv("TODOIST_CLIENT_SECRET")
    if not client_id or not client_secret:
        return HTMLResponse("<h2>OAuth Error</h2><p>Todoist OAuth not configured.</p>", status_code=500)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://todoist.com/oauth/access_token",
                data={"client_id": client_id, "client_secret": client_secret, "code": code},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        access_token = data["access_token"]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.todoist.com/sync/v9/sync",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"sync_token": "*", "resource_types": ["user"]},
                timeout=15.0,
            )
            response.raise_for_status()
            user_data = response.json()
            email = user_data.get("user", {}).get("email", "")

        credentials = {
            "provider": "todoist",
            "email": email,
            "access_token": access_token,
            "refresh_token": "",
            "token_expiry": "",
            "scopes": ["data:read_write"],
        }

        await app.save_credential_raw(
            tenant_id=tenant_id, service="todoist",
            credentials=credentials, account_name=account_name,
        )

        if redirect_after:
            return oauth_success_redirect(redirect_after, "todoist", email)
        return oauth_success_html("todoist", email, "Todoist")
    except Exception as e:
        logger.error(f"Todoist OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(f"<h2>OAuth Error</h2><p>{e}</p>", status_code=500)


# --- Hue OAuth ---


@router.get("/api/oauth/hue/authorize")
async def hue_oauth_authorize(
    request: Request,
    tenant_id: str = "default",
    redirect_after: Optional[str] = None,
    account_name: str = "primary",
):
    """Initiate Philips Hue OAuth flow. Returns authorization URL."""
    from ...oauth.hue_oauth import HueOAuth

    app = require_app()

    state = await app.save_oauth_state(
        tenant_id=tenant_id, service="hue",
        redirect_after=redirect_after, account_name=account_name,
    )
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/hue/callback"

    try:
        url = HueOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/oauth/hue/callback")
async def hue_oauth_callback(request: Request, code: str, state: str):
    """Hue OAuth callback -- exchange code for tokens and store credentials."""
    from ...oauth.hue_oauth import HueOAuth

    app = require_app()

    state_data = await app.consume_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    tenant_id = state_data["tenant_id"]
    account_name = state_data["account_name"]
    redirect_after = state_data["redirect_after"]

    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/hue/callback"

    try:
        tokens = await HueOAuth.exchange_code(code=code, redirect_uri=redirect_uri)

        credentials = {
            "provider": "philips_hue",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
        }

        await app.save_credential_raw(
            tenant_id=tenant_id, service="philips_hue",
            credentials=credentials, account_name=account_name,
        )

        if redirect_after:
            return oauth_success_redirect(redirect_after, "hue", "")
        return oauth_success_html("hue", "", "Philips Hue")
    except Exception as e:
        logger.error(f"Hue OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(f"<h2>OAuth Error</h2><p>{e}</p>", status_code=500)


# --- Sonos OAuth ---


@router.get("/api/oauth/sonos/authorize")
async def sonos_oauth_authorize(
    request: Request,
    tenant_id: str = "default",
    redirect_after: Optional[str] = None,
    account_name: str = "primary",
):
    """Initiate Sonos OAuth flow. Returns authorization URL."""
    from ...oauth.sonos_oauth import SonosOAuth

    app = require_app()

    state = await app.save_oauth_state(
        tenant_id=tenant_id, service="sonos",
        redirect_after=redirect_after, account_name=account_name,
    )
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/sonos/callback"

    try:
        url = SonosOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/oauth/sonos/callback")
async def sonos_oauth_callback(request: Request, code: str, state: str):
    """Sonos OAuth callback -- exchange code for tokens and store credentials."""
    from ...oauth.sonos_oauth import SonosOAuth

    app = require_app()

    state_data = await app.consume_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    tenant_id = state_data["tenant_id"]
    account_name = state_data["account_name"]
    redirect_after = state_data["redirect_after"]

    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/sonos/callback"

    try:
        tokens = await SonosOAuth.exchange_code(code=code, redirect_uri=redirect_uri)

        credentials = {
            "provider": "sonos",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
        }

        await app.save_credential_raw(
            tenant_id=tenant_id, service="sonos",
            credentials=credentials, account_name=account_name,
        )

        if redirect_after:
            return oauth_success_redirect(redirect_after, "sonos", "")
        return oauth_success_html("sonos", "", "Sonos")
    except Exception as e:
        logger.error(f"Sonos OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(f"<h2>OAuth Error</h2><p>{e}</p>", status_code=500)


# --- Dropbox OAuth ---


@router.get("/api/oauth/dropbox/authorize")
async def dropbox_oauth_authorize(
    request: Request,
    tenant_id: str = "default",
    redirect_after: Optional[str] = None,
    account_name: str = "primary",
):
    """Initiate Dropbox OAuth flow. Returns authorization URL."""
    from ...oauth.dropbox_oauth import DropboxOAuth

    app = require_app()

    state = await app.save_oauth_state(
        tenant_id=tenant_id, service="dropbox",
        redirect_after=redirect_after, account_name=account_name,
    )
    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/dropbox/callback"

    try:
        url = DropboxOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/oauth/dropbox/callback")
async def dropbox_oauth_callback(request: Request, code: str, state: str):
    """Dropbox OAuth callback -- exchange code for tokens and store credentials."""
    from ...oauth.dropbox_oauth import DropboxOAuth

    app = require_app()

    state_data = await app.consume_oauth_state(state)
    if not state_data:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    tenant_id = state_data["tenant_id"]
    account_name = state_data["account_name"]
    redirect_after = state_data["redirect_after"]

    base_url = get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/dropbox/callback"

    try:
        tokens = await DropboxOAuth.exchange_code(code=code, redirect_uri=redirect_uri)
        email = await DropboxOAuth.fetch_user_email(tokens["access_token"])

        credentials = {
            "provider": "dropbox",
            "email": email,
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
        }

        await app.save_credential_raw(
            tenant_id=tenant_id, service="dropbox",
            credentials=credentials, account_name=account_name,
        )

        if redirect_after:
            return oauth_success_redirect(redirect_after, "dropbox", email)
        return oauth_success_html("dropbox", email, "Dropbox")
    except Exception as e:
        logger.error(f"Dropbox OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(f"<h2>OAuth Error</h2><p>{e}</p>", status_code=500)
