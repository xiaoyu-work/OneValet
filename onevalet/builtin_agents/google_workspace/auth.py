"""Google OAuth token helper for ReAct tools and agents."""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import httpx

from onevalet.standard_agent import AgentToolContext

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
# Refresh 5 minutes before expiry
REFRESH_BUFFER = timedelta(minutes=5)


async def get_google_token(context: AgentToolContext) -> Tuple[Optional[str], Optional[str]]:
    """
    Get a valid Google access token from the credential store.
    Handles token refresh if expiring.

    Args:
        context: AgentToolContext with user_id and credentials store

    Returns:
        (access_token, None) on success
        (None, error_message) on failure
    """
    if not context or not context.credentials:
        return None, "Google account not connected. Please connect Google in Settings."

    # Get credentials from store (reuse "gmail" — same OAuth token)
    try:
        entries = await context.credentials.list(context.tenant_id, service="gmail")
    except Exception as e:
        logger.error(f"Failed to read credentials: {e}")
        return None, "Failed to read Google credentials."

    if not entries:
        return None, "Google account not connected. Please connect Google in Settings."

    creds = entries[0].get("credentials", {})
    access_token = creds.get("access_token", "")
    refresh_token = creds.get("refresh_token", "")
    token_expiry_str = creds.get("token_expiry", "")

    if not access_token:
        return None, "Google access token missing. Please reconnect Google in Settings."

    # Check if token needs refresh
    needs_refresh = False
    if token_expiry_str:
        try:
            from dateutil import parser
            token_expiry = parser.parse(token_expiry_str)
            if token_expiry <= datetime.now(timezone.utc) + REFRESH_BUFFER:
                needs_refresh = True
        except Exception:
            needs_refresh = True  # Can't parse expiry, try refresh

    if needs_refresh:
        if not refresh_token:
            return None, "Google token expired and no refresh token. Please reconnect Google in Settings."

        new_token, error = await _refresh_token(refresh_token)
        if error:
            return None, error

        # Update credential store
        creds["access_token"] = new_token["access_token"]
        creds["token_expiry"] = new_token["token_expiry"]
        try:
            for svc in ("gmail", "google_calendar", "google_tasks", "google_drive"):
                await context.credentials.save(
                    tenant_id=context.tenant_id,
                    service=svc,
                    credentials=creds,
                    account_name="primary",
                )
        except Exception as e:
            logger.warning(f"Failed to persist refreshed token: {e}")

        return new_token["access_token"], None

    return access_token, None


async def get_google_token_for_agent(tenant_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get a valid Google access token for agent use (no AgentToolContext).

    Uses AccountResolver's default credential store. If the token is expired
    and a refresh token is available, attempts refresh.

    Args:
        tenant_id: Tenant/user ID

    Returns:
        (access_token, None) on success
        (None, error_message) on failure
    """
    from onevalet.providers.email.resolver import AccountResolver

    store = AccountResolver._default_store
    if not store:
        return None, "Credential store not available."

    try:
        entries = await store.list(tenant_id, service="gmail")
    except Exception as e:
        logger.error(f"Failed to read credentials: {e}")
        return None, "Failed to read Google credentials."

    if not entries:
        return None, "Google account not connected. Please connect Google in Settings."

    creds = entries[0].get("credentials", {})
    access_token = creds.get("access_token", "")
    refresh_token = creds.get("refresh_token", "")
    token_expiry_str = creds.get("token_expiry", "")

    if not access_token:
        return None, "Google access token missing. Please reconnect Google in Settings."

    # Check if token needs refresh
    needs_refresh = False
    if token_expiry_str:
        try:
            from dateutil import parser
            token_expiry = parser.parse(token_expiry_str)
            if token_expiry <= datetime.now(timezone.utc) + REFRESH_BUFFER:
                needs_refresh = True
        except Exception:
            needs_refresh = True

    if needs_refresh:
        if not refresh_token:
            return None, "Google token expired and no refresh token. Please reconnect Google in Settings."

        new_token, error = await _refresh_token(refresh_token)
        if error:
            return None, error

        # Update credential store
        creds["access_token"] = new_token["access_token"]
        creds["token_expiry"] = new_token["token_expiry"]
        try:
            for svc in ("gmail", "google_calendar", "google_tasks", "google_drive"):
                await store.save(
                    tenant_id=tenant_id,
                    service=svc,
                    credentials=creds,
                    account_name="primary",
                )
        except Exception as e:
            logger.warning(f"Failed to persist refreshed token: {e}")

        return new_token["access_token"], None

    return access_token, None


async def _refresh_token(refresh_token: str) -> Tuple[Optional[dict], Optional[str]]:
    """Refresh Google OAuth token. Returns (token_dict, None) or (None, error)."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None, "Google OAuth not configured. Set Google OAuth App credentials in Settings."

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None, "Google token refresh failed. Please reconnect Google in Settings."

            data = response.json()
            expires_in = data.get("expires_in", 3600)
            token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            return {
                "access_token": data["access_token"],
                "token_expiry": token_expiry.isoformat(),
            }, None
    except Exception as e:
        logger.error(f"Token refresh error: {e}", exc_info=True)
        return None, f"Token refresh error: {e}"
