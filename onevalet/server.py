"""
OneValet REST API Server

Usage:
    python -m onevalet
    # → http://0.0.0.0:8000
"""

import json
import logging
import os
import pathlib
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import httpx

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .app import OneValet

logger = logging.getLogger(__name__)

_config_path = os.getenv("ONEVALET_CONFIG", "config.yaml")
_STATIC_DIR = pathlib.Path(__file__).parent / "static"

_app: Optional[OneValet] = None


def _try_load_app():
    """Attempt to load OneValet from config. Silent if config missing."""
    global _app
    try:
        if os.path.exists(_config_path):
            _app = OneValet(_config_path)
            logger.info(f"OneValet loaded from {_config_path}")
        else:
            logger.warning(f"Config not found: {_config_path}. Starting in setup mode.")
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Starting in setup mode.")
        _app = None


_try_load_app()

api = FastAPI(title="OneValet", version="0.1.1")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_TENANT_ID = "default"
_SUPPORTED_PROVIDERS = ["openai", "anthropic", "azure", "dashscope", "gemini", "ollama"]


def _require_app() -> OneValet:
    """Raise 503 if app is not configured."""
    if _app is None:
        raise HTTPException(503, "Not configured. Complete setup in Settings.")
    return _app


# ─── Models ───

class ChatRequest(BaseModel):
    message: str
    tenant_id: str = "default"
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    status: str


class CredentialSaveRequest(BaseModel):
    account_name: str = "primary"
    credentials: dict


class LLMConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class EmbeddingConfigRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None


class ConfigRequest(BaseModel):
    llm: LLMConfigRequest
    database: str
    embedding: EmbeddingConfigRequest
    system_prompt: Optional[str] = None


# ─── Demo UI (registered only with --ui) ───

def _register_ui_routes(app: FastAPI):
    """Register demo frontend routes. Only called when --ui flag is set."""

    @app.get("/")
    async def index():
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/settings")
    async def settings_page():
        return FileResponse(_STATIC_DIR / "settings.html", media_type="text/html")


# ─── Status & Config ───

@api.get("/api/status")
async def get_status():
    return {"configured": _app is not None}


def _mask_api_key(key: str) -> dict:
    """Mask an API key for display."""
    if key and len(key) > 8:
        return {"api_key_display": key[:4] + "..." + key[-4:], "api_key_set": True}
    elif key:
        return {"api_key_display": "****", "api_key_set": True}
    return {"api_key_display": "", "api_key_set": False}


def _mask_config(cfg: dict) -> dict:
    """Return config with api_key masked for display."""
    llm_cfg = cfg.get("llm", {})
    result = {
        "llm": {
            "provider": llm_cfg.get("provider", ""),
            "model": llm_cfg.get("model", ""),
            "base_url": llm_cfg.get("base_url", ""),
            **_mask_api_key(llm_cfg.get("api_key", "")),
        },
        "database": cfg.get("database", ""),
        "system_prompt": cfg.get("system_prompt", ""),
    }
    embedding_cfg = cfg.get("embedding")
    if embedding_cfg:
        result["embedding"] = {
            "provider": embedding_cfg.get("provider", ""),
            "model": embedding_cfg.get("model", ""),
            "base_url": embedding_cfg.get("base_url", ""),
            "api_version": embedding_cfg.get("api_version", ""),
            **_mask_api_key(embedding_cfg.get("api_key", "")),
        }
    return result


@api.get("/api/config")
async def get_config():
    """Return current configuration (API key masked)."""
    if _app is not None:
        return _mask_config(_app.config)
    # Try reading raw YAML if file exists but failed to load
    if os.path.exists(_config_path):
        with open(_config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f.read()) or {}
        return _mask_config(raw)
    return {}


@api.post("/api/config")
async def save_config(req: ConfigRequest):
    """Save configuration to config.yaml and reinitialize the app."""
    global _app

    if req.llm.provider not in _SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Unsupported provider: {req.llm.provider}")

    # Build config dict
    llm_config = {
        "provider": req.llm.provider,
        "model": req.llm.model,
    }
    if req.llm.api_key:
        llm_config["api_key"] = req.llm.api_key
    elif _app is not None:
        old_llm = _app.config.get("llm", {})
        if old_llm.get("api_key"):
            llm_config["api_key"] = old_llm["api_key"]

    if req.llm.base_url:
        llm_config["base_url"] = req.llm.base_url

    config = {
        "llm": llm_config,
        "database": req.database,
    }
    emb = {k: v for k, v in req.embedding.model_dump().items() if v}
    if emb:
        config["embedding"] = emb
    if req.system_prompt:
        config["system_prompt"] = req.system_prompt

    # Shut down existing app
    if _app is not None:
        try:
            await _app.shutdown()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        _app = None

    # Write config.yaml
    with open(_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Reload and initialize
    try:
        _app = OneValet(_config_path)
        await _app._ensure_initialized()
        return {"success": True, "message": "Configuration saved and initialized."}
    except Exception as e:
        _app = None
        logger.error(f"Config saved but initialization failed: {e}")
        raise HTTPException(
            422,
            f"Config saved but initialization failed: {e}. "
            f"Check your database URL and API key.",
        )


# ─── Chat ───

@api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    app = _require_app()
    result = await app.chat(
        message_or_tenant_id=req.tenant_id,
        message=req.message,
        metadata=req.metadata,
    )
    return ChatResponse(
        response=result.raw_message or "",
        status=result.status.value if result.status else "completed",
    )


@api.post("/stream")
async def stream(req: ChatRequest):
    app = _require_app()

    async def event_generator():
        async for event in app.stream(
            message_or_tenant_id=req.tenant_id,
            message=req.message,
            metadata=req.metadata,
        ):
            data = json.dumps({
                "type": event.type.value if event.type else "unknown",
                "data": event.data,
            }, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@api.get("/health")
async def health():
    return {"status": "ok"}


@api.post("/api/clear-session")
async def clear_session():
    """Clear conversation history for the default tenant."""
    app = _require_app()
    await app._ensure_initialized()
    app._momex.clear_history(tenant_id=_TENANT_ID, session_id=_TENANT_ID)
    return {"status": "ok", "message": "Session history cleared"}


@api.get("/api/test-tool-calling")
async def test_tool_calling():
    """Minimal test: does this Azure deployment support tool calling at all?"""
    app = _require_app()
    await app._ensure_initialized()

    test_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
    ]

    # Test 1: tool_choice=auto
    resp1 = await app._llm_client.chat_completion(
        messages=messages, tools=[test_tool]
    )
    auto_result = {
        "tool_choice": "auto",
        "has_tool_calls": resp1.has_tool_calls,
        "tool_calls": [tc.to_dict() for tc in resp1.tool_calls] if resp1.tool_calls else [],
        "content_len": len(resp1.content or ""),
        "stop_reason": str(resp1.stop_reason),
    }

    # Test 2: tool_choice=required
    resp2 = await app._llm_client.chat_completion(
        messages=messages, tools=[test_tool], tool_choice="required"
    )
    required_result = {
        "tool_choice": "required",
        "has_tool_calls": resp2.has_tool_calls,
        "tool_calls": [tc.to_dict() for tc in resp2.tool_calls] if resp2.tool_calls else [],
        "content_len": len(resp2.content or ""),
        "stop_reason": str(resp2.stop_reason),
    }

    return {"auto": auto_result, "required": required_result}


# ─── Credentials ───

_SENSITIVE_KEYS = {"access_token", "refresh_token", "client_secret", "client_id"}


def _sanitize_credential(entry: dict) -> dict:
    """Strip sensitive fields, keep metadata + email."""
    creds = entry.get("credentials", {})
    return {
        "service": entry.get("service"),
        "account_name": entry.get("account_name"),
        "email": creds.get("email", ""),
        "created_at": str(entry.get("created_at", "")),
        "updated_at": str(entry.get("updated_at", "")),
    }


@api.get("/api/credentials")
async def list_credentials():
    app = _require_app()
    await app._ensure_initialized()
    entries = await app._credential_store.list(_TENANT_ID)
    return [_sanitize_credential(e) for e in entries]


@api.post("/api/credentials/{service}")
async def save_credential(service: str, req: CredentialSaveRequest):
    app = _require_app()
    await app._ensure_initialized()
    await app._credential_store.save(
        tenant_id=_TENANT_ID,
        service=service,
        credentials=req.credentials,
        account_name=req.account_name,
    )
    # Reload API keys / OAuth app credentials into env vars immediately
    await app._load_api_keys_to_env()
    return {"saved": True}


@api.delete("/api/credentials/{service}/{account_name}")
async def delete_credential(service: str, account_name: str):
    app = _require_app()
    await app._ensure_initialized()
    deleted = await app._credential_store.delete(
        tenant_id=_TENANT_ID,
        service=service,
        account_name=account_name,
    )
    return {"deleted": deleted}


# ─── OAuth ───

_oauth_states: dict[str, dict] = {}


def _generate_state(service: str) -> str:
    """Generate a cryptographic state token for CSRF protection."""
    token = secrets.token_urlsafe(32)
    _oauth_states[token] = {"created_at": time.time(), "service": service}
    # Garbage-collect expired states (>10 min)
    cutoff = time.time() - 600
    expired = [k for k, v in _oauth_states.items() if v["created_at"] < cutoff]
    for k in expired:
        del _oauth_states[k]
    return token


def _validate_state(state: str) -> Optional[str]:
    """Validate and consume a state token. Returns service name or None."""
    entry = _oauth_states.pop(state, None)
    if not entry:
        return None
    if time.time() - entry["created_at"] > 600:
        return None
    return entry["service"]


def _get_base_url(request: Request) -> str:
    """Determine base URL from request, respecting reverse proxy headers."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get(
        "x-forwarded-host", request.headers.get("host", "localhost:8000")
    )
    return f"{proto}://{host}"


@api.get("/api/oauth/google/authorize")
async def google_oauth_authorize(request: Request):
    """Initiate Google OAuth flow. Returns authorization URL."""
    from .oauth.google_oauth import GoogleOAuth

    app = _require_app()
    await app._ensure_initialized()

    state = _generate_state("google")
    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/google/callback"

    try:
        url = GoogleOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/api/oauth/google/callback")
async def google_oauth_callback(request: Request, code: str, state: str):
    """Google OAuth callback — exchange code for tokens and store credentials."""
    from .oauth.google_oauth import GoogleOAuth

    service = _validate_state(state)
    if not service:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    app = _require_app()
    await app._ensure_initialized()

    base_url = _get_base_url(request)
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

        # Save to both gmail and google_calendar (scopes cover both)
        for svc in ("gmail", "google_calendar", "google_tasks", "google_drive"):
            await app._credential_store.save(
                tenant_id=_TENANT_ID,
                service=svc,
                credentials=credentials,
                account_name="primary",
            )

        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
            f"<h2>Connected!</h2>"
            f"<p>Gmail, Google Calendar, Tasks, Drive, Docs &amp; Sheets connected as <b>{email}</b></p>"
            f"<script>"
            f"window.opener&&window.opener.postMessage('oauth_complete','*');"
            f"setTimeout(()=>window.close(),1500);"
            f"</script></body></html>"
        )
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{e}</p>",
            status_code=500,
        )


@api.get("/api/oauth/microsoft/authorize")
async def microsoft_oauth_authorize(request: Request):
    """Initiate Microsoft OAuth flow. Returns authorization URL."""
    from .oauth.microsoft_oauth import MicrosoftOAuth

    app = _require_app()
    await app._ensure_initialized()

    state = _generate_state("microsoft")
    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/microsoft/callback"

    try:
        url = MicrosoftOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/api/oauth/microsoft/callback")
async def microsoft_oauth_callback(request: Request, code: str, state: str):
    """Microsoft OAuth callback — exchange code for tokens and store credentials."""
    from .oauth.microsoft_oauth import MicrosoftOAuth

    service = _validate_state(state)
    if not service:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    app = _require_app()
    await app._ensure_initialized()

    base_url = _get_base_url(request)
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

        # Save to both outlook and outlook_calendar (scopes cover both)
        for svc in ("outlook", "outlook_calendar", "microsoft_todo", "onedrive"):
            await app._credential_store.save(
                tenant_id=_TENANT_ID,
                service=svc,
                credentials=credentials,
                account_name="primary",
            )

        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
            f"<h2>Connected!</h2>"
            f"<p>Outlook, Calendar, To Do &amp; OneDrive connected as <b>{email}</b></p>"
            f"<script>"
            f"window.opener&&window.opener.postMessage('oauth_complete','*');"
            f"setTimeout(()=>window.close(),1500);"
            f"</script></body></html>"
        )
    except Exception as e:
        logger.error(f"Microsoft OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{e}</p>",
            status_code=500,
        )


@api.get("/api/oauth/todoist/authorize")
async def todoist_oauth_authorize(request: Request):
    """Initiate Todoist OAuth flow. Returns authorization URL."""
    app = _require_app()
    await app._ensure_initialized()

    client_id = os.getenv("TODOIST_CLIENT_ID")
    if not client_id:
        raise HTTPException(400, "Todoist OAuth not configured. Set TODOIST_CLIENT_ID in Settings > OAuth Apps.")

    state = _generate_state("todoist")
    params = {
        "client_id": client_id,
        "scope": "data:read_write",
        "state": state,
    }
    url = f"https://todoist.com/oauth/authorize?{urlencode(params)}"
    return {"authorize_url": url}


@api.get("/api/oauth/todoist/callback")
async def todoist_oauth_callback(request: Request, code: str, state: str):
    """Todoist OAuth callback — exchange code for token and store credentials."""
    service = _validate_state(state)
    if not service:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    app = _require_app()
    await app._ensure_initialized()

    client_id = os.getenv("TODOIST_CLIENT_ID")
    client_secret = os.getenv("TODOIST_CLIENT_SECRET")
    if not client_id or not client_secret:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Todoist OAuth not configured.</p>",
            status_code=500,
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://todoist.com/oauth/access_token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        access_token = data["access_token"]

        # Fetch user email from Todoist Sync API
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

        await app._credential_store.save(
            tenant_id=_TENANT_ID,
            service="todoist",
            credentials=credentials,
            account_name="primary",
        )

        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
            f"<h2>Connected!</h2>"
            f"<p>Todoist connected as <b>{email}</b></p>"
            f"<script>"
            f"window.opener&&window.opener.postMessage('oauth_complete','*');"
            f"setTimeout(()=>window.close(),1500);"
            f"</script></body></html>"
        )
    except Exception as e:
        logger.error(f"Todoist OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{e}</p>",
            status_code=500,
        )


# ─── Hue OAuth ───


@api.get("/api/oauth/hue/authorize")
async def hue_oauth_authorize(request: Request):
    """Initiate Philips Hue OAuth flow. Returns authorization URL."""
    from .oauth.hue_oauth import HueOAuth

    app = _require_app()
    await app._ensure_initialized()

    state = _generate_state("hue")
    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/hue/callback"

    try:
        url = HueOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/api/oauth/hue/callback")
async def hue_oauth_callback(request: Request, code: str, state: str):
    """Hue OAuth callback — exchange code for tokens and store credentials."""
    from .oauth.hue_oauth import HueOAuth

    service = _validate_state(state)
    if not service:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    app = _require_app()
    await app._ensure_initialized()

    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/hue/callback"

    try:
        tokens = await HueOAuth.exchange_code(code=code, redirect_uri=redirect_uri)

        credentials = {
            "provider": "philips_hue",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
        }

        await app._credential_store.save(
            tenant_id=_TENANT_ID,
            service="philips_hue",
            credentials=credentials,
            account_name="primary",
        )

        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
            f"<h2>Connected!</h2>"
            f"<p>Philips Hue connected.</p>"
            f"<script>"
            f"window.opener&&window.opener.postMessage('oauth_complete','*');"
            f"setTimeout(()=>window.close(),1500);"
            f"</script></body></html>"
        )
    except Exception as e:
        logger.error(f"Hue OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{e}</p>",
            status_code=500,
        )


# ─── Sonos OAuth ───


@api.get("/api/oauth/sonos/authorize")
async def sonos_oauth_authorize(request: Request):
    """Initiate Sonos OAuth flow. Returns authorization URL."""
    from .oauth.sonos_oauth import SonosOAuth

    app = _require_app()
    await app._ensure_initialized()

    state = _generate_state("sonos")
    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/sonos/callback"

    try:
        url = SonosOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/api/oauth/sonos/callback")
async def sonos_oauth_callback(request: Request, code: str, state: str):
    """Sonos OAuth callback — exchange code for tokens and store credentials."""
    from .oauth.sonos_oauth import SonosOAuth

    service = _validate_state(state)
    if not service:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    app = _require_app()
    await app._ensure_initialized()

    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/sonos/callback"

    try:
        tokens = await SonosOAuth.exchange_code(code=code, redirect_uri=redirect_uri)

        credentials = {
            "provider": "sonos",
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_expiry": tokens["token_expiry"],
        }

        await app._credential_store.save(
            tenant_id=_TENANT_ID,
            service="sonos",
            credentials=credentials,
            account_name="primary",
        )

        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
            f"<h2>Connected!</h2>"
            f"<p>Sonos connected.</p>"
            f"<script>"
            f"window.opener&&window.opener.postMessage('oauth_complete','*');"
            f"setTimeout(()=>window.close(),1500);"
            f"</script></body></html>"
        )
    except Exception as e:
        logger.error(f"Sonos OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{e}</p>",
            status_code=500,
        )


# ─── Dropbox OAuth ───


@api.get("/api/oauth/dropbox/authorize")
async def dropbox_oauth_authorize(request: Request):
    """Initiate Dropbox OAuth flow. Returns authorization URL."""
    from .oauth.dropbox_oauth import DropboxOAuth

    app = _require_app()
    await app._ensure_initialized()

    state = _generate_state("dropbox")
    base_url = _get_base_url(request)
    redirect_uri = f"{base_url}/api/oauth/dropbox/callback"

    try:
        url = DropboxOAuth.build_authorize_url(redirect_uri=redirect_uri, state=state)
        return {"authorize_url": url}
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/api/oauth/dropbox/callback")
async def dropbox_oauth_callback(request: Request, code: str, state: str):
    """Dropbox OAuth callback — exchange code for tokens and store credentials."""
    from .oauth.dropbox_oauth import DropboxOAuth

    service = _validate_state(state)
    if not service:
        return HTMLResponse(
            "<h2>OAuth Error</h2><p>Invalid or expired state. Please try again.</p>",
            status_code=400,
        )

    app = _require_app()
    await app._ensure_initialized()

    base_url = _get_base_url(request)
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

        await app._credential_store.save(
            tenant_id=_TENANT_ID,
            service="dropbox",
            credentials=credentials,
            account_name="primary",
        )

        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
            f"<h2>Connected!</h2>"
            f"<p>Dropbox connected as <b>{email}</b></p>"
            f"<script>"
            f"window.opener&&window.opener.postMessage('oauth_complete','*');"
            f"setTimeout(()=>window.close(),1500);"
            f"</script></body></html>"
        )
    except Exception as e:
        logger.error(f"Dropbox OAuth callback failed: {e}", exc_info=True)
        return HTMLResponse(
            f"<h2>OAuth Error</h2><p>{e}</p>",
            status_code=500,
        )


# ─── Main ───

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OneValet API Server")
    parser.add_argument("--ui", action="store_true", help="Serve demo frontend (/ and /settings)")
    parser.add_argument("--host", default=os.getenv("ONEVALET_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ONEVALET_PORT", "8000")))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s - %(message)s",
    )

    if args.ui:
        _register_ui_routes(api)

    uvicorn.run(api, host=args.host, port=args.port)
