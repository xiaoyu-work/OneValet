"""FastAPI app creation, CORS, global state, and helper functions."""

import logging
import os
import pathlib
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from ..app import OneValet

logger = logging.getLogger(__name__)

_config_path = os.getenv("ONEVALET_CONFIG", "config.yaml")
_STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "static"

_app: Optional[OneValet] = None

_SUPPORTED_PROVIDERS = ["openai", "anthropic", "azure", "dashscope", "gemini", "ollama"]

# Issue #15: Use None default so we can detect "not set" vs "empty string"
_INTERNAL_SERVICE_KEY = os.getenv("ONEVALET_SERVICE_KEY")


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


# Issue #14: Removed module-level _try_load_app() call.
# Now lazy-loaded inside require_app() on first request.


def require_app() -> OneValet:
    """Raise 503 if app is not configured. Lazy-loads on first call."""
    global _app
    if _app is None:
        _try_load_app()
    if _app is None:
        raise HTTPException(503, "Not configured. Complete setup in Settings.")
    return _app


def verify_service_key(request: Request):
    """Verify X-Service-Key header for internal endpoints."""
    if _INTERNAL_SERVICE_KEY is None:
        raise HTTPException(
            500,
            "ONEVALET_SERVICE_KEY is not configured. "
            "Set the ONEVALET_SERVICE_KEY environment variable to enable internal endpoints.",
        )
    key = request.headers.get("x-service-key", "")
    if key != _INTERNAL_SERVICE_KEY:
        raise HTTPException(403, "Invalid service key")


# ── Issue #3: Optional API key authentication ──

_API_KEY = os.getenv("ONEVALET_API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_auth_warning_logged = False


async def verify_api_key(
    request: Request,
    api_key_header_value: Optional[str] = Security(_api_key_header),
):
    """Verify API key from Authorization: Bearer <key> or X-API-Key header.

    When ONEVALET_API_KEY is not set, all requests are allowed (dev mode).
    When set, a valid key is required on sensitive endpoints.
    """
    if _API_KEY is None:
        # Dev mode: no authentication required
        return None

    # Check X-API-Key header first
    if api_key_header_value and api_key_header_value == _API_KEY:
        return api_key_header_value

    # Check Authorization: Bearer <key>
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == _API_KEY:
            return token

    raise HTTPException(401, "Invalid or missing API key")


def mask_api_key(key: str) -> dict:
    """Mask an API key for display."""
    if key and len(key) > 8:
        return {"api_key_display": key[:4] + "..." + key[-4:], "api_key_set": True}
    elif key:
        return {"api_key_display": "****", "api_key_set": True}
    return {"api_key_display": "", "api_key_set": False}


def mask_config(cfg: dict) -> dict:
    """Return config with api_key masked for display."""
    llm_cfg = cfg.get("llm", {})
    result = {
        "llm": {
            "provider": llm_cfg.get("provider", ""),
            "model": llm_cfg.get("model", ""),
            "base_url": llm_cfg.get("base_url", ""),
            **mask_api_key(llm_cfg.get("api_key", "")),
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
            **mask_api_key(embedding_cfg.get("api_key", "")),
        }
    return result


def sanitize_credential(entry: dict) -> dict:
    """Strip sensitive fields, keep metadata + email."""
    creds = entry.get("credentials", {})
    return {
        "service": entry.get("service"),
        "account_name": entry.get("account_name"),
        "email": creds.get("email", ""),
        "created_at": str(entry.get("created_at", "")),
        "updated_at": str(entry.get("updated_at", "")),
    }


def get_base_url(request: Request) -> str:
    """Determine base URL from request, respecting reverse proxy headers."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get(
        "x-forwarded-host", request.headers.get("host", "localhost:8000")
    )
    return f"{proto}://{host}"


def oauth_success_html(provider: str, email: str, detail: str) -> HTMLResponse:
    """HTML popup response for demo UI (backward compat when no redirect_after)."""
    return HTMLResponse(
        f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
        f"<h2>Connected!</h2>"
        f"<p>{detail} connected as <b>{email}</b></p>"
        f"<script>"
        f"window.opener&&window.opener.postMessage('oauth_complete','*');"
        f"setTimeout(()=>window.close(),1500);"
        f"</script></body></html>"
    )


def oauth_success_redirect(redirect_after: str, provider: str, email: str):
    """Redirect to caller-specified URL after successful OAuth."""
    from fastapi.responses import RedirectResponse
    sep = "&" if "?" in redirect_after else "?"
    url = f"{redirect_after}{sep}success=true&provider={provider}&email={email}"
    return RedirectResponse(url)


def set_app(new_app: Optional[OneValet]):
    """Set the global _app instance (used by config route on save)."""
    global _app
    _app = new_app


def get_app_instance() -> Optional[OneValet]:
    """Get the current global _app instance (may be None)."""
    return _app


# --- FastAPI app creation (after all helpers are defined to avoid circular imports) ---

def _create_api() -> FastAPI:
    """Create and configure the FastAPI app with routes."""
    _api = FastAPI(title="OneValet", version="0.1.1")

    # Issue #2: Configurable CORS origins from env var
    allowed_origins_str = os.getenv(
        "ONEVALET_ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]
    _api.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Issue #3: Log warning if no API key is set
    if _API_KEY is None:
        logger.warning(
            "ONEVALET_API_KEY is not set. API endpoints are unauthenticated. "
            "Set ONEVALET_API_KEY environment variable to enable authentication."
        )

    from .routes import register_routes
    register_routes(_api)
    return _api


api = _create_api()
