"""
OneValet unified error handling.

All API and SSE errors use OneValetError with a machine-readable ``code``,
a technical ``message`` (for logs/debugging), and optional ``details``.

Usage in routes::

    from onevalet.errors import OneValetError, E

    raise OneValetError(E.NOT_FOUND, "Task not found", details={"resource": "task"})

The FastAPI exception handler (registered via ``install_error_handler``)
serialises it as::

    {
      "error": {
        "code": "not_found",
        "message": "Task not found",
        "details": {"resource": "task"}
      }
    }
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------

class E:
    """Error code constants."""

    # Input / validation
    VALIDATION_ERROR = "validation_error"

    # Resource lookup
    NOT_FOUND = "not_found"

    # Service availability (backend not ready)
    SERVICE_UNAVAILABLE = "service_unavailable"

    # User hasn't connected a required OAuth service
    SERVICE_NOT_CONNECTED = "service_not_connected"

    # OAuth flow errors
    OAUTH_STATE_EXPIRED = "oauth_state_expired"
    OAUTH_NOT_CONFIGURED = "oauth_not_configured"
    OAUTH_FAILED = "oauth_failed"

    # Provider not recognised
    PROVIDER_NOT_SUPPORTED = "provider_not_supported"

    # Configuration
    CONFIG_ERROR = "config_error"

    # Agent lifecycle
    AGENT_FAILED = "agent_failed"

    # Catch-all
    INTERNAL_ERROR = "internal_error"


# Default HTTP status codes per error code
_DEFAULT_STATUS: Dict[str, int] = {
    E.VALIDATION_ERROR: 400,
    E.NOT_FOUND: 404,
    E.SERVICE_UNAVAILABLE: 503,
    E.SERVICE_NOT_CONNECTED: 400,
    E.OAUTH_STATE_EXPIRED: 400,
    E.OAUTH_NOT_CONFIGURED: 400,
    E.OAUTH_FAILED: 502,
    E.PROVIDER_NOT_SUPPORTED: 400,
    E.CONFIG_ERROR: 422,
    E.AGENT_FAILED: 500,
    E.INTERNAL_ERROR: 500,
}


# ---------------------------------------------------------------------------
# Exception class
# ---------------------------------------------------------------------------

class OneValetError(Exception):
    """Structured error raised anywhere in OneValet."""

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message or code
        self.status_code = status_code or _DEFAULT_STATUS.get(code, 500)
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON responses and SSE error events."""
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            d["details"] = self.details
        return d


# ---------------------------------------------------------------------------
# FastAPI integration (lazy import to avoid hard dependency at module level)
# ---------------------------------------------------------------------------

def install_error_handler(app) -> None:
    """Register the global exception handler on a FastAPI app."""
    from fastapi.responses import JSONResponse

    @app.exception_handler(OneValetError)
    async def _handler(request, exc: OneValetError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.to_dict()},
        )
