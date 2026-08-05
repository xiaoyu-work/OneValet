"""Request/tenant ContextVars for concurrent-safe correlation tracking.

Using ``contextvars`` (PEP 567) rather than instance-level state ensures that
concurrent requests running on the same orchestrator instance cannot overwrite
each other's tracing context.  Values set in a parent coroutine are inherited
by child tasks created via ``asyncio.create_task``.

Usage::

    from koa.observability.context import bind_request_context

    with bind_request_context(request_id="abc123", tenant_id="u42"):
        await do_work()
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

#: Current request id (12-char hex by default).  ``None`` when not in a request.
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "koa_request_id", default=None
)

#: Current tenant id.  ``None`` when not bound.
tenant_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "koa_tenant_id", default=None
)

#: Optional caller-supplied idempotency key (e.g. from ``X-Idempotency-Key``).
idempotency_key_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "koa_idempotency_key", default=None
)


def get_request_id() -> Optional[str]:
    """Return the current request id, or ``None`` if unbound."""
    return request_id_var.get()


def get_tenant_id() -> Optional[str]:
    """Return the current tenant id, or ``None`` if unbound."""
    return tenant_id_var.get()


def get_idempotency_key() -> Optional[str]:
    """Return the caller-supplied idempotency key, or ``None``."""
    return idempotency_key_var.get()


def new_request_id() -> str:
    """Generate a globally unique, URL-safe request id.

    Request ids are durable run ids and global database keys, not display
    labels. Truncating UUID4 to 48 bits makes collisions plausible at the
    tens-of-millions scale (and a collision overwrites another transcript),
    so keep all 128 bits. The hex form remains safe in logs and URL paths.
    """
    return uuid.uuid4().hex


@contextmanager
def bind_request_context(
    *,
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Iterator[str]:
    """Bind request-scoped context vars; restores previous values on exit.

    If ``request_id`` is ``None``, a new 32-char UUID hex id is generated.

    Yields the effective request id.
    """
    rid = request_id or new_request_id()
    rtok = request_id_var.set(rid)
    ttok = tenant_id_var.set(tenant_id) if tenant_id is not None else None
    ktok = idempotency_key_var.set(idempotency_key) if idempotency_key is not None else None
    try:
        yield rid
    finally:
        request_id_var.reset(rtok)
        if ttok is not None:
            tenant_id_var.reset(ttok)
        if ktok is not None:
            idempotency_key_var.reset(ktok)
