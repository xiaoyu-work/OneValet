"""Optional OpenTelemetry tracing integration.

If ``opentelemetry-api`` / ``opentelemetry-sdk`` are not installed, all helpers
become no-ops so callers can unconditionally wrap regions in spans.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_tracer: Optional[Any] = None
_enabled = False


def configure_tracing(
    *,
    enabled: bool = True,
    service_name: str = "koa",
    otlp_endpoint: Optional[str] = None,
) -> None:
    """Set up OpenTelemetry tracing.

    If ``opentelemetry`` isn't installed, logs a warning and degrades to no-op.
    When ``otlp_endpoint`` is provided, configures an OTLP span exporter;
    otherwise no exporter is attached (useful for tests / local dev).
    """
    global _tracer, _enabled
    if not enabled:
        _enabled = False
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "opentelemetry SDK not installed; tracing disabled. "
            "Install via: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        _enabled = False
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        except ImportError:  # pragma: no cover
            logger.warning("opentelemetry-exporter-otlp not installed; skipping exporter")
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("koa")
    _enabled = True
    logger.info("OpenTelemetry tracing enabled (service=%s)", service_name)


def get_tracer() -> Optional[Any]:
    """Return the configured tracer or ``None`` if tracing is disabled."""
    return _tracer


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[Optional[Any]]:
    """Start a span named ``name`` with the given attributes.

    Yields the span object (or ``None`` if tracing is disabled) so callers
    can add events/attributes inline.
    """
    if not _enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:  # type: ignore[attr-defined]
        for k, v in attributes.items():
            try:
                span.set_attribute(k, v)
            except Exception:  # pragma: no cover
                pass
        yield span
