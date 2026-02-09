"""
OneValet Hook Decorators - One-click logging and tracing for agents

This module provides easy-to-use decorators for adding observability:
- @logged: Add automatic logging to agent classes
- @traced: Add distributed tracing
- @metered: Add metrics collection

Example:
    @logged
    class MyAgent(StandardAgent):
        async def on_running(self, msg):
            return self.make_result(status=AgentStatus.COMPLETED, raw_message="Done!")

    # All state transitions, method calls, and errors are logged automatically

With options:
    @logged(level="DEBUG", include_inputs=True, include_outputs=True)
    class MyAgent(StandardAgent):
        ...
"""

import logging
import functools
from datetime import datetime
from typing import (
    Dict, Any, List, Optional, Callable, TypeVar, Type, Union
)

from .models import HookType, HookConfig, HookContext, HookPhase, HookResult
from .handlers import LoggingHook, MetricsHook, TracingHook


# Type variables for decorators
T = TypeVar('T')
AgentClass = TypeVar('AgentClass', bound=type)


def logged(
    _cls: Optional[AgentClass] = None,
    *,
    level: str = "INFO",
    include_inputs: bool = False,
    include_outputs: bool = False,
    log_state_changes: bool = True,
    log_field_collection: bool = True,
    logger_name: Optional[str] = None,
) -> Union[AgentClass, Callable[[AgentClass], AgentClass]]:
    """
    Class decorator to add automatic logging to an agent.

    This decorator wraps key agent methods to automatically log:
    - Agent initialization
    - State transitions
    - Field collection
    - Method calls (on_initializing, on_running, etc.)
    - Errors

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        include_inputs: Log input messages
        include_outputs: Log output results
        log_state_changes: Log state transitions
        log_field_collection: Log when fields are collected
        logger_name: Custom logger name (default: onevalet.agents.<class_name>)

    Example:
        @logged
        class GreetingAgent(StandardAgent):
            ...

        @logged(level="DEBUG", include_inputs=True)
        class VerboseAgent(StandardAgent):
            ...
    """

    def decorator(cls: AgentClass) -> AgentClass:
        # Get or create logger
        _logger_name = logger_name or f"onevalet.agents.{cls.__name__}"
        _logger = logging.getLogger(_logger_name)
        _log_level = getattr(logging, level.upper(), logging.INFO)

        # Store original __init__
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # Log initialization
            _logger.log(
                _log_level,
                f"[{cls.__name__}] Initialized agent_id={self.agent_id} user_id={self.user_id}"
            )

            # Store logging config on instance
            self._logging_config = {
                "level": _log_level,
                "include_inputs": include_inputs,
                "include_outputs": include_outputs,
                "log_state_changes": log_state_changes,
                "log_field_collection": log_field_collection,
                "logger": _logger,
            }

        cls.__init__ = new_init

        # Wrap transition_to for state change logging
        if log_state_changes and hasattr(cls, 'transition_to'):
            original_transition = cls.transition_to

            @functools.wraps(original_transition)
            def new_transition(self, new_status):
                old_status = self.status
                result = original_transition(self, new_status)

                if result and hasattr(self, '_logging_config'):
                    cfg = self._logging_config
                    cfg["logger"].log(
                        cfg["level"],
                        f"[{cls.__name__}] State: {old_status.value} -> {new_status.value} "
                        f"(agent_id={self.agent_id})"
                    )

                return result

            cls.transition_to = new_transition

        # Wrap reply for method call logging
        if hasattr(cls, 'reply'):
            original_reply = cls.reply

            @functools.wraps(original_reply)
            async def new_reply(self, msg=None):
                cfg = getattr(self, '_logging_config', {})
                logger_inst = cfg.get("logger", _logger)
                log_lvl = cfg.get("level", _log_level)

                # Log input
                if cfg.get("include_inputs") and msg:
                    input_text = msg.get_text() if hasattr(msg, 'get_text') else str(msg)
                    if len(input_text) > 200:
                        input_text = input_text[:200] + "..."
                    logger_inst.log(
                        log_lvl,
                        f"[{cls.__name__}] Input: {input_text} (agent_id={self.agent_id})"
                    )

                start_time = datetime.now()

                try:
                    result = await original_reply(self, msg)

                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                    # Log output
                    if cfg.get("include_outputs") and result:
                        output_text = result.raw_message if hasattr(result, 'raw_message') else str(result)
                        if len(output_text) > 200:
                            output_text = output_text[:200] + "..."
                        logger_inst.log(
                            log_lvl,
                            f"[{cls.__name__}] Output: {output_text} "
                            f"(agent_id={self.agent_id}, duration={duration_ms:.1f}ms)"
                        )
                    else:
                        logger_inst.log(
                            log_lvl,
                            f"[{cls.__name__}] Completed status={result.status.value if hasattr(result, 'status') else 'unknown'} "
                            f"(agent_id={self.agent_id}, duration={duration_ms:.1f}ms)"
                        )

                    return result

                except Exception as e:
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                    logger_inst.error(
                        f"[{cls.__name__}] Error: {type(e).__name__}: {e} "
                        f"(agent_id={self.agent_id}, duration={duration_ms:.1f}ms)",
                        exc_info=True
                    )
                    raise

            cls.reply = new_reply

        # Wrap _extract_and_collect_fields for field collection logging
        if log_field_collection and hasattr(cls, '_extract_and_collect_fields'):
            original_extract = cls._extract_and_collect_fields

            @functools.wraps(original_extract)
            async def new_extract(self, user_input):
                fields_before = set(self.collected_fields.keys())

                await original_extract(self, user_input)

                fields_after = set(self.collected_fields.keys())
                new_fields = fields_after - fields_before

                if new_fields and hasattr(self, '_logging_config'):
                    cfg = self._logging_config
                    for field_name in new_fields:
                        value = self.collected_fields[field_name]
                        # Truncate value for logging
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:50] + "..."
                        cfg["logger"].log(
                            cfg["level"],
                            f"[{cls.__name__}] Field collected: {field_name}={value_str} "
                            f"(agent_id={self.agent_id})"
                        )

            cls._extract_and_collect_fields = new_extract

        return cls

    # Handle both @logged and @logged() syntax
    if _cls is None:
        return decorator
    else:
        return decorator(_cls)


def traced(
    _cls: Optional[AgentClass] = None,
    *,
    service_name: Optional[str] = None,
    sample_rate: float = 1.0,
) -> Union[AgentClass, Callable[[AgentClass], AgentClass]]:
    """
    Class decorator to add distributed tracing to an agent.

    Creates spans for agent execution with attributes like
    agent_id, user_id, status, etc.

    Args:
        service_name: Service name for traces
        sample_rate: Sampling rate (0.0 - 1.0)

    Example:
        @traced
        class MyAgent(StandardAgent):
            ...

        @traced(service_name="my-service", sample_rate=0.5)
        class SampledAgent(StandardAgent):
            ...
    """

    def decorator(cls: AgentClass) -> AgentClass:
        import uuid

        _service = service_name or f"onevalet.{cls.__name__}"

        # Store original reply
        original_reply = cls.reply

        @functools.wraps(original_reply)
        async def new_reply(self, msg=None):
            import random

            # Check sample rate
            if random.random() > sample_rate:
                return await original_reply(self, msg)

            # Create trace context
            trace_id = uuid.uuid4().hex
            span_id = uuid.uuid4().hex[:16]

            # Store trace context on instance
            self._trace_context = {
                "trace_id": trace_id,
                "span_id": span_id,
                "service": _service,
                "start_time": datetime.now(),
            }

            try:
                result = await original_reply(self, msg)

                # Complete span
                end_time = datetime.now()
                duration = (end_time - self._trace_context["start_time"]).total_seconds()

                # Log trace (could be exported to tracing backend)
                logging.getLogger("onevalet.tracing").debug(
                    f"Trace completed: trace_id={trace_id} span_id={span_id} "
                    f"agent={cls.__name__} duration={duration:.3f}s "
                    f"status={result.status.value if hasattr(result, 'status') else 'unknown'}"
                )

                return result

            except Exception as e:
                # Log error in span
                end_time = datetime.now()
                duration = (end_time - self._trace_context["start_time"]).total_seconds()

                logging.getLogger("onevalet.tracing").error(
                    f"Trace error: trace_id={trace_id} span_id={span_id} "
                    f"agent={cls.__name__} duration={duration:.3f}s "
                    f"error={type(e).__name__}: {e}"
                )
                raise

        cls.reply = new_reply

        return cls

    if _cls is None:
        return decorator
    else:
        return decorator(_cls)


def metered(
    _cls: Optional[AgentClass] = None,
    *,
    track_tokens: bool = True,
    track_latency: bool = True,
    track_errors: bool = True,
) -> Union[AgentClass, Callable[[AgentClass], AgentClass]]:
    """
    Class decorator to add metrics collection to an agent.

    Collects metrics like:
    - Invocation count
    - Success/error rates
    - Latency percentiles
    - Token usage

    Args:
        track_tokens: Track token usage
        track_latency: Track latency
        track_errors: Track error rates

    Example:
        @metered
        class MyAgent(StandardAgent):
            ...

        @metered(track_tokens=True)
        class TrackedAgent(StandardAgent):
            ...
    """

    def decorator(cls: AgentClass) -> AgentClass:

        # Initialize metrics storage on class
        if not hasattr(cls, '_metrics'):
            cls._metrics = {
                "invocations": 0,
                "successes": 0,
                "errors": 0,
                "total_latency_ms": 0,
                "total_tokens": 0,
            }

        original_reply = cls.reply

        @functools.wraps(original_reply)
        async def new_reply(self, msg=None):
            start_time = datetime.now()
            cls._metrics["invocations"] += 1

            try:
                result = await original_reply(self, msg)

                if track_latency:
                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    cls._metrics["total_latency_ms"] += latency_ms

                cls._metrics["successes"] += 1

                return result

            except Exception as e:
                if track_errors:
                    cls._metrics["errors"] += 1

                if track_latency:
                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    cls._metrics["total_latency_ms"] += latency_ms

                raise

        cls.reply = new_reply

        # Add metrics getter
        @classmethod
        def get_metrics(cls_self) -> Dict[str, Any]:
            """Get collected metrics for this agent class."""
            m = cls_self._metrics
            invocations = m["invocations"]
            return {
                "invocations": invocations,
                "successes": m["successes"],
                "errors": m["errors"],
                "success_rate": m["successes"] / invocations if invocations > 0 else 0,
                "error_rate": m["errors"] / invocations if invocations > 0 else 0,
                "avg_latency_ms": m["total_latency_ms"] / invocations if invocations > 0 else 0,
                "total_tokens": m["total_tokens"],
            }

        cls.get_metrics = get_metrics

        return cls

    if _cls is None:
        return decorator
    else:
        return decorator(_cls)


# Convenience combinations
def observable(
    _cls: Optional[AgentClass] = None,
    *,
    log_level: str = "INFO",
    include_inputs: bool = False,
    include_outputs: bool = False,
    trace: bool = True,
    meter: bool = True,
) -> Union[AgentClass, Callable[[AgentClass], AgentClass]]:
    """
    Combined decorator for full observability (logging + tracing + metrics).

    Example:
        @observable
        class MyAgent(StandardAgent):
            ...

        @observable(log_level="DEBUG", include_inputs=True)
        class VerboseAgent(StandardAgent):
            ...
    """

    def decorator(cls: AgentClass) -> AgentClass:
        # Apply decorators in order
        result_cls = logged(
            level=log_level,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
        )(cls)

        if trace:
            result_cls = traced()(result_cls)

        if meter:
            result_cls = metered()(result_cls)

        return result_cls

    if _cls is None:
        return decorator
    else:
        return decorator(_cls)
