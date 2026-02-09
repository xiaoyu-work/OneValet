"""
FlowAgent Hook Manager - Central manager for all hooks

This module provides:
- HookManager: Register and execute hooks
- Decorator for automatic hook execution
"""

import asyncio
from datetime import datetime
from functools import wraps
from typing import Dict, Any, List, Optional, Callable, Awaitable, TypeVar, Union
import yaml

from .models import (
    HookType,
    HookPhase,
    HookConfig,
    HookContext,
    HookResult,
)
from .handlers import (
    HookHandler,
    LoggingHook,
    MetricsHook,
    TracingHook,
    RateLimitingHook,
    CustomHook,
)


class HookExecutionError(Exception):
    """Raised when hook execution fails and should_proceed is False"""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class HookManager:
    """
    Central manager for all hooks.

    Registers and executes hooks in order for each phase.
    Supports loading configuration from YAML.

    Example:
        manager = HookManager()
        manager.register(LoggingHook(HookConfig(hook_type=HookType.LOGGING)))
        manager.register(MetricsHook(HookConfig(hook_type=HookType.METRICS)))

        # Execute hooks
        context = HookContext(agent_id="a1", agent_type="Test", user_id="u1", phase=HookPhase.PRE_EXECUTE)
        results = await manager.execute_pre(context)
    """

    def __init__(self):
        self._hooks: Dict[HookType, HookHandler] = {}
        self._hook_order: List[HookType] = [
            HookType.RATE_LIMITING,  # Check rate limits first
            HookType.TRACING,        # Start tracing
            HookType.LOGGING,        # Log start
            HookType.METRICS,        # Record metrics
            HookType.CUSTOM,         # Custom hooks last
        ]

    def register(self, hook: HookHandler) -> None:
        """Register a hook handler"""
        self._hooks[hook.hook_type] = hook

    def unregister(self, hook_type: HookType) -> Optional[HookHandler]:
        """Unregister a hook handler"""
        return self._hooks.pop(hook_type, None)

    def get_hook(self, hook_type: HookType) -> Optional[HookHandler]:
        """Get a registered hook handler"""
        return self._hooks.get(hook_type)

    def is_enabled(self, hook_type: HookType) -> bool:
        """Check if a hook type is enabled"""
        hook = self._hooks.get(hook_type)
        return hook is not None and hook.enabled

    async def execute_pre(self, context: HookContext) -> List[HookResult]:
        """Execute pre-execute hooks in order"""
        context.phase = HookPhase.PRE_EXECUTE
        results = []

        for hook_type in self._hook_order:
            hook = self._hooks.get(hook_type)
            if hook and hook.enabled:
                try:
                    result = await hook.on_pre_execute(context)
                    results.append(result)

                    # Check if we should proceed
                    if not result.should_proceed:
                        raise HookExecutionError(
                            result.error or "Hook blocked execution",
                            retry_after=result.retry_after
                        )
                except HookExecutionError:
                    raise
                except Exception as e:
                    results.append(HookResult(
                        hook_type=hook_type,
                        success=False,
                        error=str(e)
                    ))

        return results

    async def execute_post(self, context: HookContext) -> List[HookResult]:
        """Execute post-execute hooks in reverse order"""
        context.phase = HookPhase.POST_EXECUTE
        results = []

        # Execute in reverse order for proper unwinding
        for hook_type in reversed(self._hook_order):
            hook = self._hooks.get(hook_type)
            if hook and hook.enabled:
                try:
                    result = await hook.on_post_execute(context)
                    results.append(result)
                except Exception as e:
                    results.append(HookResult(
                        hook_type=hook_type,
                        success=False,
                        error=str(e)
                    ))

        return results

    async def execute_error(self, context: HookContext) -> List[HookResult]:
        """Execute error hooks in reverse order"""
        context.phase = HookPhase.ON_ERROR
        results = []

        for hook_type in reversed(self._hook_order):
            hook = self._hooks.get(hook_type)
            if hook and hook.enabled:
                try:
                    result = await hook.on_error(context)
                    results.append(result)
                except Exception as e:
                    results.append(HookResult(
                        hook_type=hook_type,
                        success=False,
                        error=str(e)
                    ))

        return results

    def load_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Load hooks configuration from dictionary.

        Expected format:
        {
            "logging": true,
            "metrics": {"enabled": true, "log_level": "DEBUG"},
            "tracing": false,
            "rate_limiting": {
                "enabled": true,
                "settings": {
                    "max_requests_per_minute": 30
                }
            }
        }
        """
        hook_mapping = {
            "logging": (HookType.LOGGING, LoggingHook),
            "metrics": (HookType.METRICS, MetricsHook),
            "tracing": (HookType.TRACING, TracingHook),
            "rate_limiting": (HookType.RATE_LIMITING, RateLimitingHook),
        }

        for key, (hook_type, hook_class) in hook_mapping.items():
            if key in config:
                hook_config = HookConfig.from_dict(config[key], hook_type)
                if hook_config.enabled:
                    self.register(hook_class(hook_config))

    def load_from_yaml(self, yaml_path: str) -> None:
        """Load hooks configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        if "hooks" in config:
            self.load_from_dict(config["hooks"])

    def get_metrics_hook(self) -> Optional[MetricsHook]:
        """Get the metrics hook for querying metrics"""
        hook = self._hooks.get(HookType.METRICS)
        if isinstance(hook, MetricsHook):
            return hook
        return None

    def get_tracing_hook(self) -> Optional[TracingHook]:
        """Get the tracing hook for querying spans"""
        hook = self._hooks.get(HookType.TRACING)
        if isinstance(hook, TracingHook):
            return hook
        return None

    def get_rate_limiting_hook(self) -> Optional[RateLimitingHook]:
        """Get the rate limiting hook for querying state"""
        hook = self._hooks.get(HookType.RATE_LIMITING)
        if isinstance(hook, RateLimitingHook):
            return hook
        return None


# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


def with_hooks(
    hook_manager: HookManager,
    agent_id_arg: str = "agent_id",
    agent_type_arg: str = "agent_type",
    user_id_arg: str = "user_id",
) -> Callable[[F], F]:
    """
    Decorator to automatically execute hooks around a function.

    Example:
        manager = HookManager()

        @with_hooks(manager)
        async def execute_agent(agent_id, agent_type, user_id, message):
            # Agent execution logic
            return result
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract context from arguments
            # Try to get from kwargs first, then positional args
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            def get_arg(name: str) -> Optional[str]:
                if name in kwargs:
                    return kwargs[name]
                try:
                    idx = params.index(name)
                    if idx < len(args):
                        return args[idx]
                except (ValueError, IndexError):
                    pass
                return None

            agent_id = get_arg(agent_id_arg) or "unknown"
            agent_type = get_arg(agent_type_arg) or "unknown"
            user_id = get_arg(user_id_arg) or "unknown"

            # Create context
            context = HookContext(
                agent_id=agent_id,
                agent_type=agent_type,
                user_id=user_id,
                phase=HookPhase.PRE_EXECUTE,
                method_name=func.__name__,
                started_at=datetime.now(),
            )

            # Pre-execute hooks
            await hook_manager.execute_pre(context)

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Update context
                context.completed_at = datetime.now()
                context.output_result = result

                # Post-execute hooks
                await hook_manager.execute_post(context)

                return result

            except Exception as e:
                # Update context with error
                context.completed_at = datetime.now()
                context.error = e
                context.error_type = type(e).__name__

                # Error hooks
                await hook_manager.execute_error(context)

                raise

        return wrapper  # type: ignore

    return decorator


class HookableAgent:
    """
    Mixin class that adds hook support to agents.

    Inherit from this class to automatically execute hooks
    on agent method calls.

    Example:
        class MyAgent(HookableAgent, BaseAgent):
            def __init__(self, hook_manager: HookManager):
                self.set_hook_manager(hook_manager)

            async def on_running(self, msg):
                # Hooks are automatically executed
                return await self._execute_with_hooks(
                    self._do_running,
                    msg,
                    method_name="on_running"
                )

            async def _do_running(self, msg):
                # Actual logic here
                pass
    """

    _hook_manager: Optional[HookManager] = None

    def set_hook_manager(self, manager: HookManager) -> None:
        """Set the hook manager"""
        self._hook_manager = manager

    async def _execute_with_hooks(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        method_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute a function with hooks"""
        if not self._hook_manager:
            return await func(*args, **kwargs)

        # Get agent info
        agent_id = getattr(self, 'agent_id', 'unknown')
        agent_type = getattr(self, 'agent_type', type(self).__name__)
        user_id = getattr(self, 'user_id', 'unknown')

        # Create context
        context = HookContext(
            agent_id=agent_id,
            agent_type=agent_type,
            user_id=user_id,
            phase=HookPhase.PRE_EXECUTE,
            method_name=method_name or func.__name__,
            started_at=datetime.now(),
            collected_fields=getattr(self, 'collected_fields', {}),
            execution_state=getattr(self, 'execution_state', {}),
        )

        if args:
            context.input_message = args[0]

        # Pre-execute hooks
        await self._hook_manager.execute_pre(context)

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Update context
            context.completed_at = datetime.now()
            context.output_result = result

            # Post-execute hooks
            await self._hook_manager.execute_post(context)

            return result

        except HookExecutionError:
            raise

        except Exception as e:
            # Update context with error
            context.completed_at = datetime.now()
            context.error = e
            context.error_type = type(e).__name__

            # Error hooks
            await self._hook_manager.execute_error(context)

            raise


# Global hook manager instance
_global_hook_manager: Optional[HookManager] = None


def get_global_hook_manager() -> HookManager:
    """Get or create the global hook manager"""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def configure_hooks(config: Union[Dict[str, Any], str]) -> HookManager:
    """
    Configure the global hook manager.

    Args:
        config: Dictionary config or path to YAML file

    Returns:
        Configured HookManager
    """
    manager = get_global_hook_manager()

    if isinstance(config, str):
        manager.load_from_yaml(config)
    else:
        manager.load_from_dict(config)

    return manager
