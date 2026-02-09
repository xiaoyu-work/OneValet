"""
Tests for FlowAgent Hook System.

Tests:
- Hook configuration
- Logging hook
- Metrics hook
- Tracing hook
- Rate limiting hook
- Hook manager
- Decorator and mixin
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from flowagents.hooks import (
    # Models
    HookType,
    HookPhase,
    HookConfig,
    HookContext,
    HookResult,
    MetricsData,
    TracingSpan,
    RateLimitConfig,
    RateLimitState,
    # Handlers
    HookHandler,
    LoggingHook,
    MetricsHook,
    TracingHook,
    RateLimitingHook,
    CustomHook,
    # Manager
    HookManager,
    HookExecutionError,
    HookableAgent,
    with_hooks,
    configure_hooks,
)


# ============================================================================
# Test Models
# ============================================================================

class TestHookConfig:
    """Tests for HookConfig"""

    def test_from_bool(self):
        """Test creating config from boolean"""
        config = HookConfig.from_dict(True, HookType.LOGGING)

        assert config.enabled is True
        assert config.hook_type == HookType.LOGGING

    def test_from_dict(self):
        """Test creating config from dictionary"""
        config = HookConfig.from_dict({
            "enabled": True,
            "log_level": "DEBUG",
            "include_inputs": False,
        }, HookType.METRICS)

        assert config.enabled is True
        assert config.log_level == "DEBUG"
        assert config.include_inputs is False


class TestHookContext:
    """Tests for HookContext"""

    def test_create_context(self):
        """Test creating hook context"""
        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        assert ctx.agent_id == "agent_1"
        assert ctx.phase == HookPhase.PRE_EXECUTE

    def test_duration_calculation(self):
        """Test duration calculation"""
        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.POST_EXECUTE,
            started_at=datetime.now(),
        )
        ctx.completed_at = ctx.started_at + timedelta(milliseconds=100)

        assert ctx.duration_ms is not None
        assert 99 < ctx.duration_ms < 101

    def test_to_dict(self):
        """Test context serialization"""
        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
            status="running",
        )

        d = ctx.to_dict()

        assert d["agent_id"] == "agent_1"
        assert d["phase"] == "pre_execute"
        assert d["status"] == "running"


class TestMetricsData:
    """Tests for MetricsData"""

    def test_record_invocation(self):
        """Test recording invocations"""
        metrics = MetricsData(agent_type="Test", user_id="user_1")

        metrics.record_invocation(100, success=True)
        metrics.record_invocation(200, success=True)
        metrics.record_invocation(50, success=False)

        assert metrics.invocation_count == 3
        assert metrics.success_count == 2
        assert metrics.error_count == 1
        assert metrics.avg_duration_ms == 350 / 3
        assert metrics.min_duration_ms == 50
        assert metrics.max_duration_ms == 200

    def test_record_with_tokens(self):
        """Test recording with token usage"""
        metrics = MetricsData(agent_type="Test", user_id="user_1")

        metrics.record_invocation(
            100,
            success=True,
            tokens={"total": 100, "prompt": 80, "completion": 20},
            cost=0.01
        )

        assert metrics.total_tokens == 100
        assert metrics.prompt_tokens == 80
        assert metrics.completion_tokens == 20
        assert metrics.total_cost == 0.01


class TestTracingSpan:
    """Tests for TracingSpan"""

    def test_create_span(self):
        """Test creating a span"""
        span = TracingSpan(
            span_id="span_1",
            trace_id="trace_1",
            name="test.operation",
        )

        assert span.span_id == "span_1"
        assert span.status == "ok"

    def test_set_attributes(self):
        """Test setting span attributes"""
        span = TracingSpan(span_id="s1", trace_id="t1")

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 123)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123

    def test_add_event(self):
        """Test adding events"""
        span = TracingSpan(span_id="s1", trace_id="t1")

        span.add_event("test_event", {"data": "value"})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "test_event"

    def test_end_span(self):
        """Test ending a span"""
        span = TracingSpan(span_id="s1", trace_id="t1")

        span.end(status="error", error_message="Something failed")

        assert span.end_time is not None
        assert span.status == "error"
        assert span.error_message == "Something failed"

    def test_to_dict(self):
        """Test span serialization"""
        span = TracingSpan(span_id="s1", trace_id="t1", name="test")
        span.end()

        d = span.to_dict()

        assert d["span_id"] == "s1"
        assert d["trace_id"] == "t1"
        assert d["name"] == "test"
        assert d["duration_ms"] is not None


class TestRateLimitState:
    """Tests for RateLimitState"""

    def test_check_and_update(self):
        """Test rate limit checking"""
        config = RateLimitConfig(max_requests_per_minute=5)
        state = RateLimitState(user_id="user_1")

        # First 5 requests should succeed
        for i in range(5):
            allowed, _ = state.check_and_update(config)
            assert allowed is True

        # 6th request should fail
        allowed, retry_after = state.check_and_update(config)
        assert allowed is False
        assert retry_after is not None


# ============================================================================
# Test Handlers
# ============================================================================

class TestLoggingHook:
    """Tests for LoggingHook"""

    @pytest.mark.asyncio
    async def test_pre_execute(self):
        """Test logging on pre-execute"""
        config = HookConfig(hook_type=HookType.LOGGING, enabled=True)
        hook = LoggingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
            input_message="Hello",
        )

        result = await hook.on_pre_execute(ctx)

        assert result.success is True
        assert result.hook_type == HookType.LOGGING

    @pytest.mark.asyncio
    async def test_post_execute(self):
        """Test logging on post-execute"""
        config = HookConfig(hook_type=HookType.LOGGING, enabled=True)
        hook = LoggingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.POST_EXECUTE,
            started_at=datetime.now(),
            status="completed",
        )
        ctx.completed_at = ctx.started_at + timedelta(milliseconds=50)

        result = await hook.on_post_execute(ctx)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_on_error(self):
        """Test logging on error"""
        config = HookConfig(hook_type=HookType.LOGGING, enabled=True)
        hook = LoggingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.ON_ERROR,
            error=ValueError("Test error"),
            error_type="ValueError",
        )

        result = await hook.on_error(ctx)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_disabled_hook(self):
        """Test disabled hook doesn't log"""
        config = HookConfig(hook_type=HookType.LOGGING, enabled=False)
        hook = LoggingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        result = await hook.on_pre_execute(ctx)

        assert result.success is True


class TestMetricsHook:
    """Tests for MetricsHook"""

    @pytest.mark.asyncio
    async def test_record_metrics(self):
        """Test recording metrics"""
        config = HookConfig(hook_type=HookType.METRICS, enabled=True)
        hook = MetricsHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.POST_EXECUTE,
            started_at=datetime.now(),
        )
        ctx.completed_at = ctx.started_at + timedelta(milliseconds=100)

        result = await hook.on_post_execute(ctx)

        assert result.success is True
        assert "duration_ms" in result.data

        # Check metrics were recorded
        metrics = hook.get_metrics(agent_type="TestAgent", user_id="user_1")
        assert len(metrics) == 1

    @pytest.mark.asyncio
    async def test_global_metrics(self):
        """Test global metrics aggregation"""
        config = HookConfig(hook_type=HookType.METRICS, enabled=True)
        hook = MetricsHook(config)

        # Record from multiple users
        for user_id in ["user_1", "user_2", "user_3"]:
            ctx = HookContext(
                agent_id=f"agent_{user_id}",
                agent_type="TestAgent",
                user_id=user_id,
                phase=HookPhase.POST_EXECUTE,
                started_at=datetime.now(),
            )
            ctx.completed_at = ctx.started_at + timedelta(milliseconds=100)
            await hook.on_post_execute(ctx)

        global_metrics = hook.get_global_metrics()

        assert "TestAgent" in global_metrics
        assert global_metrics["TestAgent"].invocation_count == 3


class TestTracingHook:
    """Tests for TracingHook"""

    @pytest.mark.asyncio
    async def test_create_span(self):
        """Test span creation on pre-execute"""
        config = HookConfig(hook_type=HookType.TRACING, enabled=True)
        hook = TracingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
            method_name="on_running",
        )

        result = await hook.on_pre_execute(ctx)

        assert result.success is True
        assert "span_id" in result.data
        assert "trace_id" in result.data

    @pytest.mark.asyncio
    async def test_complete_span(self):
        """Test span completion on post-execute"""
        config = HookConfig(hook_type=HookType.TRACING, enabled=True)
        hook = TracingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
            started_at=datetime.now(),
        )

        await hook.on_pre_execute(ctx)

        ctx.phase = HookPhase.POST_EXECUTE
        ctx.completed_at = ctx.started_at + timedelta(milliseconds=50)
        ctx.status = "completed"

        result = await hook.on_post_execute(ctx)

        assert result.success is True
        assert "span" in result.data

        # Check completed spans
        spans = hook.get_completed_spans()
        assert len(spans) == 1
        assert spans[0].status == "ok"

    @pytest.mark.asyncio
    async def test_error_span(self):
        """Test span with error"""
        config = HookConfig(hook_type=HookType.TRACING, enabled=True)
        hook = TracingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        await hook.on_pre_execute(ctx)

        ctx.phase = HookPhase.ON_ERROR
        ctx.error = ValueError("Test error")
        ctx.error_type = "ValueError"

        result = await hook.on_error(ctx)

        assert result.success is True

        spans = hook.get_completed_spans()
        assert len(spans) == 1
        assert spans[0].status == "error"


class TestRateLimitingHook:
    """Tests for RateLimitingHook"""

    @pytest.mark.asyncio
    async def test_allow_request(self):
        """Test allowing request under limit"""
        config = HookConfig(
            hook_type=HookType.RATE_LIMITING,
            enabled=True,
            settings={"max_requests_per_minute": 10}
        )
        hook = RateLimitingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        result = await hook.on_pre_execute(ctx)

        assert result.success is True
        assert result.should_proceed is True

    @pytest.mark.asyncio
    async def test_block_request(self):
        """Test blocking request over limit"""
        config = HookConfig(
            hook_type=HookType.RATE_LIMITING,
            enabled=True,
            settings={"max_requests_per_minute": 2}
        )
        hook = RateLimitingHook(config)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        # First 2 requests should succeed
        await hook.on_pre_execute(ctx)
        await hook.on_pre_execute(ctx)

        # 3rd request should fail
        result = await hook.on_pre_execute(ctx)

        assert result.should_proceed is False
        assert result.retry_after is not None

    @pytest.mark.asyncio
    async def test_per_user_limits(self):
        """Test per-user rate limiting"""
        config = HookConfig(
            hook_type=HookType.RATE_LIMITING,
            enabled=True,
            settings={"max_requests_per_minute": 2}
        )
        hook = RateLimitingHook(config)

        # User 1 uses their limit
        ctx1 = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )
        await hook.on_pre_execute(ctx1)
        await hook.on_pre_execute(ctx1)

        # User 2 should still be allowed
        ctx2 = HookContext(
            agent_id="agent_2",
            agent_type="TestAgent",
            user_id="user_2",
            phase=HookPhase.PRE_EXECUTE,
        )
        result = await hook.on_pre_execute(ctx2)

        assert result.should_proceed is True


class TestCustomHook:
    """Tests for CustomHook"""

    @pytest.mark.asyncio
    async def test_custom_handlers(self):
        """Test custom hook handlers"""
        pre_called = []
        post_called = []

        async def custom_pre(ctx: HookContext) -> HookResult:
            pre_called.append(ctx.agent_id)
            return HookResult(hook_type=HookType.CUSTOM)

        async def custom_post(ctx: HookContext) -> HookResult:
            post_called.append(ctx.agent_id)
            return HookResult(hook_type=HookType.CUSTOM)

        config = HookConfig(hook_type=HookType.CUSTOM, enabled=True)
        hook = CustomHook(config, pre_execute=custom_pre, post_execute=custom_post)

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        await hook.on_pre_execute(ctx)
        await hook.on_post_execute(ctx)

        assert "agent_1" in pre_called
        assert "agent_1" in post_called


# ============================================================================
# Test Hook Manager
# ============================================================================

class TestHookManager:
    """Tests for HookManager"""

    @pytest.mark.asyncio
    async def test_register_hooks(self):
        """Test registering hooks"""
        manager = HookManager()

        logging_hook = LoggingHook(HookConfig(hook_type=HookType.LOGGING))
        metrics_hook = MetricsHook(HookConfig(hook_type=HookType.METRICS))

        manager.register(logging_hook)
        manager.register(metrics_hook)

        assert manager.is_enabled(HookType.LOGGING)
        assert manager.is_enabled(HookType.METRICS)
        assert not manager.is_enabled(HookType.TRACING)

    @pytest.mark.asyncio
    async def test_execute_hooks(self):
        """Test executing hooks"""
        manager = HookManager()
        manager.register(LoggingHook(HookConfig(hook_type=HookType.LOGGING)))
        manager.register(MetricsHook(HookConfig(hook_type=HookType.METRICS)))

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
            started_at=datetime.now(),
        )

        pre_results = await manager.execute_pre(ctx)

        assert len(pre_results) == 2

        ctx.completed_at = datetime.now()
        post_results = await manager.execute_post(ctx)

        assert len(post_results) == 2

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_execution(self):
        """Test that rate limiting blocks execution"""
        manager = HookManager()
        manager.register(RateLimitingHook(HookConfig(
            hook_type=HookType.RATE_LIMITING,
            settings={"max_requests_per_minute": 1}
        )))

        ctx = HookContext(
            agent_id="agent_1",
            agent_type="TestAgent",
            user_id="user_1",
            phase=HookPhase.PRE_EXECUTE,
        )

        # First request succeeds
        await manager.execute_pre(ctx)

        # Second request should raise
        with pytest.raises(HookExecutionError) as exc_info:
            await manager.execute_pre(ctx)

        assert exc_info.value.retry_after is not None

    def test_load_from_dict(self):
        """Test loading hooks from dictionary"""
        manager = HookManager()
        manager.load_from_dict({
            "logging": True,
            "metrics": {"enabled": True, "log_level": "DEBUG"},
            "tracing": False,
        })

        assert manager.is_enabled(HookType.LOGGING)
        assert manager.is_enabled(HookType.METRICS)
        assert not manager.is_enabled(HookType.TRACING)

    @pytest.mark.asyncio
    async def test_get_metrics_hook(self):
        """Test getting metrics hook for querying"""
        manager = HookManager()
        manager.register(MetricsHook(HookConfig(hook_type=HookType.METRICS)))

        metrics_hook = manager.get_metrics_hook()

        assert metrics_hook is not None
        assert isinstance(metrics_hook, MetricsHook)


# ============================================================================
# Test Decorator
# ============================================================================

class TestWithHooksDecorator:
    """Tests for with_hooks decorator"""

    @pytest.mark.asyncio
    async def test_decorator_executes_hooks(self):
        """Test that decorator executes hooks"""
        manager = HookManager()
        metrics_hook = MetricsHook(HookConfig(hook_type=HookType.METRICS))
        manager.register(metrics_hook)

        @with_hooks(manager)
        async def test_function(agent_id: str, agent_type: str, user_id: str, data: str) -> str:
            return f"Result: {data}"

        result = await test_function(
            agent_id="a1",
            agent_type="TestAgent",
            user_id="u1",
            data="test"
        )

        assert result == "Result: test"

        # Check metrics were recorded
        metrics = metrics_hook.get_global_metrics()
        assert "TestAgent" in metrics
        assert metrics["TestAgent"].invocation_count == 1

    @pytest.mark.asyncio
    async def test_decorator_handles_errors(self):
        """Test that decorator handles errors properly"""
        manager = HookManager()
        manager.register(MetricsHook(HookConfig(hook_type=HookType.METRICS)))

        @with_hooks(manager)
        async def failing_function(agent_id: str, agent_type: str, user_id: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_function(
                agent_id="a1",
                agent_type="TestAgent",
                user_id="u1"
            )


# ============================================================================
# Test HookableAgent Mixin
# ============================================================================

class TestHookableAgent:
    """Tests for HookableAgent mixin"""

    @pytest.mark.asyncio
    async def test_execute_with_hooks(self):
        """Test executing with hooks via mixin"""

        class TestAgent(HookableAgent):
            def __init__(self):
                self.agent_id = "test_1"
                self.agent_type = "TestAgent"
                self.user_id = "user_1"
                self.collected_fields = {}
                self.execution_state = {}

            async def execute(self, message: str) -> str:
                return await self._execute_with_hooks(
                    self._do_execute,
                    message,
                    method_name="execute"
                )

            async def _do_execute(self, message: str) -> str:
                return f"Processed: {message}"

        manager = HookManager()
        metrics_hook = MetricsHook(HookConfig(hook_type=HookType.METRICS))
        manager.register(metrics_hook)

        agent = TestAgent()
        agent.set_hook_manager(manager)

        result = await agent.execute("Hello")

        assert result == "Processed: Hello"

        # Check metrics
        metrics = metrics_hook.get_global_metrics()
        assert "TestAgent" in metrics


# ============================================================================
# Test Configure Hooks
# ============================================================================

class TestConfigureHooks:
    """Tests for configure_hooks function"""

    def test_configure_from_dict(self):
        """Test configuring hooks from dictionary"""
        manager = configure_hooks({
            "logging": True,
            "metrics": True,
        })

        assert manager.is_enabled(HookType.LOGGING)
        assert manager.is_enabled(HookType.METRICS)


# ============================================================================
# Test Agent Type Filtering
# ============================================================================

class TestAgentTypeFiltering:
    """Tests for agent type filtering in hooks"""

    @pytest.mark.asyncio
    async def test_include_agent_types(self):
        """Test including specific agent types"""
        config = HookConfig(
            hook_type=HookType.LOGGING,
            enabled=True,
            agent_types=["AllowedAgent"]
        )
        hook = LoggingHook(config)

        # Allowed agent
        assert hook.should_apply("AllowedAgent") is True

        # Not allowed agent
        assert hook.should_apply("OtherAgent") is False

    @pytest.mark.asyncio
    async def test_exclude_agent_types(self):
        """Test excluding specific agent types"""
        config = HookConfig(
            hook_type=HookType.LOGGING,
            enabled=True,
            exclude_agent_types=["ExcludedAgent"]
        )
        hook = LoggingHook(config)

        # Excluded agent
        assert hook.should_apply("ExcludedAgent") is False

        # Other agent
        assert hook.should_apply("OtherAgent") is True
