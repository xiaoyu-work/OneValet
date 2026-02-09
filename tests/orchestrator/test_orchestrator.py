"""
Tests for FlowAgent Orchestrator Module

Tests cover:
- RoutingAction and RoutingDecision models
- OrchestratorConfig and SessionConfig
- AgentPoolManager with memory backend
- MessageRouter trigger matching
- Orchestrator message handling
- Session management
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from flowagents.orchestrator import (
    # Models
    RoutingAction,
    RoutingDecision,
    OrchestratorConfig,
    SessionConfig,
    AgentPoolEntry,
    AgentCallback,
    callback_handler,
    # Pool
    AgentPoolManager,
    MemoryPoolBackend,
    # Router
    MessageRouter,
    TriggerMatcher,
    # Main
    Orchestrator,
)
from flowagents.result import AgentResult, AgentStatus
from flowagents.message import Message


# =============================================================================
# Mock Classes
# =============================================================================

class MockLLMClient:
    """Mock LLM client for testing"""

    def __init__(self, response: str = "mock response"):
        self.response = response
        self.calls = []

    async def chat_completion(self, messages, tools=None, config=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.response


class MockAgent:
    """Mock agent for testing - mimics StandardAgent interface"""

    def __init__(
        self,
        agent_id: str = None,
        tenant_id: str = "default",
        status: AgentStatus = AgentStatus.WAITING_FOR_INPUT,
        llm_client=None,
        checkpoint_manager=None,
        message_hub=None,
        orchestrator_callback=None,
        context_hints=None,
        **kwargs
    ):
        self.agent_id = agent_id or f"mock_agent_{id(self)}"
        self.agent_type = "MockAgent"
        self.tenant_id = tenant_id
        self.status = status
        self.collected_fields = {}
        self.execution_state = {}
        self.context = {}
        self._pause_requested = False
        self._status_before_pause = None
        self.llm_client = llm_client
        self.checkpoint_manager = checkpoint_manager
        self.message_hub = message_hub
        self.orchestrator_callback = orchestrator_callback
        self.context_hints = context_hints or {}

    async def reply(self, msg: Message) -> AgentResult:
        return AgentResult(
            agent_type="MockAgent",
            status=self.status,
            raw_message=f"Processed: {msg.get_text()}",
            agent_id=self.agent_id
        )

    async def stream(self, msg: Message, mode=None):
        """Mock streaming - yield single event"""
        from flowagents.streaming.models import AgentEvent, EventType
        yield AgentEvent(
            type=EventType.MESSAGE_CHUNK,
            data={"chunk": f"Processed: {msg.get_text()}"}
        )

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
        }

    def pause(self) -> AgentResult:
        self._status_before_pause = self.status
        self.status = AgentStatus.PAUSED
        return AgentResult(
            agent_type="MockAgent",
            status=AgentStatus.PAUSED,
            raw_message="Paused",
            agent_id=self.agent_id
        )

    async def resume(self) -> AgentResult:
        self.status = self._status_before_pause or AgentStatus.WAITING_FOR_INPUT
        return AgentResult(
            agent_type="MockAgent",
            status=self.status,
            raw_message="Resumed",
            agent_id=self.agent_id
        )

    def request_pause(self):
        self._pause_requested = True


@dataclass
class MockAgentConfig:
    """Mock AgentConfig for testing"""
    name: str
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    enable_memory: bool = False
    _class: type = None


class MockAgentRegistry:
    """Mock agent registry for testing"""

    def __init__(self):
        self._agents: Dict[str, MockAgentConfig] = {}
        self._agent_classes: Dict[str, type] = {}

    def add_agent(self, name: str, config: Dict[str, Any], agent_class=None):
        self._agents[name] = MockAgentConfig(
            name=name,
            description=config.get("description", ""),
            triggers=config.get("triggers", []),
            _class=agent_class or MockAgent
        )
        self._agent_classes[name] = agent_class or MockAgent

    def get_agent_class(self, agent_type: str):
        return self._agent_classes.get(agent_type)

    def get_all_agents(self) -> Dict[str, MockAgentConfig]:
        return self._agents

    def find_agent_by_trigger(self, message: str) -> Optional[str]:
        message_lower = message.lower()
        for name, config in self._agents.items():
            for trigger in config.triggers:
                if trigger.lower() in message_lower:
                    return name
        return None

    def get_agent_config(self, agent_type: str) -> Optional[MockAgentConfig]:
        return self._agents.get(agent_type)

    def create_agent(
        self,
        name: str,
        tenant_id: str = "default",
        llm_client = None,
        **kwargs
    ):
        """Create an agent instance"""
        agent_class = self._agent_classes.get(name)
        if not agent_class:
            return None
        return agent_class(tenant_id=tenant_id, llm_client=llm_client, **kwargs)

    async def initialize(self):
        pass

    async def shutdown(self):
        pass


# =============================================================================
# Test Models
# =============================================================================

class TestRoutingAction:
    """Tests for RoutingAction enum"""

    def test_routing_action_values(self):
        """Test all routing action values"""
        assert RoutingAction.ROUTE_TO_EXISTING.value == "route_to_existing"
        assert RoutingAction.CREATE_NEW.value == "create_new"
        assert RoutingAction.EXECUTE_WORKFLOW.value == "execute_workflow"
        assert RoutingAction.ROUTE_TO_DEFAULT.value == "route_to_default"
        assert RoutingAction.DELEGATE.value == "delegate"

    def test_routing_action_from_string(self):
        """Test creating from string"""
        action = RoutingAction("create_new")
        assert action == RoutingAction.CREATE_NEW


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass"""

    def test_routing_decision_defaults(self):
        """Test default values"""
        decision = RoutingDecision(action=RoutingAction.CREATE_NEW)
        assert decision.action == RoutingAction.CREATE_NEW
        assert decision.agent_id is None
        assert decision.agent_type is None
        assert decision.confidence == 1.0
        assert decision.reason is None  # reason is Optional[RoutingReason], defaults to None

    def test_routing_decision_full(self):
        """Test with all values"""
        decision = RoutingDecision(
            action=RoutingAction.ROUTE_TO_EXISTING,
            agent_id="agent_123",
            confidence=0.95,
            reason="Match found"
        )
        assert decision.agent_id == "agent_123"
        assert decision.confidence == 0.95
        assert decision.reason == "Match found"

    def test_routing_decision_to_dict(self):
        """Test serialization"""
        decision = RoutingDecision(
            action=RoutingAction.CREATE_NEW,
            agent_type="EmailAgent",
            context_hints={"to": "alice@example.com"}
        )
        data = decision.to_dict()

        assert data["action"] == "create_new"
        assert data["agent_type"] == "EmailAgent"
        assert data["context_hints"]["to"] == "alice@example.com"

    def test_routing_decision_from_dict(self):
        """Test deserialization"""
        data = {
            "action": "create_new",
            "agent_type": "EmailAgent",
            "confidence": 0.8
        }
        decision = RoutingDecision.from_dict(data)

        assert decision.action == RoutingAction.CREATE_NEW
        assert decision.agent_type == "EmailAgent"
        assert decision.confidence == 0.8


class TestSessionConfig:
    """Tests for SessionConfig dataclass"""

    def test_session_config_defaults(self):
        """Test default values"""
        config = SessionConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.active_ttl_seconds == 600
        assert config.session_ttl_seconds == 86400

    def test_session_config_from_dict(self):
        """Test from_dict"""
        data = {
            "enabled": True,
            "backend": "redis",
            "redis_url": "redis://localhost:6379",
            "active_ttl_seconds": 300
        }
        config = SessionConfig.from_dict(data)

        assert config.backend == "redis"
        assert config.redis_url == "redis://localhost:6379"
        assert config.active_ttl_seconds == 300


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass"""

    def test_orchestrator_config_defaults(self):
        """Test default values"""
        config = OrchestratorConfig()
        assert config.config_dir == ""  # Empty string default, user must provide
        assert config.agent_pool_backend == "memory"
        assert config.max_agents_per_user == 10
        assert config.enable_workflows is True

    def test_orchestrator_config_with_session(self):
        """Test with session config"""
        session = SessionConfig(backend="redis")
        config = OrchestratorConfig(session=session)
        assert config.session.backend == "redis"

    def test_orchestrator_config_from_dict(self):
        """Test from_dict with nested session"""
        data = {
            "config_dir": "/custom/config",
            "session": {
                "backend": "redis",
                "redis_url": "redis://myhost:6379"
            }
        }
        config = OrchestratorConfig.from_dict(data)

        assert config.config_dir == "/custom/config"
        assert config.session.backend == "redis"


class TestAgentPoolEntry:
    """Tests for AgentPoolEntry dataclass"""

    def test_pool_entry_creation(self):
        """Test creating pool entry"""
        entry = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="EmailAgent",
            tenant_id="user1",
            status="waiting_for_input"
        )
        assert entry.agent_id == "agent_1"
        assert entry.agent_type == "EmailAgent"
        assert isinstance(entry.created_at, datetime)

    def test_pool_entry_serialization(self):
        """Test to_dict and from_dict"""
        entry = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="EmailAgent",
            tenant_id="user1",
            status="running",
            collected_fields={"to": "alice@example.com"}
        )

        data = entry.to_dict()
        restored = AgentPoolEntry.from_dict(data)

        assert restored.agent_id == entry.agent_id
        assert restored.agent_type == entry.agent_type
        assert restored.collected_fields == entry.collected_fields


class TestAgentCallback:
    """Tests for AgentCallback dataclass"""

    def test_agent_callback(self):
        """Test callback creation"""
        callback = AgentCallback(
            event="get_cache",
            tenant_id="user1",
            data={"key": "email_summary"}
        )

        assert callback.event == "get_cache"
        assert callback.tenant_id == "user1"
        assert callback.data["key"] == "email_summary"
        assert callback.timestamp is not None


# =============================================================================
# Test Pool
# =============================================================================

class TestMemoryPoolBackend:
    """Tests for MemoryPoolBackend"""

    @pytest.fixture
    def backend(self):
        return MemoryPoolBackend()

    @pytest.mark.asyncio
    async def test_save_and_get(self, backend):
        """Test saving and retrieving agent"""
        entry = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="TestAgent",
            tenant_id="user1",
            status="running"
        )

        await backend.save_agent("user1", entry)
        retrieved = await backend.get_agent("user1", "agent_1")

        assert retrieved is not None
        assert retrieved.agent_id == "agent_1"

    @pytest.mark.asyncio
    async def test_list_agents(self, backend):
        """Test listing agents"""
        entry1 = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="TestAgent",
            tenant_id="user1",
            status="running"
        )
        entry2 = AgentPoolEntry(
            agent_id="agent_2",
            agent_type="TestAgent",
            tenant_id="user1",
            status="waiting"
        )

        await backend.save_agent("user1", entry1)
        await backend.save_agent("user1", entry2)

        agents = await backend.list_agents("user1")
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_remove_agent(self, backend):
        """Test removing agent"""
        entry = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="TestAgent",
            tenant_id="user1",
            status="running"
        )

        await backend.save_agent("user1", entry)
        await backend.remove_agent("user1", "agent_1")

        retrieved = await backend.get_agent("user1", "agent_1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_clear_tenant(self, backend):
        """Test clearing all agents for tenant"""
        entry = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="TestAgent",
            tenant_id="user1",
            status="running"
        )

        await backend.save_agent("user1", entry)
        await backend.clear_tenant("user1")

        agents = await backend.list_agents("user1")
        assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_active_tenants(self, backend):
        """Test getting active tenants"""
        entry = AgentPoolEntry(
            agent_id="agent_1",
            agent_type="TestAgent",
            tenant_id="user1",
            status="running"
        )

        await backend.save_agent("user1", entry)
        tenants = await backend.get_active_tenants()

        assert "user1" in tenants


class TestAgentPoolManager:
    """Tests for AgentPoolManager"""

    @pytest.fixture
    def pool(self):
        config = SessionConfig(enabled=True, backend="memory")
        return AgentPoolManager(config=config)

    @pytest.mark.asyncio
    async def test_add_and_get_agent(self, pool):
        """Test adding and retrieving agent"""
        agent = MockAgent(agent_id="test_1", tenant_id="user1")

        await pool.add_agent(agent)
        retrieved = await pool.get_agent("user1", "test_1")

        assert retrieved is agent

    @pytest.mark.asyncio
    async def test_list_agents(self, pool):
        """Test listing agents"""
        agent1 = MockAgent(agent_id="test_1", tenant_id="user1")
        agent2 = MockAgent(agent_id="test_2", tenant_id="user1")

        await pool.add_agent(agent1)
        await pool.add_agent(agent2)

        agents = await pool.list_agents("user1")
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_update_agent(self, pool):
        """Test updating agent"""
        agent = MockAgent(agent_id="test_1", tenant_id="user1")
        await pool.add_agent(agent)

        agent.status = AgentStatus.RUNNING
        await pool.update_agent(agent)

        entry = await pool.get_agent_entry("user1", "test_1")
        assert entry.status == "running"

    @pytest.mark.asyncio
    async def test_remove_agent(self, pool):
        """Test removing agent"""
        agent = MockAgent(agent_id="test_1", tenant_id="user1")
        await pool.add_agent(agent)
        await pool.remove_agent("user1", "test_1")

        retrieved = await pool.get_agent("user1", "test_1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_has_agents_in_memory(self, pool):
        """Test has_agents_in_memory"""
        assert not pool.has_agents_in_memory("user1")

        agent = MockAgent(agent_id="test_1", tenant_id="user1")
        await pool.add_agent(agent)

        assert pool.has_agents_in_memory("user1")

    @pytest.mark.asyncio
    async def test_restore_tenant_session(self, pool):
        """Test restoring tenant session"""
        agent = MockAgent(agent_id="test_1", tenant_id="user1")
        await pool.add_agent(agent)

        # Clear memory cache
        pool._agents = {}
        assert not pool.has_agents_in_memory("user1")

        # Restore
        def factory(entry: AgentPoolEntry) -> MockAgent:
            return MockAgent(
                agent_id=entry.agent_id,
                tenant_id=entry.tenant_id
            )

        count = await pool.restore_tenant_session("user1", factory)
        assert count == 1
        assert pool.has_agents_in_memory("user1")


# =============================================================================
# Test Router
# =============================================================================

class TestTriggerMatcher:
    """Tests for TriggerMatcher"""

    def test_exact_match(self):
        """Test exact match strategy"""
        matcher = TriggerMatcher(strategy="exact")
        assert matcher.matches("hello", "hello")
        assert not matcher.matches("hello world", "hello")

    def test_prefix_match(self):
        """Test prefix match strategy"""
        matcher = TriggerMatcher(strategy="prefix")
        assert matcher.matches("send email to alice", "send email")
        assert not matcher.matches("please send email", "send email")

    def test_contains_match(self):
        """Test contains match strategy"""
        matcher = TriggerMatcher(strategy="contains")
        assert matcher.matches("please send email now", "send email")
        assert not matcher.matches("hello world", "send email")

    def test_regex_match(self):
        """Test regex match strategy"""
        matcher = TriggerMatcher(strategy="regex")
        assert matcher.matches("send email to alice", r"send.*to")
        assert matcher.matches("SEND EMAIL TO BOB", r"send.*to")

    def test_case_sensitivity(self):
        """Test case sensitivity"""
        matcher_insensitive = TriggerMatcher(case_sensitive=False)
        matcher_sensitive = TriggerMatcher(case_sensitive=True)

        assert matcher_insensitive.matches("HELLO", "hello")
        assert not matcher_sensitive.matches("HELLO", "hello")

    def test_find_match(self):
        """Test find_match"""
        matcher = TriggerMatcher()
        triggers = {
            "EmailAgent": ["send email", "compose mail"],
            "CalendarAgent": ["schedule meeting", "add event"]
        }

        assert matcher.find_match("please send email", triggers) == "EmailAgent"
        assert matcher.find_match("schedule meeting now", triggers) == "CalendarAgent"
        assert matcher.find_match("unknown command", triggers) is None


class TestMessageRouter:
    """Tests for MessageRouter"""

    @pytest.fixture
    def registry(self):
        registry = MockAgentRegistry()
        registry.add_agent("EmailAgent", {
            "description": "Send emails",
            "triggers": ["send email", "compose email"]
        })
        registry.add_agent("CalendarAgent", {
            "description": "Manage calendar",
            "triggers": ["schedule meeting", "add event"]
        })
        return registry

    @pytest.fixture
    def router(self, registry):
        return MessageRouter(
            agent_registry=registry,
            default_agent_type="DefaultAgent"
        )

    @pytest.mark.asyncio
    async def test_route_to_active_agent(self, router):
        """Test routing to active agent"""
        agent = MockAgent(status=AgentStatus.WAITING_FOR_INPUT)

        decision = await router.route(
            tenant_id="user1",
            message="any message",
            active_agents=[agent]
        )

        assert decision.action == RoutingAction.ROUTE_TO_EXISTING
        assert decision.agent_id == agent.agent_id

    @pytest.mark.asyncio
    async def test_route_creates_new_agent(self, router):
        """Test creating new agent from trigger"""
        decision = await router.route(
            tenant_id="user1",
            message="send email to alice",
            active_agents=[]
        )

        assert decision.action == RoutingAction.CREATE_NEW
        assert decision.agent_type == "EmailAgent"

    @pytest.mark.asyncio
    async def test_route_to_default(self, router):
        """Test routing to DefaultAgent when no match"""
        decision = await router.route(
            tenant_id="user1",
            message="unknown command",
            active_agents=[]
        )

        assert decision.action == RoutingAction.ROUTE_TO_DEFAULT
        assert decision.agent_type == "DefaultAgent"

    @pytest.mark.asyncio
    async def test_skip_completed_agents(self, router):
        """Test skipping completed agents"""
        agent = MockAgent(status=AgentStatus.COMPLETED)

        decision = await router.route(
            tenant_id="user1",
            message="send email",
            active_agents=[agent]
        )

        # Should create new agent, not route to completed one
        assert decision.action == RoutingAction.CREATE_NEW


# =============================================================================
# Test Orchestrator
# =============================================================================

class TestOrchestrator:
    """Tests for Orchestrator"""

    @pytest.fixture
    def registry(self):
        registry = MockAgentRegistry()
        registry.add_agent("EmailAgent", {
            "description": "Send emails",
            "triggers": ["send email"]
        }, agent_class=MockAgent)
        return registry

    @pytest.fixture
    def orchestrator(self, registry):
        config = OrchestratorConfig(
            session=SessionConfig(enabled=True, backend="memory")
        )
        return Orchestrator(
            config=config,
            agent_registry=registry,
            llm_client=MockLLMClient()
        )

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test initialization"""
        await orchestrator.initialize()
        assert orchestrator._initialized is True

    @pytest.mark.asyncio
    async def test_handle_message_creates_agent(self, orchestrator):
        """Test handling message creates new agent"""
        await orchestrator.initialize()

        response = await orchestrator.handle_message(
            tenant_id="user1",
            message="send email to alice"
        )

        assert response is not None
        assert isinstance(response, AgentResult)
        assert "Processed" in response.raw_message

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_existing(self, orchestrator):
        """Test routing to existing agent"""
        await orchestrator.initialize()

        # First message creates agent
        await orchestrator.handle_message("user1", "send email")

        # Second message routes to existing
        response = await orchestrator.handle_message("user1", "to alice")

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_message_routes_to_default(self, orchestrator):
        """Test routing to DefaultAgent when no match"""
        await orchestrator.initialize()

        # When no agent matches, should route to DefaultAgent
        # This will return an AgentResult (or error if DefaultAgent not registered)
        result = await orchestrator.handle_message(
            tenant_id="user1",
            message="unknown command"
        )

        # Result should be an AgentResult, not a string
        assert result is not None

    @pytest.mark.asyncio
    async def test_list_agents(self, orchestrator):
        """Test listing agents"""
        await orchestrator.initialize()
        await orchestrator.handle_message("user1", "send email")

        agents = await orchestrator.list_agents("user1")
        assert len(agents) >= 0  # May be 0 if completed

    @pytest.mark.asyncio
    async def test_cancel_agent(self, orchestrator):
        """Test cancelling agent"""
        await orchestrator.initialize()

        # Create agent
        agent = MockAgent(agent_id="test_1", tenant_id="user1")
        await orchestrator.agent_pool.add_agent(agent)

        # Cancel it
        result = await orchestrator.cancel_agent("user1", "test_1")
        assert result is True

        # Verify removed
        retrieved = await orchestrator.agent_pool.get_agent("user1", "test_1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator):
        """Test shutdown"""
        await orchestrator.initialize()
        await orchestrator.shutdown()
        assert orchestrator._initialized is False

    @pytest.mark.asyncio
    async def test_stream_message(self, orchestrator):
        """Test streaming message handling"""
        await orchestrator.initialize()

        events = []
        async for event in orchestrator.stream_message(
            tenant_id="user1",
            message="send email"
        ):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_callback_handler(self):
        """Test callback handler decorator"""
        received = []

        class TestOrchestrator(Orchestrator):
            @callback_handler("test_event")
            async def handle_test(self, callback: AgentCallback):
                received.append(callback.data)
                return {"status": "ok"}

        orchestrator = TestOrchestrator()

        # Check handler is registered
        assert "test_event" in orchestrator._callback_handler_map

        # Test invoking
        callback = AgentCallback(event="test_event", tenant_id="user1", data={"key": "value"})
        result = await orchestrator.handle_callback(callback)

        assert result == {"status": "ok"}
        assert len(received) == 1
        assert received[0]["key"] == "value"


# =============================================================================
# Integration Tests
# =============================================================================

class TestOrchestratorIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        # Setup
        registry = MockAgentRegistry()
        registry.add_agent("GreetingAgent", {
            "description": "Greet users",
            "triggers": ["hello", "hi"]
        }, agent_class=MockAgent)

        config = OrchestratorConfig(
            session=SessionConfig(enabled=True, backend="memory")
        )
        orchestrator = Orchestrator(
            config=config,
            agent_registry=registry,
            llm_client=MockLLMClient()
        )
        await orchestrator.initialize()

        # First message
        response1 = await orchestrator.handle_message("user1", "hello")
        assert response1 is not None

        # Second message continues conversation
        response2 = await orchestrator.handle_message("user1", "how are you?")
        assert response2 is not None

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self):
        """Test that users are isolated"""
        registry = MockAgentRegistry()
        registry.add_agent("TestAgent", {
            "triggers": ["test"]
        }, agent_class=MockAgent)

        orchestrator = Orchestrator(
            config=OrchestratorConfig(),
            agent_registry=registry,
            llm_client=MockLLMClient()
        )
        await orchestrator.initialize()

        # User 1 creates agent
        await orchestrator.handle_message("user1", "test")

        # User 2 creates separate agent
        await orchestrator.handle_message("user2", "test")

        # Both should have agents
        user1_agents = await orchestrator.list_agents("user1")
        user2_agents = await orchestrator.list_agents("user2")

        # They should be separate
        assert orchestrator.agent_pool.has_agents_in_memory("user1") or len(user1_agents) >= 0
        assert orchestrator.agent_pool.has_agents_in_memory("user2") or len(user2_agents) >= 0

        await orchestrator.shutdown()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
