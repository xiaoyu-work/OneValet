"""
Tests for FlowAgent MsgHub System.

Tests:
- Message broadcasting
- Participant management
- Visibility modes
- Sequential and parallel execution
- Shared context
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from flowagents.msghub import (
    # Models
    Message,
    MessageRole,
    MessageType,
    ParticipantInfo,
    MsgHubConfig,
    VisibilityMode,
    SharedContext,
    MsgHubState,
    HubExecutionResult,
    # Hub
    MsgHub,
    MsgHubError,
    msghub,
)


# ============================================================================
# Mock Classes
# ============================================================================

class MockReply:
    """Mock reply object"""
    def __init__(self, message: str, data: Any = None):
        self.raw_message = message
        self.data = data or message


class MockAgent:
    """Mock agent for testing"""
    def __init__(self, agent_type: str, reply_text: str = None, reply_data: Any = None):
        self.agent_id = f"{agent_type}_{id(self)}"
        self.agent_type = agent_type
        self.collected_fields: Dict[str, Any] = {}
        self._reply_text = reply_text or f"Reply from {agent_type}"
        self._reply_data = reply_data
        self.received_messages: List[Any] = []

    async def reply(self, message: Any) -> MockReply:
        self.received_messages.append(message)
        # Simulate processing
        await asyncio.sleep(0.01)
        return MockReply(self._reply_text, self._reply_data or {"result": self._reply_text})


class ContextAwareAgent(MockAgent):
    """Agent that reads and writes to hub context"""
    def __init__(self, agent_type: str, context_key: str = None, context_value: Any = None):
        super().__init__(agent_type)
        self._context_key = context_key
        self._context_value = context_value

    async def reply(self, message: Any) -> MockReply:
        self.received_messages.append(message)

        # Read hub context if available
        hub_context = self.collected_fields.get('_hub_context', {})
        hub_messages = self.collected_fields.get('_hub_messages', [])

        # Write to collected_fields (will be added to hub context)
        if self._context_key:
            self.collected_fields[self._context_key] = self._context_value

        result = {
            "saw_messages": len(hub_messages),
            "saw_context": hub_context,
            "produced": self._context_value
        }

        return MockReply(f"Processed with {len(hub_messages)} messages", result)


# ============================================================================
# Test Message Model
# ============================================================================

class TestMessage:
    """Tests for Message model"""

    def test_create_message(self):
        """Test creating a message"""
        msg = Message(
            id="msg_1",
            role=MessageRole.USER,
            content="Hello",
            sender_id="user_1"
        )

        assert msg.id == "msg_1"
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.sender_id == "user_1"
        assert msg.message_type == MessageType.TEXT

    def test_message_to_dict(self):
        """Test message serialization"""
        msg = Message(
            id="msg_1",
            role=MessageRole.AGENT,
            content="Hello",
            sender_id="agent_1",
            sender_type="TestAgent",
            data={"key": "value"}
        )

        d = msg.to_dict()

        assert d["id"] == "msg_1"
        assert d["role"] == "agent"
        assert d["content"] == "Hello"
        assert d["data"] == {"key": "value"}

    def test_message_from_dict(self):
        """Test message deserialization"""
        d = {
            "id": "msg_2",
            "role": "system",
            "content": "System message",
            "sender_id": "system",
            "message_type": "text",
            "timestamp": "2024-01-01T12:00:00",
        }

        msg = Message.from_dict(d)

        assert msg.id == "msg_2"
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "System message"


# ============================================================================
# Test SharedContext
# ============================================================================

class TestSharedContext:
    """Tests for SharedContext"""

    def test_set_get(self):
        """Test set and get"""
        ctx = SharedContext()
        ctx.set("key1", "value1", "updater_1")

        assert ctx.get("key1") == "value1"
        assert ctx.updated_by == "updater_1"

    def test_update(self):
        """Test bulk update"""
        ctx = SharedContext()
        ctx.update({"a": 1, "b": 2}, "updater_2")

        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_delete(self):
        """Test delete"""
        ctx = SharedContext()
        ctx.set("key", "value")

        assert ctx.delete("key") is True
        assert ctx.get("key") is None
        assert ctx.delete("nonexistent") is False


# ============================================================================
# Test MsgHub Basic Operations
# ============================================================================

class TestMsgHubBasic:
    """Tests for basic MsgHub operations"""

    @pytest.mark.asyncio
    async def test_create_hub(self):
        """Test creating a hub"""
        hub = MsgHub()

        assert hub.is_active is True
        assert hub.message_count == 0
        assert hub.participant_count == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")

        async with MsgHub(participants=[agent1, agent2]) as hub:
            assert hub.is_active is True
            assert hub.participant_count == 2

        assert hub.is_active is False

    @pytest.mark.asyncio
    async def test_add_participant(self):
        """Test adding participants"""
        hub = MsgHub()
        agent = MockAgent("TestAgent")

        await hub.__aenter__()

        participant = await hub.add_participant(agent)

        assert participant.agent_type == "TestAgent"
        assert participant.is_active is True
        assert hub.participant_count == 1

        await hub.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_remove_participant(self):
        """Test removing participants"""
        agent = MockAgent("TestAgent")
        async with MsgHub(participants=[agent]) as hub:
            assert hub.participant_count == 1

            result = await hub.remove_participant(agent.agent_id)

            assert result is True
            assert hub.participant_count == 0


# ============================================================================
# Test Broadcasting
# ============================================================================

class TestMsgHubBroadcast:
    """Tests for message broadcasting"""

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting a message"""
        async with MsgHub() as hub:
            msg = await hub.broadcast(
                content="Hello everyone",
                sender_id="user_1",
                role=MessageRole.USER
            )

            assert msg.content == "Hello everyone"
            assert hub.message_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_user_message(self):
        """Test broadcasting user message"""
        async with MsgHub() as hub:
            msg = await hub.broadcast_user_message("Hello", user_id="alice")

            assert msg.role == MessageRole.USER
            assert msg.sender_id == "alice"

    @pytest.mark.asyncio
    async def test_broadcast_system_message(self):
        """Test broadcasting system message"""
        async with MsgHub() as hub:
            msg = await hub.broadcast_system_message("System update")

            assert msg.role == MessageRole.SYSTEM
            assert msg.sender_id == "system"

    @pytest.mark.asyncio
    async def test_message_limit(self):
        """Test message limit enforcement"""
        config = MsgHubConfig(max_messages=5)
        async with MsgHub(config=config) as hub:
            for i in range(10):
                await hub.broadcast_user_message(f"Message {i}")

            assert hub.message_count == 5
            # Should have messages 5-9
            messages = hub.get_messages()
            assert messages[0].content == "Message 5"
            assert messages[-1].content == "Message 9"

    @pytest.mark.asyncio
    async def test_message_callback(self):
        """Test message callback"""
        received = []

        async with MsgHub() as hub:
            hub.on_message(lambda msg: received.append(msg))

            await hub.broadcast_user_message("Test 1")
            await hub.broadcast_user_message("Test 2")

        assert len(received) == 2
        assert received[0].content == "Test 1"


# ============================================================================
# Test Visibility Modes
# ============================================================================

class TestVisibilityModes:
    """Tests for visibility modes"""

    @pytest.mark.asyncio
    async def test_visibility_all(self):
        """Test ALL visibility mode"""
        config = MsgHubConfig(visibility_mode=VisibilityMode.ALL)
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")

        async with MsgHub(participants=[agent1, agent2], config=config) as hub:
            await hub.broadcast_user_message("Hello")
            await hub.broadcast(
                content="Reply 1",
                sender_id=agent1.agent_id,
                role=MessageRole.AGENT
            )

            # Agent2 should see all messages
            messages = hub.get_messages(participant_id=agent2.agent_id)
            assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_visibility_sequential(self):
        """Test SEQUENTIAL visibility mode"""
        config = MsgHubConfig(visibility_mode=VisibilityMode.SEQUENTIAL)
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")
        agent3 = MockAgent("Agent3")

        async with MsgHub(participants=[agent1, agent2, agent3], config=config) as hub:
            await hub.broadcast_user_message("Initial")
            await hub.broadcast(
                content="From Agent1",
                sender_id=agent1.agent_id,
                role=MessageRole.AGENT
            )
            await hub.broadcast(
                content="From Agent2",
                sender_id=agent2.agent_id,
                role=MessageRole.AGENT
            )

            # Agent1 only sees user messages
            msg1 = hub.get_messages(participant_id=agent1.agent_id)
            agent1_sees = [m.content for m in msg1]
            assert "Initial" in agent1_sees
            assert "From Agent1" not in agent1_sees  # Can't see own or later messages

            # Agent2 sees user + Agent1
            msg2 = hub.get_messages(participant_id=agent2.agent_id)
            agent2_sees = [m.content for m in msg2]
            assert "Initial" in agent2_sees
            assert "From Agent1" in agent2_sees

            # Agent3 sees all previous agents
            msg3 = hub.get_messages(participant_id=agent3.agent_id)
            agent3_sees = [m.content for m in msg3]
            assert "Initial" in agent3_sees
            assert "From Agent1" in agent3_sees
            assert "From Agent2" in agent3_sees

    @pytest.mark.asyncio
    async def test_can_see_all_false(self):
        """Test participant that can't see messages before joining"""
        config = MsgHubConfig(visibility_mode=VisibilityMode.ALL)
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")

        async with MsgHub(participants=[agent1], config=config) as hub:
            await hub.broadcast_user_message("Before Agent2")

            # Add agent2 without can_see_all
            await hub.add_participant(agent2, can_see_all=False)

            await hub.broadcast_user_message("After Agent2")

            # Agent2 should only see messages after joining
            messages = hub.get_messages(participant_id=agent2.agent_id)
            assert len(messages) == 1
            assert messages[0].content == "After Agent2"


# ============================================================================
# Test Agent Execution
# ============================================================================

class TestMsgHubExecution:
    """Tests for agent execution"""

    @pytest.mark.asyncio
    async def test_execute_single_agent(self):
        """Test executing a single agent"""
        agent = MockAgent("TestAgent", reply_text="Done!")

        async with MsgHub(participants=[agent]) as hub:
            reply = await hub.execute(agent, "Do something")

            assert reply.raw_message == "Done!"
            # Should have 2 messages: user input + agent reply
            assert hub.message_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_context_injection(self):
        """Test that hub context is injected into agent"""
        agent = ContextAwareAgent("ContextAgent")

        async with MsgHub(participants=[agent]) as hub:
            hub.set_context("existing_key", "existing_value")
            await hub.broadcast_user_message("Previous message")

            reply = await hub.execute(agent, "Process with context")

            # Agent should have received hub context
            assert "_hub_messages" in agent.collected_fields
            assert "_hub_context" in agent.collected_fields

    @pytest.mark.asyncio
    async def test_execute_with_context_update(self):
        """Test updating context from agent execution"""
        agent = ContextAwareAgent(
            "ContextAgent",
            context_key="result",
            context_value="agent_produced_value"
        )

        async with MsgHub(participants=[agent]) as hub:
            await hub.execute(
                agent,
                "Process",
                update_context_keys=["result"]
            )

            # Hub context should be updated
            assert hub.get_context("result") == "agent_produced_value"

    @pytest.mark.asyncio
    async def test_execute_sequential(self):
        """Test sequential execution"""
        agent1 = MockAgent("Researcher", reply_text="Found 3 frameworks")
        agent2 = MockAgent("Writer", reply_text="Wrote comparison")
        agent3 = MockAgent("Reviewer", reply_text="Approved")

        async with MsgHub() as hub:
            result = await hub.execute_sequential(
                [agent1, agent2, agent3],
                "Research Python frameworks"
            )

            assert result.status == "completed"
            assert result.total_agents == 3
            assert result.completed_agents == 3
            assert result.failed_agents == 0
            assert len(result.final_messages) > 0

    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        """Test parallel execution"""
        agent1 = MockAgent("Search1")
        agent2 = MockAgent("Search2")
        agent3 = MockAgent("Search3")

        async with MsgHub() as hub:
            result = await hub.execute_parallel(
                [agent1, agent2, agent3],
                "Search query"
            )

            assert result.status == "completed"
            assert result.total_agents == 3
            assert result.completed_agents == 3

    @pytest.mark.asyncio
    async def test_execute_with_failure(self):
        """Test execution with agent failure"""
        class FailingAgent:
            agent_id = "failing_1"
            agent_type = "FailingAgent"

            async def reply(self, message):
                raise ValueError("Agent failed!")

        agent1 = MockAgent("Agent1")
        agent2 = FailingAgent()

        async with MsgHub() as hub:
            result = await hub.execute_parallel([agent1, agent2], "Test")

            assert result.status == "partial"
            assert result.completed_agents == 1
            assert result.failed_agents == 1
            assert len(result.errors) == 1


# ============================================================================
# Test Shared Context
# ============================================================================

class TestMsgHubContext:
    """Tests for shared context"""

    @pytest.mark.asyncio
    async def test_set_get_context(self):
        """Test setting and getting context"""
        async with MsgHub() as hub:
            hub.set_context("key1", "value1")
            hub.set_context("key2", {"nested": "data"})

            assert hub.get_context("key1") == "value1"
            assert hub.get_context("key2") == {"nested": "data"}
            assert hub.get_context("nonexistent") is None

    @pytest.mark.asyncio
    async def test_update_context(self):
        """Test bulk context update"""
        async with MsgHub() as hub:
            hub.update_context({
                "a": 1,
                "b": 2,
                "c": 3
            })

            all_context = hub.get_context()
            assert all_context == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.asyncio
    async def test_context_in_execution_result(self):
        """Test that final context is in execution result"""
        agent = MockAgent("Agent1")

        async with MsgHub() as hub:
            hub.set_context("initial", "value")

            result = await hub.execute_sequential([agent], "Test")

            assert "initial" in result.final_context


# ============================================================================
# Test Context Manager Helper
# ============================================================================

class TestMsgHubContextManager:
    """Tests for msghub context manager helper"""

    @pytest.mark.asyncio
    async def test_msghub_helper(self):
        """Test msghub() helper function"""
        agent = MockAgent("TestAgent")

        async with msghub(participants=[agent]) as hub:
            assert hub.is_active is True
            assert hub.participant_count == 1

            await hub.execute(agent, "Hello")

        assert hub.is_active is False

    @pytest.mark.asyncio
    async def test_msghub_with_config(self):
        """Test msghub() with config"""
        config = MsgHubConfig(
            visibility_mode=VisibilityMode.SEQUENTIAL,
            max_messages=50
        )

        async with msghub(config=config, hub_id="custom_hub") as hub:
            assert hub.hub_id == "custom_hub"
            assert hub.config.visibility_mode == VisibilityMode.SEQUENTIAL


# ============================================================================
# Test Format Context
# ============================================================================

class TestFormatContext:
    """Tests for format_context_for_agent"""

    @pytest.mark.asyncio
    async def test_format_context(self):
        """Test formatting context for agent"""
        agent1 = MockAgent("Agent1")
        agent2 = MockAgent("Agent2")

        async with MsgHub(participants=[agent1, agent2]) as hub:
            await hub.broadcast_user_message("Hello")
            await hub.broadcast(
                content="Agent1 response",
                sender_id=agent1.agent_id,
                sender_type="Agent1",
                role=MessageRole.AGENT
            )
            hub.set_context("key", "value")

            formatted = hub.format_context_for_agent(agent2.agent_id)

            assert "Conversation History" in formatted
            assert "Hello" in formatted
            assert "Agent1 response" in formatted
            assert "Shared Context" in formatted
            assert "key: value" in formatted


# ============================================================================
# Test Error Cases
# ============================================================================

class TestMsgHubErrors:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_broadcast_to_closed_hub(self):
        """Test broadcasting to closed hub raises error"""
        hub = MsgHub()
        await hub.__aenter__()
        await hub.close()

        with pytest.raises(MsgHubError):
            await hub.broadcast_user_message("Hello")

    @pytest.mark.asyncio
    async def test_add_participant_to_closed_hub(self):
        """Test adding participant to closed hub raises error"""
        hub = MsgHub()
        await hub.__aenter__()
        await hub.close()

        agent = MockAgent("TestAgent")
        with pytest.raises(MsgHubError):
            await hub.add_participant(agent)


# ============================================================================
# Test State
# ============================================================================

class TestMsgHubState:
    """Tests for MsgHubState"""

    def test_state_to_dict(self):
        """Test state serialization"""
        state = MsgHubState(hub_id="test_hub")
        state.messages.append(Message(
            id="msg_1",
            role=MessageRole.USER,
            content="Hello",
            sender_id="user_1"
        ))

        d = state.to_dict()

        assert d["hub_id"] == "test_hub"
        assert len(d["messages"]) == 1
        assert d["is_active"] is True

    def test_state_properties(self):
        """Test state properties"""
        state = MsgHubState(hub_id="test_hub")
        state.messages = [
            Message(id="1", role=MessageRole.USER, content="1", sender_id="u"),
            Message(id="2", role=MessageRole.USER, content="2", sender_id="u"),
        ]
        state.participants = {
            "agent_1": ParticipantInfo(agent_id="agent_1", agent_type="Type1"),
            "agent_2": ParticipantInfo(agent_id="agent_2", agent_type="Type2", is_active=False),
        }

        assert state.message_count == 2
        assert state.participant_count == 1  # Only active ones


# ============================================================================
# Test HubExecutionResult
# ============================================================================

class TestHubExecutionResult:
    """Tests for HubExecutionResult"""

    def test_success_rate(self):
        """Test success rate calculation"""
        result = HubExecutionResult(
            hub_id="hub_1",
            status="partial",
            total_agents=4,
            completed_agents=3,
            failed_agents=1
        )

        assert result.success_rate == 0.75

    def test_duration(self):
        """Test duration calculation"""
        result = HubExecutionResult(
            hub_id="hub_1",
            status="completed",
            started_at=datetime.now()
        )
        result.completed_at = result.started_at + timedelta(seconds=5)

        assert result.duration_seconds == 5.0
