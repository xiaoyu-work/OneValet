"""
Tests for FlowAgent Streaming System.

Tests:
- Stream modes (VALUES, UPDATES, MESSAGES, EVENTS)
- Event types and creation
- StreamBuffer operations
- EventEmitter callbacks
- StreamEngine streaming
"""

import pytest
import asyncio
from datetime import datetime

from flowagents.streaming import (
    # Models
    StreamMode,
    EventType,
    AgentEvent,
    StateChangeEvent,
    MessageChunkEvent,
    ToolCallEvent,
    ToolResultEvent,
    ProgressEvent,
    ErrorEvent,
    # Engine
    StreamEngine,
    StreamBuffer,
    EventEmitter,
)


# ============================================================================
# Test Models
# ============================================================================

class TestStreamMode:
    """Tests for StreamMode enum"""

    def test_stream_modes(self):
        """Test all stream modes exist"""
        assert StreamMode.VALUES == "values"
        assert StreamMode.UPDATES == "updates"
        assert StreamMode.MESSAGES == "messages"
        assert StreamMode.EVENTS == "events"


class TestEventType:
    """Tests for EventType enum"""

    def test_state_events(self):
        """Test state-related event types"""
        assert EventType.STATE_CHANGE == "state_change"
        assert EventType.FIELD_COLLECTED == "field_collected"

    def test_message_events(self):
        """Test message-related event types"""
        assert EventType.MESSAGE_START == "message_start"
        assert EventType.MESSAGE_CHUNK == "message_chunk"
        assert EventType.MESSAGE_END == "message_end"

    def test_tool_events(self):
        """Test tool-related event types"""
        assert EventType.TOOL_CALL_START == "tool_call_start"
        assert EventType.TOOL_RESULT == "tool_result"


class TestAgentEvent:
    """Tests for AgentEvent model"""

    def test_event_creation(self):
        """Test creating an agent event"""
        event = AgentEvent(
            type=EventType.STATE_CHANGE,
            data={"old_status": "initializing", "new_status": "running"},
            agent_id="agent_123",
            agent_type="TestAgent"
        )

        assert event.type == EventType.STATE_CHANGE
        assert event.data["old_status"] == "initializing"
        assert event.agent_id == "agent_123"
        assert event.agent_type == "TestAgent"
        assert isinstance(event.timestamp, datetime)

    def test_event_to_dict(self):
        """Test serializing event to dictionary"""
        event = AgentEvent(
            type=EventType.PROGRESS_UPDATE,
            data={"current": 5, "total": 10},
            agent_id="agent_123"
        )

        data = event.to_dict()

        assert data["type"] == "progress_update"
        assert data["data"]["current"] == 5
        assert data["agent_id"] == "agent_123"
        assert "timestamp" in data

    def test_event_from_dict(self):
        """Test deserializing event from dictionary"""
        data = {
            "type": "state_change",
            "data": {"old_status": "a", "new_status": "b"},
            "timestamp": "2025-01-28T12:00:00",
            "agent_id": "test",
            "sequence": 5
        }

        event = AgentEvent.from_dict(data)

        assert event.type == EventType.STATE_CHANGE
        assert event.data["old_status"] == "a"
        assert event.sequence == 5


class TestSpecializedEvents:
    """Tests for specialized event classes"""

    def test_state_change_event(self):
        """Test StateChangeEvent"""
        event = StateChangeEvent(
            old_status="waiting",
            new_status="running",
            agent_id="agent_1"
        )

        assert event.type == EventType.STATE_CHANGE
        assert event.data["old_status"] == "waiting"
        assert event.data["new_status"] == "running"

    def test_message_chunk_event(self):
        """Test MessageChunkEvent"""
        event = MessageChunkEvent(
            chunk="Hello",
            message_id="msg_123",
            agent_id="agent_1"
        )

        assert event.type == EventType.MESSAGE_CHUNK
        assert event.data["chunk"] == "Hello"
        assert event.data["message_id"] == "msg_123"

    def test_progress_event(self):
        """Test ProgressEvent with auto-calculated percentage"""
        event = ProgressEvent(
            current=50,
            total=100,
            message="Halfway done",
            agent_id="agent_1"
        )

        assert event.type == EventType.PROGRESS_UPDATE
        assert event.data["current"] == 50
        assert event.data["total"] == 100
        assert event.data["percentage"] == 50.0
        assert event.data["message"] == "Halfway done"

    def test_error_event(self):
        """Test ErrorEvent"""
        event = ErrorEvent(
            error="Something went wrong",
            error_type="ValueError",
            recoverable=True,
            agent_id="agent_1"
        )

        assert event.type == EventType.ERROR
        assert event.data["error"] == "Something went wrong"
        assert event.data["error_type"] == "ValueError"
        assert event.data["recoverable"] is True


# ============================================================================
# Test StreamBuffer
# ============================================================================

class TestStreamBuffer:
    """Tests for StreamBuffer"""

    def test_add_and_get(self):
        """Test adding and retrieving events"""
        buffer = StreamBuffer(max_size=100)

        event1 = AgentEvent(type=EventType.STATE_CHANGE, data={"test": 1})
        event2 = AgentEvent(type=EventType.PROGRESS_UPDATE, data={"test": 2})

        buffer.add(event1)
        buffer.add(event2)

        events = buffer.get_all()
        assert len(events) == 2
        assert events[0].sequence == 0
        assert events[1].sequence == 1

    def test_max_size(self):
        """Test buffer respects max size"""
        buffer = StreamBuffer(max_size=3)

        for i in range(5):
            event = AgentEvent(type=EventType.STATE_CHANGE, data={"i": i})
            buffer.add(event)

        events = buffer.get_all()
        assert len(events) == 3
        # Should have last 3 events
        assert events[0].data["i"] == 2
        assert events[2].data["i"] == 4

    def test_get_since(self):
        """Test getting events since sequence number"""
        buffer = StreamBuffer()

        for i in range(5):
            event = AgentEvent(type=EventType.STATE_CHANGE, data={"i": i})
            buffer.add(event)

        events = buffer.get_since(2)
        assert len(events) == 2
        assert events[0].sequence == 3
        assert events[1].sequence == 4

    def test_get_by_type(self):
        """Test filtering events by type"""
        buffer = StreamBuffer()

        buffer.add(AgentEvent(type=EventType.STATE_CHANGE, data={}))
        buffer.add(AgentEvent(type=EventType.PROGRESS_UPDATE, data={}))
        buffer.add(AgentEvent(type=EventType.STATE_CHANGE, data={}))

        state_events = buffer.get_by_type(EventType.STATE_CHANGE)
        assert len(state_events) == 2

        progress_events = buffer.get_by_type(EventType.PROGRESS_UPDATE)
        assert len(progress_events) == 1

    def test_clear(self):
        """Test clearing buffer"""
        buffer = StreamBuffer()

        buffer.add(AgentEvent(type=EventType.STATE_CHANGE, data={}))
        buffer.add(AgentEvent(type=EventType.STATE_CHANGE, data={}))

        assert len(buffer) == 2
        buffer.clear()
        assert len(buffer) == 0


# ============================================================================
# Test EventEmitter
# ============================================================================

class TestEventEmitter:
    """Tests for EventEmitter"""

    @pytest.mark.asyncio
    async def test_emit_to_handler(self):
        """Test emitting event to registered handler"""
        emitter = EventEmitter()
        received_events = []

        async def handler(event):
            received_events.append(event)

        emitter.on(EventType.STATE_CHANGE, handler)

        event = AgentEvent(type=EventType.STATE_CHANGE, data={"test": 1})
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].data["test"] == 1

    @pytest.mark.asyncio
    async def test_emit_only_matching_type(self):
        """Test handler only receives matching event type"""
        emitter = EventEmitter()
        received_events = []

        async def handler(event):
            received_events.append(event)

        emitter.on(EventType.STATE_CHANGE, handler)

        # Emit non-matching event
        event = AgentEvent(type=EventType.PROGRESS_UPDATE, data={})
        await emitter.emit(event)

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_global_handler(self):
        """Test global handler receives all events"""
        emitter = EventEmitter()
        received_events = []

        async def handler(event):
            received_events.append(event)

        emitter.on_any(handler)

        await emitter.emit(AgentEvent(type=EventType.STATE_CHANGE, data={}))
        await emitter.emit(AgentEvent(type=EventType.PROGRESS_UPDATE, data={}))

        assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        """Test removing a handler"""
        emitter = EventEmitter()
        received_events = []

        async def handler(event):
            received_events.append(event)

        emitter.on(EventType.STATE_CHANGE, handler)

        # Emit once
        await emitter.emit(AgentEvent(type=EventType.STATE_CHANGE, data={}))
        assert len(received_events) == 1

        # Remove handler
        emitter.off(EventType.STATE_CHANGE, handler)

        # Emit again
        await emitter.emit(AgentEvent(type=EventType.STATE_CHANGE, data={}))
        assert len(received_events) == 1  # Still 1


# ============================================================================
# Test StreamEngine
# ============================================================================

class TestStreamEngine:
    """Tests for StreamEngine"""

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test emitting events through engine"""
        engine = StreamEngine(agent_id="test_agent", agent_type="TestAgent")

        event = await engine.emit(
            EventType.STATE_CHANGE,
            {"old_status": "a", "new_status": "b"}
        )

        assert event.type == EventType.STATE_CHANGE
        assert event.agent_id == "test_agent"

        # Check buffer
        history = engine.get_history()
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_emit_helpers(self):
        """Test emit helper methods"""
        engine = StreamEngine()

        await engine.emit_state_change("waiting", "running")
        await engine.emit_message_chunk("Hello")
        await engine.emit_tool_call("search", {"query": "test"})
        await engine.emit_progress(5, 10, "Halfway")
        await engine.emit_error("Test error", "TestError")

        history = engine.get_history()
        assert len(history) == 5

        types = [e.type for e in history]
        assert EventType.STATE_CHANGE in types
        assert EventType.MESSAGE_CHUNK in types
        assert EventType.TOOL_CALL_START in types
        assert EventType.PROGRESS_UPDATE in types
        assert EventType.ERROR in types

    @pytest.mark.asyncio
    async def test_stream_events_mode(self):
        """Test streaming in EVENTS mode (all events)"""
        engine = StreamEngine()
        received_events = []
        started = asyncio.Event()

        # Start streaming in background
        async def collect_events():
            started.set()
            async for event in engine.stream(mode=StreamMode.EVENTS):
                received_events.append(event)
                if len(received_events) >= 3:
                    break

        # Run collector with timeout
        task = asyncio.create_task(collect_events())

        # Wait for collector to start
        await started.wait()
        await asyncio.sleep(0.01)  # Give it time to setup

        # Emit events
        await engine.emit_state_change("a", "b")
        await engine.emit_message_chunk("Hello")
        await engine.emit_progress(1, 10)

        # Wait for collector
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pass

        engine.close()

        assert len(received_events) == 3

    @pytest.mark.asyncio
    async def test_stream_messages_mode(self):
        """Test streaming in MESSAGES mode (only message events)"""
        engine = StreamEngine()
        received_events = []
        started = asyncio.Event()

        async def collect_events():
            started.set()
            count = 0
            async for event in engine.stream(mode=StreamMode.MESSAGES):
                received_events.append(event)
                count += 1
                if count >= 2:
                    break

        task = asyncio.create_task(collect_events())

        # Wait for collector to start
        await started.wait()
        await asyncio.sleep(0.01)

        # Emit mixed events
        await engine.emit_state_change("a", "b")  # Should NOT be received
        await engine.emit(EventType.MESSAGE_START, {"id": "1"})
        await engine.emit_message_chunk("Hello")
        await engine.emit(EventType.MESSAGE_END, {"id": "1"})

        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pass

        engine.close()

        # Should only have message events
        assert len(received_events) == 2
        assert all(e.type in {EventType.MESSAGE_START, EventType.MESSAGE_CHUNK, EventType.MESSAGE_END}
                   for e in received_events)

    @pytest.mark.asyncio
    async def test_stream_with_history(self):
        """Test streaming includes history"""
        engine = StreamEngine()

        # Emit some events first
        await engine.emit_state_change("a", "b")
        await engine.emit_progress(1, 10)

        received_events = []

        async def collect_events():
            async for event in engine.stream(mode=StreamMode.EVENTS, include_history=True):
                received_events.append(event)
                if len(received_events) >= 3:
                    break

        task = asyncio.create_task(collect_events())

        # Emit one more event
        await engine.emit_state_change("b", "c")

        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            pass

        engine.close()

        # Should have history + new event
        assert len(received_events) == 3

    @pytest.mark.asyncio
    async def test_get_history_filters(self):
        """Test history filtering"""
        engine = StreamEngine()

        await engine.emit_state_change("a", "b")
        await engine.emit_progress(1, 10)
        await engine.emit_state_change("b", "c")

        # Get all
        all_events = engine.get_history()
        assert len(all_events) == 3

        # Get by type
        state_events = engine.get_history(event_type=EventType.STATE_CHANGE)
        assert len(state_events) == 2

        # Get since sequence
        recent = engine.get_history(since_sequence=1)
        assert len(recent) == 1

    def test_clear_history(self):
        """Test clearing history"""
        engine = StreamEngine()

        # Manually add to buffer
        engine.buffer.add(AgentEvent(type=EventType.STATE_CHANGE, data={}))
        engine.buffer.add(AgentEvent(type=EventType.STATE_CHANGE, data={}))

        assert len(engine.get_history()) == 2

        engine.clear_history()

        assert len(engine.get_history()) == 0
