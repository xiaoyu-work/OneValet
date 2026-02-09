"""
OneValet Streaming Engine - Core streaming functionality

This module provides:
- StreamEngine: Main streaming coordinator
- StreamBuffer: Buffer for collecting events before emission
- EventEmitter: Callback-based event distribution
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Dict, Any, List, Optional, Callable, Awaitable,
    AsyncIterator, Set, TypeVar, Generic
)
from collections import deque
import uuid
import logging

from .models import StreamMode, EventType, AgentEvent

logger = logging.getLogger(__name__)


# Type for event handlers
EventHandler = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class StreamBuffer:
    """
    Buffer for collecting events before emission.

    Supports:
    - Event collection with optional size limit
    - Event filtering by type
    - Event retrieval and clearing
    """

    max_size: int = 1000
    events: deque = field(default_factory=lambda: deque(maxlen=1000))
    sequence_counter: int = 0

    def __post_init__(self):
        self.events = deque(maxlen=self.max_size)

    def add(self, event: AgentEvent) -> None:
        """Add an event to the buffer"""
        event.sequence = self.sequence_counter
        self.sequence_counter += 1
        self.events.append(event)

    def get_all(self) -> List[AgentEvent]:
        """Get all events in the buffer"""
        return list(self.events)

    def get_since(self, sequence: int) -> List[AgentEvent]:
        """Get events since a specific sequence number"""
        return [e for e in self.events if e.sequence > sequence]

    def get_by_type(self, event_type: EventType) -> List[AgentEvent]:
        """Get events of a specific type"""
        return [e for e in self.events if e.type == event_type]

    def clear(self) -> None:
        """Clear all events from buffer"""
        self.events.clear()

    def __len__(self) -> int:
        return len(self.events)


class EventEmitter:
    """
    Callback-based event distribution.

    Allows registering handlers for specific event types
    and emitting events to all registered handlers.
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

    def on(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def on_any(self, handler: EventHandler) -> None:
        """Register a handler for all events"""
        self._global_handlers.append(handler)

    def off(self, event_type: EventType, handler: EventHandler) -> bool:
        """Unregister a handler for a specific event type"""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    def off_any(self, handler: EventHandler) -> bool:
        """Unregister a global handler"""
        try:
            self._global_handlers.remove(handler)
            return True
        except ValueError:
            return False

    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to all registered handlers"""
        # Call type-specific handlers
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.warning(
                        f"Event handler error for {event.type}: {e}",
                        exc_info=True
                    )

        # Call global handlers
        for handler in self._global_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.warning(
                    f"Global event handler error: {e}",
                    exc_info=True
                )

    def clear(self) -> None:
        """Remove all handlers"""
        self._handlers.clear()
        self._global_handlers.clear()


class StreamEngine:
    """
    Main streaming coordinator for agent execution.

    Provides:
    - Multi-mode streaming (VALUES, UPDATES, MESSAGES, EVENTS)
    - Event buffering for catch-up scenarios
    - Async iterator interface for streaming
    - Integration with agent execution

    Example usage:
        engine = StreamEngine()

        # Start streaming
        async for event in engine.stream(mode=StreamMode.EVENTS):
            print(event)

        # Or use callbacks
        engine.emitter.on(EventType.STATE_CHANGE, handle_state_change)
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None
    ):
        self.buffer = StreamBuffer(max_size=buffer_size)
        self.emitter = EventEmitter()
        self.agent_id = agent_id
        self.agent_type = agent_type

        # Active stream state
        self._active_streams: Set[str] = set()
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._is_closed = False

    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        **kwargs
    ) -> AgentEvent:
        """
        Emit an event.

        Creates an AgentEvent, adds it to the buffer,
        calls registered handlers, and pushes to active streams.
        """
        event = AgentEvent(
            type=event_type,
            data=data,
            agent_id=kwargs.get("agent_id", self.agent_id),
            agent_type=kwargs.get("agent_type", self.agent_type),
            timestamp=datetime.now(),
        )

        # Add to buffer
        self.buffer.add(event)

        # Emit to handlers
        await self.emitter.emit(event)

        # Push to active stream queues
        for stream_id, queue in self._stream_queues.items():
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Skip if queue is full
                pass

        return event

    async def emit_state_change(
        self,
        old_status: str,
        new_status: str
    ) -> AgentEvent:
        """Emit a state change event"""
        return await self.emit(
            EventType.STATE_CHANGE,
            {"old_status": old_status, "new_status": new_status}
        )

    async def emit_message_chunk(
        self,
        chunk: str,
        message_id: Optional[str] = None
    ) -> AgentEvent:
        """Emit a message chunk event"""
        return await self.emit(
            EventType.MESSAGE_CHUNK,
            {"chunk": chunk, "message_id": message_id}
        )

    async def emit_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> AgentEvent:
        """Emit a tool call start event"""
        return await self.emit(
            EventType.TOOL_CALL_START,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "call_id": call_id or uuid.uuid4().hex[:8]
            }
        )

    async def emit_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        call_id: Optional[str] = None
    ) -> AgentEvent:
        """Emit a tool result event"""
        return await self.emit(
            EventType.TOOL_RESULT,
            {
                "tool_name": tool_name,
                "result": result,
                "success": success,
                "error": error,
                "call_id": call_id
            }
        )

    async def emit_progress(
        self,
        current: int,
        total: int,
        message: Optional[str] = None
    ) -> AgentEvent:
        """Emit a progress update event"""
        percentage = (current / total) * 100 if total > 0 else 0
        return await self.emit(
            EventType.PROGRESS_UPDATE,
            {
                "current": current,
                "total": total,
                "percentage": percentage,
                "message": message
            }
        )

    async def emit_error(
        self,
        error: str,
        error_type: Optional[str] = None,
        recoverable: bool = False
    ) -> AgentEvent:
        """Emit an error event"""
        return await self.emit(
            EventType.ERROR,
            {
                "error": error,
                "error_type": error_type,
                "recoverable": recoverable
            }
        )

    async def stream(
        self,
        mode: StreamMode = StreamMode.EVENTS,
        include_history: bool = False
    ) -> AsyncIterator[AgentEvent]:
        """
        Create an async iterator for streaming events.

        Args:
            mode: Streaming mode (affects which events are yielded)
            include_history: Whether to yield buffered events first

        Yields:
            AgentEvent objects based on the mode
        """
        stream_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        self._active_streams.add(stream_id)
        self._stream_queues[stream_id] = queue

        try:
            # Yield buffered events if requested
            if include_history:
                for event in self.buffer.get_all():
                    if self._should_yield_event(event, mode):
                        yield event

            # Stream live events
            while not self._is_closed:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if self._should_yield_event(event, mode):
                        yield event
                except asyncio.TimeoutError:
                    # Continue loop, checking if closed
                    continue

        finally:
            # Cleanup
            self._active_streams.discard(stream_id)
            del self._stream_queues[stream_id]

    def _should_yield_event(
        self,
        event: AgentEvent,
        mode: StreamMode
    ) -> bool:
        """Determine if an event should be yielded based on mode"""
        if mode == StreamMode.EVENTS:
            # Yield all events
            return True

        elif mode == StreamMode.MESSAGES:
            # Only message-related events
            return event.type in {
                EventType.MESSAGE_START,
                EventType.MESSAGE_CHUNK,
                EventType.MESSAGE_END,
            }

        elif mode == StreamMode.UPDATES:
            # State changes and progress updates
            return event.type in {
                EventType.STATE_CHANGE,
                EventType.FIELD_COLLECTED,
                EventType.PROGRESS_UPDATE,
                EventType.TOOL_CALL_START,
                EventType.TOOL_RESULT,
            }

        elif mode == StreamMode.VALUES:
            # Only state snapshots (state changes with full state)
            return event.type == EventType.STATE_CHANGE

        return False

    def close(self) -> None:
        """Close the stream engine"""
        self._is_closed = True

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        since_sequence: Optional[int] = None
    ) -> List[AgentEvent]:
        """Get buffered event history"""
        if since_sequence is not None:
            events = self.buffer.get_since(since_sequence)
        elif event_type is not None:
            events = self.buffer.get_by_type(event_type)
        else:
            events = self.buffer.get_all()

        return events

    def clear_history(self) -> None:
        """Clear event buffer"""
        self.buffer.clear()


class StreamContext:
    """
    Context manager for streaming within agent execution.

    Provides a convenient way to emit events during agent execution
    with automatic cleanup.
    """

    def __init__(
        self,
        engine: StreamEngine,
        context_type: str = "execution",
        context_id: Optional[str] = None
    ):
        self.engine = engine
        self.context_type = context_type
        self.context_id = context_id or uuid.uuid4().hex[:8]
        self._start_time: Optional[datetime] = None

    async def __aenter__(self) -> "StreamContext":
        self._start_time = datetime.now()
        await self.engine.emit(
            EventType.EXECUTION_START,
            {
                "context_type": self.context_type,
                "context_id": self.context_id,
                "start_time": self._start_time.isoformat()
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds() if self._start_time else 0

        if exc_type is not None:
            await self.engine.emit_error(
                str(exc_val),
                error_type=exc_type.__name__,
                recoverable=False
            )

        await self.engine.emit(
            EventType.EXECUTION_END,
            {
                "context_type": self.context_type,
                "context_id": self.context_id,
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "success": exc_type is None
            }
        )

    async def emit_progress(
        self,
        current: int,
        total: int,
        message: Optional[str] = None
    ) -> AgentEvent:
        """Emit progress within this context"""
        return await self.engine.emit_progress(current, total, message)

    async def emit_message(self, chunk: str) -> AgentEvent:
        """Emit a message chunk within this context"""
        return await self.engine.emit_message_chunk(chunk, self.context_id)
