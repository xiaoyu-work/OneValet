"""OneValet EventBus — Redis Streams pub/sub for event triggers."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """An event published to the EventBus."""
    source: str  # e.g. "email", "calendar"
    event_type: str  # e.g. "new_email", "event_created"
    data: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    timestamp: Optional[str] = None  # ISO format, auto-set if None


class EventBus:
    """
    Redis Streams event bus. Direct implementation — no abstract interface.

    Usage:
        bus = EventBus(redis_url="redis://localhost:6379")
        await bus.initialize()

        # Subscribe
        await bus.subscribe("email:*", callback)

        # Publish
        await bus.publish(Event(source="email", event_type="new_email", data={...}))
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", stream_prefix: str = "onevalet:events:"):
        self._redis_url = redis_url
        self._stream_prefix = stream_prefix
        self._redis = None  # lazy-initialized redis client
        self._subscriptions: Dict[str, List[Callable]] = {}  # pattern -> [callback]
        self._consumer_group = "onevalet-triggers"
        self._consumer_name = f"consumer-{id(self)}"
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize Redis connection (lazy import redis.asyncio)."""
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("redis package required for EventBus. Install with: pip install redis")
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

    async def close(self) -> None:
        """Close the EventBus."""
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.close()

    async def publish(self, event: Event) -> None:
        """Publish an event to Redis Streams."""
        if not self._redis:
            await self.initialize()

        if not event.timestamp:
            event.timestamp = datetime.now().isoformat()

        stream_name = f"{self._stream_prefix}{event.source}"
        payload = {
            "source": event.source,
            "event_type": event.event_type,
            "data": json.dumps(event.data),
            "tenant_id": event.tenant_id,
            "timestamp": event.timestamp,
        }
        await self._redis.xadd(stream_name, payload)
        logger.debug(f"Published event: {event.source}:{event.event_type}")

    async def subscribe(self, pattern: str, callback: Callable) -> None:
        """Subscribe to events matching a pattern.

        Pattern format: "source:event_type" or "source:*" for all events from a source.
        Callback signature: async (event: Event) -> None
        """
        if pattern not in self._subscriptions:
            self._subscriptions[pattern] = []
        self._subscriptions[pattern].append(callback)
        logger.info(f"Subscribed to event pattern: {pattern}")

        # Start listener if not running
        if not self._running and self._subscriptions:
            await self._start_listener()

    async def unsubscribe(self, pattern: str) -> None:
        """Remove all callbacks for a pattern."""
        self._subscriptions.pop(pattern, None)
        logger.info(f"Unsubscribed from pattern: {pattern}")

    async def _start_listener(self) -> None:
        """Start background task to listen for events."""
        if self._running:
            return
        if not self._redis:
            await self.initialize()
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        """Background loop reading from Redis Streams."""
        while self._running:
            try:
                # Collect unique stream names from subscriptions
                streams = set()
                for pattern in self._subscriptions:
                    source = pattern.split(":")[0]
                    streams.add(f"{self._stream_prefix}{source}")

                if not streams:
                    await asyncio.sleep(1)
                    continue

                # Read from streams
                stream_keys = {s: "$" for s in streams}  # "$" = only new messages
                results = await self._redis.xread(stream_keys, count=100, block=1000)

                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        event = Event(
                            source=fields.get("source", ""),
                            event_type=fields.get("event_type", ""),
                            data=json.loads(fields.get("data", "{}")),
                            tenant_id=fields.get("tenant_id", ""),
                            timestamp=fields.get("timestamp"),
                        )
                        await self._dispatch(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EventBus listener error: {e}")
                await asyncio.sleep(1)

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to matching subscribers."""
        for pattern, callbacks in self._subscriptions.items():
            if self._matches_pattern(pattern, event):
                for callback in callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Event callback error for {pattern}: {e}")

    @staticmethod
    def _matches_pattern(pattern: str, event: Event) -> bool:
        """Check if event matches subscription pattern."""
        parts = pattern.split(":", 1)
        source_pattern = parts[0]
        type_pattern = parts[1] if len(parts) > 1 else "*"

        if source_pattern != "*" and source_pattern != event.source:
            return False
        if type_pattern != "*" and type_pattern != event.event_type:
            return False
        return True
