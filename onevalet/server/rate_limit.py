"""Simple in-memory rate limiter for API endpoints."""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter keyed by client identifier.

    Args:
        requests_per_minute: Max requests per minute per client.
        requests_per_hour: Max requests per hour per client.
    """

    def __init__(
        self,
        requests_per_minute: int = 20,
        requests_per_hour: int = 200,
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._minute_windows: Dict[str, List[float]] = defaultdict(list)
        self._hour_windows: Dict[str, List[float]] = defaultdict(list)

    def check(self, client_id: str) -> Tuple[bool, dict]:
        """Check if request is allowed. Returns (allowed, info_dict)."""
        now = time.monotonic()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Clean old entries
        self._minute_windows[client_id] = [
            t for t in self._minute_windows[client_id] if t > minute_ago
        ]
        self._hour_windows[client_id] = [
            t for t in self._hour_windows[client_id] if t > hour_ago
        ]

        minute_count = len(self._minute_windows[client_id])
        hour_count = len(self._hour_windows[client_id])

        if minute_count >= self.rpm:
            return False, {"reason": "rate_limited", "retry_after": 60, "limit": "per_minute"}
        if hour_count >= self.rph:
            return False, {"reason": "rate_limited", "retry_after": 3600, "limit": "per_hour"}

        # Record request
        self._minute_windows[client_id].append(now)
        self._hour_windows[client_id].append(now)

        return True, {
            "remaining_minute": self.rpm - minute_count - 1,
            "remaining_hour": self.rph - hour_count - 1,
        }
