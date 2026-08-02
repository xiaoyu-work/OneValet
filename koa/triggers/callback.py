"""Koa Callback Notification — deliver trigger results via HTTP callback."""

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class CallbackNotification:
    """Send notifications via HTTP POST to a callback URL.

    Args:
        callback_url: URL to POST notification payloads to.
        timeout: Request timeout in seconds (default 30).
    """

    def __init__(self, callback_url: str, timeout: int = 30):
        self._callback_url = callback_url
        self._timeout = timeout

    async def send(
        self, tenant_id: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """POST notification to callback URL.

        Returns True on success, False on failure.
        """
        meta = metadata or {}
        payload = {
            "tenant_id": tenant_id,
            "message": message,
            "priority": meta.get("priority", "normal"),
            "category": meta.get("category", "general"),
            "metadata": {
                "task_id": meta.get("task_id", ""),
                "trigger_type": meta.get("trigger_type", ""),
                "source_event": meta.get("source_event", {}),
            },
        }
        # An approval request is useless without the id needed to answer it,
        # so those fields ride along when present rather than being dropped by
        # the fixed shape above.
        for key in ("ask_id", "run_id", "options"):
            if meta.get(key):
                payload["metadata"][key] = meta[key]

        for attempt in range(2):  # 1 initial + 1 retry
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self._callback_url, json=payload)
                    response.raise_for_status()
                logger.info(f"Callback notification sent for tenant {tenant_id}")
                return True
            except httpx.ConnectError as e:
                if attempt == 0:
                    logger.warning(f"Callback connection error (retrying in 2s): {e}")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"Callback connection error after retry for {tenant_id}: {e}")
            except Exception as e:
                logger.error(f"Callback notification failed for {tenant_id}: {e}")
                return False

        return False
