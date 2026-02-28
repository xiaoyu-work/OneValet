"""CronDeliveryHandler — none/announce/webhook delivery after job execution."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import CronJob, CronRunEntry, DeliveryMode

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 10


@dataclass
class DeliveryResult:
    """Outcome of a delivery attempt."""
    delivered: bool = False
    status: str = "not-requested"  # "delivered" | "not-delivered" | "not-requested"
    error: Optional[str] = None


class CronDeliveryHandler:
    """Handles delivery of cron job results.

    Supports three modes:
    - NONE: no delivery (execution only)
    - ANNOUNCE: channel delivery via existing notification channels (SMS/Push/Callback)
    - WEBHOOK: HTTP POST to a configured URL
    """

    def __init__(
        self,
        notifications: Optional[List[Any]] = None,
    ):
        self._notifications = notifications or []

    async def deliver(
        self,
        job: CronJob,
        result_text: str,
        run_entry: CronRunEntry,
    ) -> DeliveryResult:
        """Deliver job results based on job's delivery config."""
        if not job.delivery or job.delivery.mode == DeliveryMode.NONE:
            return DeliveryResult(status="not-requested")

        try:
            if job.delivery.mode == DeliveryMode.ANNOUNCE:
                await self._deliver_announce(job, result_text)
                return DeliveryResult(delivered=True, status="delivered")
            elif job.delivery.mode == DeliveryMode.WEBHOOK:
                await self._deliver_webhook(job, result_text, run_entry)
                return DeliveryResult(delivered=True, status="delivered")
            else:
                return DeliveryResult(status="not-requested")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Delivery failed for job {job.id}: {error_msg}")
            if job.delivery.best_effort:
                return DeliveryResult(delivered=False, status="not-delivered", error=error_msg)
            raise

    async def _deliver_announce(self, job: CronJob, result_text: str) -> None:
        """Route to existing notification channels based on delivery.channel."""
        if not self._notifications:
            raise RuntimeError("No notification channels configured for announce delivery")

        metadata: Dict[str, Any] = {
            "cron_job_id": job.id,
            "cron_job_name": job.name,
            "delivery_mode": "announce",
        }
        if job.delivery and job.delivery.channel:
            metadata["channel"] = job.delivery.channel
        if job.delivery and job.delivery.to:
            metadata["to"] = job.delivery.to

        message = f"[Cron: {job.name}] {result_text}"

        delivered = False
        for channel in self._notifications:
            try:
                await channel.send(job.user_id, message, metadata)
                delivered = True
            except Exception as e:
                logger.warning(f"Announce delivery failed via {type(channel).__name__}: {e}")

        if not delivered:
            raise RuntimeError("All notification channels failed")

    async def _deliver_webhook(
        self,
        job: CronJob,
        result_text: str,
        run_entry: CronRunEntry,
    ) -> None:
        """POST results to webhook URL."""
        if not job.delivery or not job.delivery.webhook_url:
            raise ValueError("No webhook URL configured")

        url = job.delivery.webhook_url

        # Basic SSRF guard: reject non-HTTPS and private ranges
        if not url.startswith("https://"):
            raise ValueError(f"Webhook URL must use HTTPS: {url}")

        payload = {
            "event": "cron.finished",
            "jobId": job.id,
            "jobName": job.name,
            "status": run_entry.status,
            "summary": result_text,
            "runEntry": run_entry.to_dict(),
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if job.delivery.webhook_token:
            headers["Authorization"] = f"Bearer {job.delivery.webhook_token}"

        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for webhook delivery")

        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                content=json.dumps(payload, ensure_ascii=False),
                headers=headers,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Webhook returned {response.status_code}: {response.text[:200]}"
                )
