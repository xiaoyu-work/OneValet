"""ShipmentPoller — hourly background task that refreshes shipment statuses
and notifies users when a package status changes.

Runs every hour. For each user with active shipments:
1. Check if current time is within waking hours (9am-10pm) in user's timezone
2. Poll 17Track for fresh status on all non-delivered shipments
3. Compare old status with new status
4. If changed: send notification via CallbackNotification
5. Update DB with new data
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 3600  # 1 hour
WAKING_HOUR_START = 9   # 9 AM
WAKING_HOUR_END = 22    # 10 PM

# Human-readable status labels
STATUS_LABELS = {
    "delivered": "Delivered",
    "in_transit": "In Transit",
    "out_for_delivery": "Out for Delivery",
    "info_received": "Info Received",
    "exception": "Exception",
    "pending": "Pending",
    "expired": "Expired",
}


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status.replace("_", " ").title())


class ShipmentPoller:
    """Background service that polls shipment statuses and notifies on changes."""

    def __init__(self, db, notification=None):
        """
        Args:
            db: Database instance (onevalet.db.Database) with asyncpg pool.
            notification: CallbackNotification instance for sending alerts.
        """
        self._db = db
        self._notification = notification
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("ShipmentPoller started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ShipmentPoller stopped")

    async def _loop(self) -> None:
        # Brief initial delay so app finishes startup
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._poll_all_users()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ShipmentPoller error: {e}")
            await asyncio.sleep(POLL_INTERVAL_S)

    async def _poll_all_users(self) -> None:
        """Find all users with active non-delivered shipments and poll them."""
        rows = await self._db.fetch(
            "SELECT DISTINCT tenant_id FROM shipments "
            "WHERE is_active = TRUE AND LOWER(COALESCE(status, '')) != 'delivered'"
        )
        if not rows:
            return

        logger.info(f"ShipmentPoller: checking {len(rows)} user(s)")
        for row in rows:
            tenant_id = row["tenant_id"]
            try:
                tz = await self._get_user_timezone(tenant_id)
                if not self._is_waking_hours(tz):
                    continue
                await self._poll_user(tenant_id)
            except Exception as e:
                logger.warning(f"ShipmentPoller: error polling {tenant_id}: {e}")

    async def _get_user_timezone(self, tenant_id: str) -> str:
        """Read timezone from tenant_profiles, default to UTC."""
        try:
            row = await self._db.fetchrow(
                "SELECT profile->>'timezone' AS tz FROM tenant_profiles "
                "WHERE tenant_id = $1",
                tenant_id,
            )
            if row and row["tz"]:
                return row["tz"]
        except Exception:
            pass
        return "UTC"

    @staticmethod
    def _is_waking_hours(tz_name: str) -> bool:
        """Return True if current time in the given timezone is 9am-10pm."""
        try:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(tz_name)
            except ImportError:
                import pytz
                tz = pytz.timezone(tz_name)
            local_now = datetime.now(tz)
            return WAKING_HOUR_START <= local_now.hour < WAKING_HOUR_END
        except Exception:
            # Invalid timezone — default to allowing poll
            return True

    async def _poll_user(self, tenant_id: str) -> None:
        """Poll all active non-delivered shipments for one user."""
        from onevalet.providers.shipment import TrackingProvider
        from onevalet.builtin_agents.shipment.shipment_repo import ShipmentRepository

        provider = TrackingProvider()
        if not provider.api_key:
            return

        repo = ShipmentRepository(self._db)
        shipments = await repo.get_user_shipments(tenant_id, is_active=True)
        if not shipments:
            return

        changes: List[Dict[str, Any]] = []

        for shipment in shipments:
            old_status = (shipment.get("status") or "unknown").lower()
            if old_status == "delivered":
                continue

            tn = shipment["tracking_number"]
            carrier = shipment.get("carrier", "")

            try:
                result = await provider.track(tn, carrier)
            except Exception as e:
                logger.debug(f"Failed to poll {tn}: {e}")
                continue

            if not result.get("success"):
                continue

            new_status = (result.get("status") or "unknown").lower()

            # Update DB regardless
            await repo.upsert_shipment(
                tenant_id=tenant_id,
                tracking_number=tn,
                carrier=carrier or result.get("carrier", ""),
                tracking_url=result.get("tracking_url"),
                status=new_status,
                description=shipment.get("description"),
                last_update=result.get("last_update"),
                estimated_delivery=result.get("estimated_delivery"),
                tracking_history=result.get("events", []),
            )

            # Auto-archive delivered
            if new_status == "delivered":
                await repo.archive_shipment_by_tracking(tenant_id, tn)

            # Record change
            if new_status != old_status:
                changes.append({
                    "tracking_number": tn,
                    "carrier": carrier,
                    "description": shipment.get("description"),
                    "old_status": old_status,
                    "new_status": new_status,
                    "last_update": result.get("last_update"),
                    "estimated_delivery": result.get("estimated_delivery"),
                })

        if changes:
            await self._notify_changes(tenant_id, changes)

    async def _notify_changes(
        self, tenant_id: str, changes: List[Dict[str, Any]]
    ) -> None:
        """Send a notification about shipment status changes."""
        if not self._notification:
            logger.info(
                f"ShipmentPoller: {len(changes)} change(s) for {tenant_id} "
                f"but no notification channel configured"
            )
            return

        lines = []
        for c in changes:
            desc = c.get("description") or c["tracking_number"]
            old_label = _status_label(c["old_status"])
            new_label = _status_label(c["new_status"])
            line = f"{desc}: {old_label} -> {new_label}"
            if c.get("last_update"):
                line += f" ({c['last_update']})"
            lines.append(line)

        message = "Package update:\n" + "\n".join(lines)

        try:
            await self._notification.send(
                tenant_id,
                message,
                {
                    "category": "shipment_update",
                    "priority": "normal",
                    "trigger_type": "shipment_poll",
                },
            )
            logger.info(
                f"ShipmentPoller: notified {tenant_id} of {len(changes)} change(s)"
            )
        except Exception as e:
            logger.warning(f"ShipmentPoller: notification failed for {tenant_id}: {e}")
