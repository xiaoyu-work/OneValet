"""
Shipment Domain Tools — Standalone API functions for ShippingAgent's mini ReAct loop.

Extracted from ShipmentAgent (tracking.py).
Each function takes (args: dict, context: DomainToolContext) -> str.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from onevalet.agents.domain_agent import DomainToolContext

from .shipment_repo import ShipmentRepository

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

def _detect_carrier(tracking_number: str) -> Optional[str]:
    """Simple carrier detection from tracking number format."""
    if not tracking_number:
        return None
    tn = tracking_number.upper()
    if tn.startswith("1Z"):
        return "ups"
    if tn.isdigit() and 12 <= len(tn) <= 22:
        if len(tn) in [12, 15, 20, 22]:
            return "fedex"
        if len(tn) in [20, 22]:
            return "usps"
    if len(tn) in [10, 11] and tn.isdigit():
        return "dhl"
    return None


def _get_repo(context: DomainToolContext) -> Optional[ShipmentRepository]:
    """Get shipment repository from context_hints."""
    if not context.context_hints:
        return None
    db = context.context_hints.get("db")
    if not db:
        return None
    return ShipmentRepository(db)


def _get_tracking_provider():
    """Get tracking provider instance."""
    try:
        from onevalet.providers.shipment import TrackingProvider
        return TrackingProvider()
    except ImportError:
        logger.warning("Shipment tracking provider not available")
        return None


def _format_shipment_status(result: Dict, description: str = None) -> str:
    """Format single shipment status for display."""
    carrier = result.get("carrier", "").upper()
    tracking = result.get("tracking_number", "")
    status = result.get("status", "unknown")
    last_update = result.get("last_update", "No update available")
    eta = result.get("estimated_delivery")
    url = result.get("tracking_url")

    desc = f" ({description})" if description else ""

    lines = [f"{carrier} {tracking}{desc}"]
    lines.append(f"Status: {status.replace('_', ' ').title()}")
    lines.append(f"Latest: {last_update}")

    if eta:
        lines.append(f"ETA: {eta}")
    if url:
        lines.append(f"Track: {url}")

    return "\n".join(lines)


def _format_all_shipments(shipments: List[Dict]) -> str:
    """Format multiple shipments for display."""
    if not shipments:
        return "No active shipments."

    lines = [f"Tracking {len(shipments)} package(s):"]

    for s in shipments:
        carrier = s.get("carrier", "").upper()
        tracking = s.get("tracking_number", "")
        status = s.get("status", "unknown").replace("_", " ").title()
        desc = f" - {s['description']}" if s.get("description") else ""
        last = s.get("last_update", "")

        if last:
            lines.append(f"- {carrier} {tracking}{desc}: {status}")
            lines.append(f"  {last}")
        else:
            lines.append(f"- {carrier} {tracking}{desc}: {status}")

    return "\n".join(lines)


def _find_matching_shipments(
    shipments: List[Dict],
    tracking_number: str = None,
    carrier: str = None,
    description_pattern: str = None,
) -> List[Dict]:
    """Find shipments matching the given criteria."""
    matches = shipments

    if tracking_number:
        matches = [s for s in matches if s["tracking_number"].upper() == tracking_number.upper()]
    if carrier:
        matches = [s for s in matches if s["carrier"].lower() == carrier.lower()]
    if description_pattern:
        pattern = description_pattern.lower()
        matches = [s for s in matches if s.get("description") and pattern in s["description"].lower()]

    return matches


# =============================================================================
# track_shipment
# =============================================================================

async def _query_one(
    tracking_number: str,
    carrier: str,
    description: str,
    provider,
    repo: Optional[ShipmentRepository],
    tenant_id: str,
) -> str:
    """Query a specific shipment by tracking number."""
    if not tracking_number:
        return "No tracking number provided."

    if not provider:
        return "Shipment tracking is not available right now."

    if not carrier:
        carrier = _detect_carrier(tracking_number)

    if not carrier:
        return f"Could not identify carrier for {tracking_number}. Please specify the carrier."

    result = await provider.track(tracking_number, carrier)

    if not result.get("success"):
        return f"Failed to track {tracking_number}: {result.get('error')}"

    status = result.get("status", "unknown")

    if repo:
        delivered_notified = True if status == "delivered" else None
        await repo.upsert_shipment(
            user_id=tenant_id,
            tracking_number=tracking_number,
            carrier=carrier,
            tracking_url=result.get("tracking_url"),
            status=status,
            description=description,
            last_update=result.get("last_update"),
            estimated_delivery=result.get("estimated_delivery"),
            tracking_history=result.get("events", []),
            delivered_notified=delivered_notified,
        )

        if status == "delivered":
            await repo.archive_shipment_by_tracking(tenant_id, tracking_number)
            logger.info(f"Archived {tracking_number} after delivery")

    return _format_shipment_status(result, description)


async def track_shipment(args: dict, context: DomainToolContext) -> str:
    """Track, query, and manage shipments.

    Supports actions: query_one, query_all, update, delete, history.
    """
    action = args.get("action", "query_one")
    tracking_number = args.get("tracking_number", "")
    carrier = args.get("carrier", "")
    description = args.get("description", "")
    description_pattern = args.get("description_pattern", "")

    repo = _get_repo(context)
    provider = _get_tracking_provider()
    tenant_id = context.tenant_id

    # ----- query_one / query_multiple -----
    if action == "query_one":
        # Support list of tracking numbers
        if isinstance(tracking_number, list):
            results = []
            errors = []
            for tn in tracking_number:
                try:
                    text = await _query_one(tn, carrier, description, provider, repo, tenant_id)
                    if "Failed" in text:
                        errors.append(f"{tn}: {text}")
                    else:
                        results.append(f"{tn}:\n{text}")
                except Exception as e:
                    errors.append(f"{tn}: {e}")

            parts = []
            if results:
                parts.append(f"Tracked {len(results)} package(s):\n")
                parts.extend(f"{i}. {r}\n" for i, r in enumerate(results, 1))
            if errors:
                parts.append(f"\nFailed to track {len(errors)} package(s):\n")
                parts.extend(f"{i}. {e}\n" for i, e in enumerate(errors, 1))
            return "".join(parts).strip() if parts else "No results."

        return await _query_one(tracking_number, carrier, description, provider, repo, tenant_id)

    # ----- query_all -----
    if action == "query_all":
        if not repo:
            return "Shipment storage is not available right now."

        shipments = await repo.get_user_shipments(tenant_id, is_active=True)
        if not shipments:
            return "You don't have any active shipments being tracked."

        async def fetch_and_update(shipment):
            tn = shipment["tracking_number"]
            sc = shipment["carrier"]
            current_status = shipment.get("status", "").lower()

            if current_status == "delivered":
                return {
                    "tracking_number": tn,
                    "carrier": sc,
                    "status": "delivered",
                    "last_update": shipment.get("last_update", ""),
                    "description": shipment.get("description"),
                    "tracking_url": shipment.get("tracking_url"),
                }

            if not provider:
                return {
                    "tracking_number": tn,
                    "carrier": sc,
                    "status": shipment.get("status", "unknown"),
                    "last_update": shipment.get("last_update", "Provider unavailable"),
                    "description": shipment.get("description"),
                    "tracking_url": shipment.get("tracking_url"),
                }

            result = await provider.track(tn, sc)
            if result.get("success"):
                await repo.upsert_shipment(
                    user_id=tenant_id,
                    tracking_number=tn,
                    carrier=sc,
                    tracking_url=result.get("tracking_url"),
                    status=result.get("status", "unknown"),
                    description=shipment.get("description"),
                    last_update=result.get("last_update"),
                    estimated_delivery=result.get("estimated_delivery"),
                    tracking_history=result.get("events", []),
                )
                return {**result, "description": shipment.get("description")}
            else:
                return {
                    "tracking_number": tn,
                    "carrier": sc,
                    "status": shipment.get("status", "unknown"),
                    "last_update": shipment.get("last_update", "Could not refresh"),
                    "description": shipment.get("description"),
                    "tracking_url": shipment.get("tracking_url"),
                }

        updated = await asyncio.gather(*[fetch_and_update(s) for s in shipments])
        return _format_all_shipments(updated)

    # ----- update -----
    if action == "update":
        if not repo:
            return "Shipment storage is not available right now."

        shipments = await repo.get_user_shipments(tenant_id, is_active=True)
        if not shipments:
            return "No active shipments to update."

        matches = _find_matching_shipments(shipments, tracking_number, carrier, description_pattern)
        if not matches:
            return "No matching shipment found."
        if len(matches) > 1:
            lines = ["Multiple shipments match. Please be more specific:"]
            for i, s in enumerate(matches, 1):
                c = s["carrier"].upper()
                tn = s["tracking_number"]
                d = f" ({s['description']})" if s.get("description") else ""
                lines.append(f"{i}. {c} {tn}{d}")
            return "\n".join(lines)

        shipment = matches[0]
        update_data = {}
        if description:
            update_data["description"] = description
        if update_data:
            await repo.update_shipment(shipment["id"], update_data)

        return f"Updated {shipment['tracking_number']}: {description or 'no changes'}"

    # ----- delete -----
    if action == "delete":
        if not repo:
            return "Shipment storage is not available right now."

        shipments = await repo.get_user_shipments(tenant_id, is_active=True)
        if not shipments:
            return "No active shipments to delete."

        matches = _find_matching_shipments(shipments, tracking_number, carrier, description_pattern)

        if not tracking_number and not carrier and not description_pattern:
            if len(shipments) == 1:
                matches = shipments
            else:
                lines = ["Multiple shipments found. Please specify which one:"]
                for i, s in enumerate(shipments, 1):
                    c = s["carrier"].upper()
                    tn = s["tracking_number"]
                    d = f" ({s['description']})" if s.get("description") else ""
                    lines.append(f"{i}. {c} {tn}{d}")
                return "\n".join(lines)

        if not matches:
            return "No matching shipment found."
        if len(matches) > 1:
            lines = ["Multiple shipments match. Please be more specific:"]
            for i, s in enumerate(matches, 1):
                c = s["carrier"].upper()
                tn = s["tracking_number"]
                d = f" ({s['description']})" if s.get("description") else ""
                lines.append(f"{i}. {c} {tn}{d}")
            return "\n".join(lines)

        shipment = matches[0]
        await repo.archive_shipment(shipment["id"])
        desc = f" ({shipment['description']})" if shipment.get("description") else ""
        return f"Stopped tracking {shipment['carrier'].upper()} {shipment['tracking_number']}{desc}"

    # ----- history -----
    if action == "history":
        if not repo:
            return "Shipment storage is not available right now."

        shipments = await repo.get_user_shipments(tenant_id, is_active=False)
        if not shipments:
            return "No delivery history found."

        lines = [f"Found {len(shipments)} delivered/archived shipment(s):"]
        for s in shipments[:10]:
            desc = f" ({s['description']})" if s.get("description") else ""
            status = s.get("status", "delivered")
            lines.append(f"- {s['carrier'].upper()} {s['tracking_number']}{desc}: {status}")
        return "\n".join(lines)

    return f"Unknown action: {action}"
