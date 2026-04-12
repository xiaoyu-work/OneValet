"""
Shipment Tools — Standalone API functions for ShippingAgent's mini ReAct loop.

Extracted from ShipmentAgent (tracking.py).
Each function takes (args: dict, context: AgentToolContext) -> str.
"""

import asyncio
import json
import logging
from typing import Annotated, Dict, List, Optional

from koa.models import AgentToolContext, ToolOutput
from koa.tool_decorator import tool

from .shipment_repo import ShipmentRepository

try:
    from koa.providers.shipment.carrier_detector import get_tracking_url
except ImportError:

    def get_tracking_url(carrier: str, tracking_number: str):  # type: ignore[misc]
        return None


logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================


def _detect_carrier(tracking_number: str) -> Optional[str]:
    """Carrier detection is handled by 17TRACK identify API in TrackingProvider.
    Return None here so tracking.py can call the identify endpoint."""
    return None


def _get_repo(context: AgentToolContext) -> Optional[ShipmentRepository]:
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
        from koa.providers.shipment import TrackingProvider

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
        matches = [
            s for s in matches if s.get("description") and pattern in s["description"].lower()
        ]

    return matches


def _build_shipment_card(data: Dict) -> Dict:
    """Build a shipment card dict from raw shipment data."""
    card = {"card_type": "shipment"}
    if data.get("carrier"):
        card["carrier"] = (
            data["carrier"].upper() if isinstance(data["carrier"], str) else str(data["carrier"])
        )
    if data.get("tracking_number"):
        card["trackingNumber"] = data["tracking_number"]
    if data.get("status"):
        card["status"] = (
            data["status"].replace("_", " ").title()
            if isinstance(data["status"], str)
            else str(data["status"])
        )
    if data.get("last_update"):
        card["lastUpdate"] = data["last_update"]
    if data.get("estimated_delivery"):
        card["eta"] = data["estimated_delivery"]
    if data.get("tracking_url"):
        card["trackingUrl"] = data["tracking_url"]
    if data.get("description"):
        card["description"] = data["description"]
    return card


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
) -> tuple:
    """Query a specific shipment by tracking number.

    Returns (formatted_text, raw_result_dict_or_None).
    Always saves to repo if possible, even when tracking info is unavailable.
    """
    if not tracking_number:
        return "No tracking number provided.", None

    if not carrier:
        carrier = _detect_carrier(tracking_number)

    # If no provider, save to repo with pending status
    if not provider:
        if repo:
            await repo.upsert_shipment(
                tenant_id=tenant_id,
                tracking_number=tracking_number,
                carrier=carrier or "unknown",
                tracking_url=None,
                status="pending",
                description=description,
                last_update="Added to tracking. Status will update when available.",
                estimated_delivery=None,
                tracking_history=[],
            )
            raw_data = {
                "tracking_number": tracking_number,
                "carrier": carrier or "unknown",
                "status": "pending",
                "last_update": "Added to tracking. Status will update when available.",
                "description": description,
            }
            msg = f"Added {tracking_number} to your tracking list."
            msg += " Live tracking is temporarily unavailable; status will update later."
            return msg, raw_data
        return "Shipment tracking is not available right now.", None

    result = await provider.track(tracking_number, carrier)

    if not result.get("success"):
        # Provider failed, but still save to repo so user can track later
        if repo:
            await repo.upsert_shipment(
                tenant_id=tenant_id,
                tracking_number=tracking_number,
                carrier=carrier,
                tracking_url=get_tracking_url(carrier, tracking_number) if carrier else None,
                status="pending",
                description=description,
                last_update=result.get("error", "Tracking info not yet available."),
                estimated_delivery=None,
                tracking_history=[],
            )
            raw_data = {
                "tracking_number": tracking_number,
                "carrier": carrier,
                "status": "pending",
                "last_update": result.get("error", "Tracking info not yet available."),
                "description": description,
                "tracking_url": get_tracking_url(carrier, tracking_number) or "" if carrier else "",
            }
            return (
                f"Added {tracking_number} to your tracking list. "
                f"Live status is not available yet ({result.get('error', 'unknown reason')}). "
                f"It will update automatically once the carrier scans the package."
            ), raw_data
        return f"Failed to track {tracking_number}: {result.get('error')}", None

    status = result.get("status", "unknown")

    if repo:
        delivered_notified = True if status == "delivered" else None
        await repo.upsert_shipment(
            tenant_id=tenant_id,
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

    raw_data = {**result, "description": description}
    return _format_shipment_status(result, description), raw_data


@tool
async def track_shipment(
    action: Annotated[str, "The operation to perform"],
    tracking_number: Annotated[str, "Package tracking number (required for query_one)"] = "",
    carrier: Annotated[str, "Carrier name (auto-detected if not provided)"] = "",
    description: Annotated[str, "Label or description for the package"] = "",
    description_pattern: Annotated[str, "Keywords to match an existing shipment description"] = "",
    *,
    context: AgentToolContext,
) -> str:
    """Track, query, and manage shipments. Supports actions: query_one (track a specific package), query_all (list all active shipments), update (change description), delete (stop tracking), history (view past deliveries)."""

    repo = _get_repo(context)
    provider = _get_tracking_provider()
    tenant_id = context.tenant_id

    # ----- query_one / query_multiple -----
    if action == "query_one":
        # Support list of tracking numbers
        if isinstance(tracking_number, list):
            results = []
            errors = []
            shipment_cards = []
            for tn in tracking_number:
                try:
                    text, raw = await _query_one(
                        tn, carrier, description, provider, repo, tenant_id
                    )
                    if "Failed" in text:
                        errors.append(f"{tn}: {text}")
                    else:
                        results.append(f"{tn}:\n{text}")
                        if raw:
                            shipment_cards.append(_build_shipment_card(raw))
                except Exception as e:
                    errors.append(f"{tn}: {e}")

            parts = []
            if results:
                parts.append(f"Tracked {len(results)} package(s):\n")
                parts.extend(f"{i}. {r}\n" for i, r in enumerate(results, 1))
            if errors:
                parts.append(f"\nFailed to track {len(errors)} package(s):\n")
                parts.extend(f"{i}. {e}\n" for i, e in enumerate(errors, 1))
            text_result = "".join(parts).strip() if parts else "No results."

            if shipment_cards:
                media = [
                    {
                        "type": "inline_cards",
                        "data": json.dumps(shipment_cards),
                        "media_type": "application/json",
                        "metadata": {"for_storage": False},
                    }
                ]
                return ToolOutput(text=text_result, media=media)
            return text_result

        text, raw = await _query_one(
            tracking_number, carrier, description, provider, repo, tenant_id
        )
        if raw:
            card = _build_shipment_card(raw)
            media = [
                {
                    "type": "inline_cards",
                    "data": json.dumps([card]),
                    "media_type": "application/json",
                    "metadata": {"for_storage": False},
                }
            ]
            return ToolOutput(text=text, media=media)
        return text

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
                    tenant_id=tenant_id,
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
        text_result = _format_all_shipments(updated)

        # Build inline cards for frontend rendering
        shipment_cards = [_build_shipment_card(s) for s in updated if s]
        if shipment_cards:
            media = [
                {
                    "type": "inline_cards",
                    "data": json.dumps(shipment_cards),
                    "media_type": "application/json",
                    "metadata": {"for_storage": False},
                }
            ]
            return ToolOutput(text=text_result, media=media)
        return text_result

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
