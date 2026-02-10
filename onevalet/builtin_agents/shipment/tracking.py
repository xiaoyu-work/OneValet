"""
Shipment Agent - Track, query, and manage package deliveries

This agent handles all shipment tracking operations:
- Track new packages (auto-add if not exists)
- Query shipment status (refreshes from carrier API)
- Update shipment info (description, etc.)
- Delete/stop tracking shipments
- Query archived/delivered shipments

State Flow:
1. INITIALIZING -> extract fields
2. WAITING_FOR_INPUT -> if selection needed (multiple matches)
3. RUNNING -> execute operation
4. COMPLETED
"""
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message
from .shipment_repo import ShipmentRepository

logger = logging.getLogger(__name__)


@valet()
class ShipmentAgent(StandardAgent):
    """Track packages and shipments. Use when the user asks about delivery status or wants to track an order."""

    action = InputField(
        prompt="What would you like to do with your shipments?",
        description="Action to perform: query_one, query_all, update, delete, history",
    )
    tracking_number = InputField(
        prompt="What's the tracking number?",
        description="Package tracking number",
        required=False,
    )
    description = InputField(
        prompt="What would you like to call this package?",
        description="Description or label for the package",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

        self.tracking_provider = None
        self.pending_matches = []
        self._init_tracking_provider()

    def _init_tracking_provider(self):
        """Initialize tracking provider"""
        try:
            from onevalet.providers.shipment import TrackingProvider
            self.tracking_provider = TrackingProvider()
        except ImportError:
            logger.warning("Shipment tracking provider not available")

    def _get_repo(self):
        """Get shipment repository from context_hints"""
        db = self.context_hints.get("db")
        if not db:
            return None
        if not hasattr(self, '_shipment_repo'):
            self._shipment_repo = ShipmentRepository(db)
        return self._shipment_repo

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract shipment operation from user input"""
        import re

        def detect_carrier(tracking_number: str) -> Optional[str]:
            """Simple carrier detection from tracking number format"""
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

        if self.pending_matches and user_input.strip().isdigit():
            selected = int(user_input.strip())
            if 1 <= selected <= len(self.pending_matches):
                shipment = self.pending_matches[selected - 1]
                self.pending_matches = []
                return {
                    "action": self.collected_fields.get("action", "delete"),
                    "tracking_number": shipment["tracking_number"],
                    "selected_shipment": shipment
                }

        def find_tracking_number(text: str) -> Optional[str]:
            words = text.split()
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                if detect_carrier(clean_word):
                    return clean_word
                if clean_word.isdigit() and len(clean_word) >= 10:
                    return clean_word
            return None

        if not self.llm_client:
            tracking = find_tracking_number(user_input)
            if tracking:
                return {"action": "query_one", "tracking_number": tracking}
            return {"action": "query_all"}

        try:
            extraction_prompt = f"""Extract shipment operation from user input.

**User Input:** "{user_input}"

**Available Actions:**
- query_one: Query specific shipment by tracking number (also adds if not exists)
- query_all: Query all active shipments
- update: Update shipment description/info
- delete: Stop tracking a shipment
- history: View delivered/archived shipments

**Common Carrier Tracking Number Formats:**
- UPS: Starts with 1Z (e.g., 1Z999AA10123456784)
- FedEx: 12-22 digits (e.g., 123456789012)
- USPS: 20-22 digits or XX123456789US format
- DHL: 10-11 digits

**Return JSON:**
{{
    "action": "query_one|query_all|update|delete|history",
    "tracking_number": "<if mentioned>",
    "carrier": "<if explicitly mentioned: ups, fedex, usps, dhl>",
    "description": "<if user provides a label like 'my iPhone'>",
    "description_pattern": "<keywords to match existing description>"
}}

**Examples:**
- "Where is 1Z999AA1" -> {{"action": "query_one", "tracking_number": "1Z999AA1"}}
- "Track my packages" -> {{"action": "query_all"}}
- "Stop tracking the FedEx one" -> {{"action": "delete", "carrier": "fedex"}}
- "Show delivered packages" -> {{"action": "history"}}
"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            if result.content:
                return json.loads(result.content)
            else:
                raise ValueError("LLM returned empty content")

        except Exception as e:
            logger.error(f"Failed to extract shipment params: {e}")
            tracking = find_tracking_number(user_input)
            if tracking:
                logger.info(f"Fallback: found tracking number {tracking} in input")
                return {"action": "query_one", "tracking_number": tracking}
            return {"action": "query_all"}

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """Handle user selection from multiple matches"""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        if self.collected_fields.get("selected_shipment"):
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        if self.pending_matches:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._format_shipment_list_for_selection(self.pending_matches)
            )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message="No shipments to select from."
        )

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute shipment operation"""
        fields = self.collected_fields
        action = fields.get("action", "query_all")

        logger.info(f"Executing shipment action: {action}")

        try:
            if action == "query_one":
                tracking_numbers = fields.get("tracking_number")
                if isinstance(tracking_numbers, list):
                    return await self._query_multiple(tracking_numbers, fields.get("carrier"), fields.get("description"))
                else:
                    return await self._query_one(tracking_numbers, fields.get("carrier"), fields.get("description"))
            elif action == "query_all":
                return await self._query_all()
            elif action == "update":
                return await self._update_shipment(fields)
            elif action == "delete":
                return await self._delete_shipment(fields)
            elif action == "history":
                return await self._query_history()
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Unknown action: {action}"
                )

        except Exception as e:
            logger.error(f"Shipment operation failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Failed to {action}: {str(e)}"
            )

    async def _query_one(self, tracking_number: str, carrier: str = None, description: str = None) -> AgentResult:
        """Query a specific shipment"""
        repo = self._get_repo()

        if not tracking_number:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No tracking number provided"
            )

        if not self.tracking_provider:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Shipment tracking is not available right now."
            )

        if not carrier:
            # Simple carrier detection
            tn = tracking_number.upper()
            if tn.startswith("1Z"):
                carrier = "ups"
            elif tn.isdigit() and 12 <= len(tn) <= 22:
                carrier = "fedex"

        if not carrier:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Could not identify carrier for {tracking_number}. Please specify the carrier."
            )

        result = await self.tracking_provider.track(tracking_number, carrier)

        if not result.get("success"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Failed to track {tracking_number}: {result.get('error')}"
            )

        status = result.get("status", "unknown")

        if repo:
            delivered_notified = True if status == "delivered" else None
            await repo.upsert_shipment(
                user_id=self.tenant_id,
                tracking_number=tracking_number,
                carrier=carrier,
                tracking_url=result.get("tracking_url"),
                status=status,
                description=description,
                last_update=result.get("last_update"),
                estimated_delivery=result.get("estimated_delivery"),
                tracking_history=result.get("events", []),
                delivered_notified=delivered_notified
            )

            if status == "delivered":
                await repo.archive_shipment_by_tracking(self.tenant_id, tracking_number)
                logger.info(f"Archived {tracking_number} after delivery")

        response = self._format_shipment_status(result, description)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=response
        )

    async def _query_multiple(self, tracking_numbers: List[str], carrier: str = None, description: str = None) -> AgentResult:
        """Query multiple tracking numbers"""
        if not tracking_numbers:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No tracking numbers provided"
            )

        results = []
        errors = []

        for tracking_number in tracking_numbers:
            try:
                result = await self._query_one(tracking_number, carrier, description)
                if result.status == AgentStatus.COMPLETED and "Failed" not in result.raw_message:
                    results.append({"tracking_number": tracking_number, "message": result.raw_message})
                else:
                    errors.append({"tracking_number": tracking_number, "error": result.raw_message})
            except Exception as e:
                errors.append({"tracking_number": tracking_number, "error": str(e)})

        response_parts = []
        if results:
            response_parts.append(f"Tracked {len(results)} package(s):\n")
            for i, r in enumerate(results, 1):
                response_parts.append(f"{i}. {r['tracking_number']}: {r['message']}\n")

        if errors:
            response_parts.append(f"\nFailed to track {len(errors)} package(s):\n")
            for i, e in enumerate(errors, 1):
                response_parts.append(f"{i}. {e['tracking_number']}: {e['error']}\n")

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message="".join(response_parts).strip()
        )

    async def _query_all(self) -> AgentResult:
        """Query all active shipments"""
        repo = self._get_repo()
        if not repo:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Shipment storage is not available right now."
            )

        shipments = await repo.get_user_shipments(self.tenant_id, is_active=True)

        if not shipments:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="You don't have any active shipments being tracked."
            )

        async def fetch_and_update(shipment):
            tracking_number = shipment["tracking_number"]
            carrier = shipment["carrier"]
            current_status = shipment.get("status", "").lower()

            if current_status == "delivered":
                return {
                    "tracking_number": tracking_number,
                    "carrier": carrier,
                    "status": "delivered",
                    "last_update": shipment.get("last_update", ""),
                    "description": shipment.get("description"),
                    "tracking_url": shipment.get("tracking_url")
                }

            if not self.tracking_provider:
                return {
                    "tracking_number": tracking_number,
                    "carrier": carrier,
                    "status": shipment.get("status", "unknown"),
                    "last_update": shipment.get("last_update", "Provider unavailable"),
                    "description": shipment.get("description"),
                    "tracking_url": shipment.get("tracking_url")
                }

            result = await self.tracking_provider.track(tracking_number, carrier)

            if result.get("success"):
                await repo.upsert_shipment(
                    user_id=self.tenant_id,
                    tracking_number=tracking_number,
                    carrier=carrier,
                    tracking_url=result.get("tracking_url"),
                    status=result.get("status", "unknown"),
                    description=shipment.get("description"),
                    last_update=result.get("last_update"),
                    estimated_delivery=result.get("estimated_delivery"),
                    tracking_history=result.get("events", [])
                )
                return {**result, "description": shipment.get("description")}
            else:
                return {
                    "tracking_number": tracking_number,
                    "carrier": carrier,
                    "status": shipment.get("status", "unknown"),
                    "last_update": shipment.get("last_update", "Could not refresh"),
                    "description": shipment.get("description"),
                    "tracking_url": shipment.get("tracking_url")
                }

        updated_shipments = await asyncio.gather(*[fetch_and_update(s) for s in shipments])
        response = self._format_all_shipments(updated_shipments)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=response
        )

    async def _update_shipment(self, fields: Dict[str, Any]) -> AgentResult:
        """Update shipment info"""
        repo = self._get_repo()
        if not repo:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Shipment storage is not available right now."
            )

        tracking_number = fields.get("tracking_number")
        carrier = fields.get("carrier")
        description = fields.get("description")
        description_pattern = fields.get("description_pattern")

        shipments = await repo.get_user_shipments(self.tenant_id, is_active=True)

        if not shipments:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No active shipments to update"
            )

        matches = self._find_matching_shipments(shipments, tracking_number, carrier, description_pattern)

        if not matches:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No matching shipment found"
            )

        if len(matches) > 1:
            self.pending_matches = matches
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._format_shipment_list_for_selection(matches)
            )

        shipment = matches[0]
        update_data = {}
        if description:
            update_data["description"] = description

        if update_data:
            await repo.update_shipment(shipment["id"], update_data)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Updated {shipment['tracking_number']}: {description or 'no changes'}"
        )

    async def _delete_shipment(self, fields: Dict[str, Any]) -> AgentResult:
        """Delete/stop tracking a shipment"""
        repo = self._get_repo()
        if not repo:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Shipment storage is not available right now."
            )

        selected = fields.get("selected_shipment")
        if selected:
            await repo.archive_shipment(selected["id"])
            desc = f" ({selected['description']})" if selected.get("description") else ""
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Stopped tracking {selected['carrier'].upper()} {selected['tracking_number']}{desc}"
            )

        tracking_number = fields.get("tracking_number")
        carrier = fields.get("carrier")
        description_pattern = fields.get("description_pattern")

        shipments = await repo.get_user_shipments(self.tenant_id, is_active=True)

        if not shipments:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No active shipments to delete"
            )

        matches = self._find_matching_shipments(shipments, tracking_number, carrier, description_pattern)

        if not tracking_number and not carrier and not description_pattern:
            if len(shipments) == 1:
                matches = shipments
            else:
                self.pending_matches = shipments
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=self._format_shipment_list_for_selection(shipments)
                )

        if not matches:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No matching shipment found"
            )

        if len(matches) > 1:
            self.pending_matches = matches
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._format_shipment_list_for_selection(matches)
            )

        shipment = matches[0]
        await repo.archive_shipment(shipment["id"])

        desc = f" ({shipment['description']})" if shipment.get("description") else ""
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Stopped tracking {shipment['carrier'].upper()} {shipment['tracking_number']}{desc}"
        )

    async def _query_history(self) -> AgentResult:
        """Query archived/delivered shipments"""
        repo = self._get_repo()
        if not repo:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Shipment storage is not available right now."
            )

        shipments = await repo.get_user_shipments(self.tenant_id, is_active=False)

        if not shipments:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No delivery history found."
            )

        lines = [f"Found {len(shipments)} delivered/archived shipment(s):"]
        for s in shipments[:10]:
            desc = f" ({s['description']})" if s.get("description") else ""
            status = s.get("status", "delivered")
            lines.append(f"- {s['carrier'].upper()} {s['tracking_number']}{desc}: {status}")

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message="\n".join(lines)
        )

    def _find_matching_shipments(self, shipments: List[Dict], tracking_number: str = None, carrier: str = None, description_pattern: str = None) -> List[Dict]:
        """Find shipments matching the given criteria"""
        matches = shipments

        if tracking_number:
            matches = [s for s in matches if s["tracking_number"].upper() == tracking_number.upper()]

        if carrier:
            matches = [s for s in matches if s["carrier"].lower() == carrier.lower()]

        if description_pattern:
            pattern = description_pattern.lower()
            matches = [s for s in matches if s.get("description") and pattern in s["description"].lower()]

        return matches

    def _format_shipment_list_for_selection(self, shipments: List[Dict]) -> str:
        """Format shipment list for user selection"""
        lines = ["Which package do you mean?"]
        for i, s in enumerate(shipments, 1):
            carrier = s["carrier"].upper()
            tracking = s["tracking_number"]
            desc = f" ({s['description']})" if s.get("description") else ""
            lines.append(f"{i}. {carrier} {tracking}{desc}")
        return "\n".join(lines)

    def _format_shipment_status(self, result: Dict, description: str = None) -> str:
        """Format single shipment status for display"""
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

    def _format_all_shipments(self, shipments: List[Dict]) -> str:
        """Format multiple shipments for display"""
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
