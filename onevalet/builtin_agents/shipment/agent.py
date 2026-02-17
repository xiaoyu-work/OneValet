"""
ShippingAgent - Agent for all shipment tracking and management.

Replaces the old ShipmentAgent (StandardAgent) with a single agent
that has its own mini ReAct loop. The orchestrator sees only one
"ShippingAgent" tool instead of a raw StandardAgent with InputFields.

The internal LLM decides which action to perform (query_one, query_all,
update, delete, history) based on the user's request.
"""

from datetime import datetime

from onevalet import valet
from onevalet.standard_agent import StandardAgent

from .tools import track_shipment


@valet(capabilities=["shipping"])
class ShippingAgent(StandardAgent):
    """Track packages and check delivery status. Use when the user mentions a tracking number, package, shipment, delivery, or asks where their order is."""

    max_turns = 5

    _SYSTEM_PROMPT_TEMPLATE = """\
You are a shipment tracking assistant with access to package tracking tools.

Available tools:
- track_shipment: Track, query, update, or delete shipments. Supports multiple actions via the "action" parameter.

Today's date: {today} ({weekday})

Instructions:
1. If the user provides a tracking number, use action "query_one" to look it up.
2. If the user asks about all their packages, use action "query_all".
3. If the user wants to update a shipment description, use action "update".
4. If the user wants to stop tracking a package, use action "delete".
5. If the user asks about past deliveries, use action "history".
6. If the request is ambiguous or missing a tracking number when needed, ASK the user for clarification.
7. After getting results, present a clear summary to the user.

Common carrier tracking number formats:
- UPS: Starts with 1Z (e.g., 1Z999AA10123456784)
- FedEx: 12-22 digits (e.g., 123456789012)
- USPS: 20-22 digits or XX123456789US format
- DHL: 10-11 digits"""

    def get_system_prompt(self) -> str:
        now = datetime.now()
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            today=now.strftime('%Y-%m-%d'),
            weekday=now.strftime('%A'),
        )

    tools = [
        track_shipment,
    ]
