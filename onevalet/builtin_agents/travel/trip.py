"""
Trip Agent - Manage user travel information

Handles:
1. Adding new trips from user messages
2. Querying upcoming trips
3. Updating/deleting trips
4. Extracting trips from email/calendar (called by other handlers)
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message
from .trip_repo import TripRepository

logger = logging.getLogger(__name__)


@valet()
class TripAgent(StandardAgent):
    """Trip management agent with extraction capabilities"""

    action = InputField(
        prompt="What would you like to do with your trips?",
        description="Action: add, query, update, delete",
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    def _get_repo(self):
        """Get TripRepository from context_hints['db']"""
        db = self.context_hints.get("db")
        if not db:
            return None
        if not hasattr(self, '_trip_repo'):
            self._trip_repo = TripRepository(db)
        return self._trip_repo

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract trip action from user input"""
        if not self.llm_client:
            return {"action": "query"}

        try:
            prompt = f"""Determine what the user wants to do with their trips.

User message: "{user_input}"

Actions:
- add: User is telling about a new trip (flight, hotel, rental car, train)
- query: User wants to see their upcoming trips
- delete: User wants to remove/cancel a trip

Return JSON:
{{
    "action": "add" | "query" | "delete",
    "search_term": "destination or flight number if mentioned"
}}

Return ONLY JSON:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You determine trip-related actions. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            extracted = json.loads(result.content.strip())
            extracted["original_message"] = user_input
            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"action": "query", "original_message": user_input}

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute trip action"""
        fields = self.collected_fields
        action = fields.get("action", "query")

        try:
            if action == "add":
                return await self._add_trip(fields)
            elif action == "query":
                return await self._query_trips(fields)
            elif action == "delete":
                return await self._delete_trip(fields)
            else:
                return await self._query_trips(fields)

        except Exception as e:
            logger.error(f"Trip action failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Sorry, I had trouble with your trip request. Try again?"
            )

    async def _add_trip(self, fields: Dict[str, Any]) -> AgentResult:
        """Add a new trip from user message"""
        original_message = fields.get("original_message", "")

        trip = await self.extract_and_save_trip(
            user_id=self.tenant_id,
            text=original_message,
            source="manual"
        )

        if not trip:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="I couldn't find trip details in your message. Can you include the flight number, dates, or hotel info?"
            )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=self._format_trip_saved(trip),
            data={"trip": trip}
        )

    async def _query_trips(self, fields: Dict[str, Any]) -> AgentResult:
        """Query user's trips"""
        repo = self._get_repo()
        if not repo:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Trip storage is not available right now."
            )

        search_term = fields.get("search_term")

        trips = await repo.get_user_trips(
            user_id=self.tenant_id,
            status="upcoming",
            search_term=search_term,
            limit=10
        )

        if not trips:
            if search_term:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No trips found matching '{search_term}'."
                )
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="You don't have any upcoming trips. Tell me about your next trip and I'll track it for you!"
            )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=self._format_trip_list(trips),
            data={"trips": trips}
        )

    async def _delete_trip(self, fields: Dict[str, Any]) -> AgentResult:
        """Delete/cancel a trip"""
        repo = self._get_repo()
        if not repo:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Trip storage is not available right now."
            )

        search_term = fields.get("search_term", "")

        if not search_term:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Which trip would you like to cancel? Tell me the destination or flight number."
            )

        trips = await repo.get_user_trips(
            user_id=self.tenant_id,
            status="upcoming",
            search_term=search_term
        )

        if not trips:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"No upcoming trip found matching '{search_term}'."
            )

        if len(trips) > 1:
            lines = ["Found multiple trips. Which one?"]
            for i, trip in enumerate(trips, 1):
                lines.append(f"{i}. {trip.get('title', 'Trip')}")
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="\n".join(lines)
            )

        trip = trips[0]
        await repo.soft_delete_trip(trip["id"])

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Cancelled: {trip.get('title', 'Trip')}"
        )

    # =========================================================================
    # Public Methods (for email handler / calendar agent to call)
    # =========================================================================

    async def extract_and_save_trip(
        self,
        user_id: str,
        text: str,
        source: str = "manual",
        source_id: str = None,
        source_account: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract trip info from text and save to database.
        Handles deduplication automatically.
        """
        if not text or not self.llm_client:
            return None

        try:
            trip_data = await self._extract_trip_info(text)

            if not trip_data or not trip_data.get("has_trip"):
                return None

            trip_data["user_id"] = user_id
            trip_data["source"] = source
            trip_data["source_id"] = source_id
            trip_data["source_account"] = source_account
            trip_data["raw_data"] = {"original_text": text[:2000]}

            return await self._save_trip_with_dedup(user_id, trip_data)

        except Exception as e:
            logger.error(f"Trip extraction failed: {e}", exc_info=True)
            return None

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _extract_trip_info(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract trip information from text."""
        now = datetime.now()
        current_time_str = now.strftime("%A, %B %d, %Y %I:%M %p")

        prompt = f"""Analyze this text and extract travel/trip information if present.

CURRENT TIME CONTEXT:
- Current date/time: {current_time_str}

Use this to convert relative dates like "next Monday", "tomorrow", "in 2 weeks" to actual dates.

TEXT:
{text[:3000]}

Look for:
1. FLIGHTS: airline, flight number, departure/arrival cities, times, confirmation code
2. HOTELS: hotel name, address, check-in/out dates, confirmation number
3. CAR RENTALS: company, pickup/dropoff locations and times
4. TRAINS/BUS: carrier, route, times

IMPORTANT:
- Extract destination city even if only mentioned casually
- Convert relative dates to ISO format
- If only destination is mentioned without origin, still extract it
- Look up airport codes if you know them

Return JSON (only include fields found or inferred from text):
{{
    "has_trip": true,
    "trip_type": "flight" | "hotel" | "car_rental" | "train",
    "title": "SFO -> MIA",
    "carrier": "United Airlines",
    "carrier_code": "UA",
    "trip_number": "UA1234",
    "booking_reference": "ABC123",
    "origin_city": "San Francisco",
    "origin_code": "SFO",
    "destination_city": "Miami",
    "destination_code": "MIA",
    "departure_time": "2024-12-15T10:30:00",
    "departure_local_time": "10:30 AM",
    "arrival_time": "2024-12-15T13:45:00",
    "arrival_local_time": "1:45 PM",
    "hotel_name": "Marriott Downtown",
    "check_in_date": "2024-12-15",
    "check_out_date": "2024-12-17",
    "rental_company": "Hertz",
    "pickup_time": "2024-12-15T14:00:00",
    "dropoff_time": "2024-12-17T10:00:00"
}}

If no travel info found: {{"has_trip": false}}

Return ONLY JSON:"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract travel information. Convert relative dates using the provided current time context. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            return json.loads(result.content.strip())

        except Exception as e:
            logger.error(f"LLM trip extraction failed: {e}")
            return {"has_trip": False}

    async def _save_trip_with_dedup(
        self,
        user_id: str,
        trip_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Save trip with deduplication."""
        repo = self._get_repo()
        if not repo:
            logger.error("No database client available for saving trip")
            return None

        try:
            existing = await self._find_existing_trip(user_id, trip_data)

            if existing:
                logger.info(f"Trip exists: {existing.get('id')}, updating")
                return await self._update_existing_trip(existing["id"], trip_data)

            insert_data = self._prepare_insert_data(trip_data)
            row = await repo.insert_trip(insert_data)

            if row:
                logger.info(f"Created trip: {row.get('id')}")
            return row

        except Exception as e:
            logger.error(f"Save trip failed: {e}", exc_info=True)
            return None

    async def _find_existing_trip(
        self,
        user_id: str,
        trip_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find existing trip for deduplication."""
        repo = self._get_repo()
        if not repo:
            return None

        try:
            source_id = trip_data.get("source_id")
            if source_id:
                found = await repo.find_by_source_id(user_id, source_id)
                if found:
                    return found

            booking_ref = trip_data.get("booking_reference")
            if booking_ref:
                found = await repo.find_by_booking_reference(user_id, booking_ref)
                if found:
                    return found

            trip_number = trip_data.get("trip_number")
            departure_time = trip_data.get("departure_time")

            if trip_number and departure_time:
                try:
                    dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
                    found = await repo.find_by_trip_number_and_date(user_id, trip_number, dt)
                    if found:
                        return found
                except Exception:
                    pass

            origin_code = trip_data.get("origin_code")
            dest_code = trip_data.get("destination_code")

            if origin_code and dest_code and departure_time:
                try:
                    dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
                    found = await repo.find_by_route_and_date(user_id, origin_code, dest_code, dt)
                    if found:
                        return found
                except Exception:
                    pass

            hotel_name = trip_data.get("hotel_name")
            check_in_date = trip_data.get("check_in_date")

            if hotel_name and check_in_date:
                found = await repo.find_by_hotel(user_id, hotel_name, check_in_date)
                if found:
                    return found

            return None

        except Exception as e:
            logger.error(f"Find existing trip error: {e}")
            return None

    async def _update_existing_trip(
        self,
        trip_id: str,
        trip_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update existing trip with new info."""
        repo = self._get_repo()
        if not repo:
            return None

        try:
            update_fields = [
                "title", "carrier", "carrier_code", "trip_number", "booking_reference",
                "origin_city", "origin_code", "destination_city", "destination_code",
                "departure_time", "departure_local_time", "departure_terminal", "departure_gate",
                "arrival_time", "arrival_local_time", "arrival_terminal", "arrival_gate",
                "hotel_name", "hotel_address", "check_in_date", "check_out_date",
                "rental_company", "pickup_time", "dropoff_time"
            ]

            update_data = {k: v for k, v in trip_data.items() if k in update_fields and v}

            if update_data:
                return await repo.update_trip(trip_id, update_data)

            return None

        except Exception as e:
            logger.error(f"Update trip error: {e}")
            return None

    def _prepare_insert_data(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for insert."""
        exclude = {"has_trip"}
        data = {k: v for k, v in trip_data.items() if k not in exclude and v is not None}

        if "status" not in data:
            data["status"] = "upcoming"

        if "title" not in data:
            data["title"] = self._generate_title(trip_data)

        if "raw_data" in data and not isinstance(data["raw_data"], str):
            data["raw_data"] = json.dumps(data["raw_data"])

        return data

    def _generate_title(self, trip_data: Dict[str, Any]) -> str:
        """Generate user-friendly title."""
        trip_type = trip_data.get("trip_type", "")

        if trip_type == "flight":
            origin = trip_data.get("origin_code") or trip_data.get("origin_city", "")
            dest = trip_data.get("destination_code") or trip_data.get("destination_city", "")
            if origin and dest:
                return f"{origin} -> {dest}"

        elif trip_type == "hotel":
            return trip_data.get("hotel_name") or f"Hotel in {trip_data.get('destination_city', '')}"

        elif trip_type == "car_rental":
            return f"{trip_data.get('rental_company', '')} Rental"

        return "Trip"

    def _format_trip_saved(self, trip: Dict[str, Any]) -> str:
        """Format response for saved trip."""
        trip_type = trip.get("trip_type", "trip")
        title = trip.get("title", "Your trip")

        lines = [f"Got it! Saved your {trip_type}:", f"{title}"]

        if trip.get("departure_local_time"):
            lines.append(f"Departs: {trip['departure_local_time']}")
        if trip.get("booking_reference"):
            lines.append(f"Conf: {trip['booking_reference']}")
        if trip.get("check_in_date"):
            lines.append(f"Check-in: {trip['check_in_date']}")

        lines.append("I'll remind you before departure!")
        return "\n".join(lines)

    def _format_trip_list(self, trips: List[Dict[str, Any]]) -> str:
        """Format trip list with all available details."""
        lines = [f"{len(trips)} upcoming trip(s):"]

        for i, trip in enumerate(trips, 1):
            trip_type = trip.get("trip_type", "")

            if trip_type == "flight":
                parts = []
                if trip.get("trip_number"):
                    parts.append(trip["trip_number"])
                origin = trip.get("origin_code") or trip.get("origin_city")
                dest = trip.get("destination_code") or trip.get("destination_city")
                if origin and dest:
                    parts.append(f"{origin} -> {dest}")
                elif dest:
                    parts.append(f"to {dest}")
                elif origin:
                    parts.append(f"from {origin}")
                if not parts and trip.get("carrier"):
                    parts.append(trip["carrier"])
                title = " ".join(parts) if parts else trip.get("title", "Flight")
            elif trip_type == "hotel":
                title = trip.get("hotel_name") or trip.get("title", "Hotel")
            elif trip_type == "car_rental":
                title = trip.get("rental_company") or trip.get("title", "Car Rental")
            else:
                title = trip.get("title", "Trip")

            line = f"{i}. {title}"

            if trip.get("departure_time"):
                try:
                    dt = datetime.fromisoformat(trip["departure_time"].replace("Z", "+00:00"))
                    line += f" - {dt.strftime('%b %d')}"
                    if trip.get("departure_local_time"):
                        line += f" {trip['departure_local_time']}"
                except Exception:
                    pass
            elif trip.get("check_in_date"):
                line += f" - Check-in {trip['check_in_date']}"

            if trip.get("booking_reference"):
                line += f" (Conf: {trip['booking_reference']})"

            lines.append(line)

        return "\n".join(lines)


# =========================================================================
# Helper function for email/calendar handlers
# =========================================================================

async def extract_trip_from_email(
    user_id: str,
    email_data: Dict[str, Any],
    llm_client,
    source_account: str = None
) -> Optional[Dict[str, Any]]:
    """Extract trip from email (called by email handler)."""
    agent = TripAgent(tenant_id=user_id, llm_client=llm_client)

    text = f"From: {email_data.get('sender', '')}\nSubject: {email_data.get('subject', '')}\n\n{email_data.get('snippet', '')}"

    return await agent.extract_and_save_trip(
        user_id=user_id,
        text=text,
        source="email",
        source_id=email_data.get("message_id"),
        source_account=source_account
    )


async def extract_trip_from_calendar(
    user_id: str,
    event_data: Dict[str, Any],
    llm_client,
    source_account: str = None
) -> Optional[Dict[str, Any]]:
    """Extract trip from calendar event (called by calendar agent)."""
    agent = TripAgent(tenant_id=user_id, llm_client=llm_client)

    text = f"Event: {event_data.get('summary', '')}\nLocation: {event_data.get('location', '')}\nStart: {event_data.get('start', '')}\nEnd: {event_data.get('end', '')}\n\n{event_data.get('description', '')}"

    return await agent.extract_and_save_trip(
        user_id=user_id,
        text=text,
        source="calendar",
        source_id=event_data.get("id"),
        source_account=source_account
    )
