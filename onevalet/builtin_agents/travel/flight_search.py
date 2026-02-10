"""
FlightSearchAgent - Search for flights using Amadeus Flight Offers Search API
"""
import os
import logging
import json
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class FlightSearchAgent(StandardAgent):
    """Flight search agent using Amadeus Flight Offers Search API"""

    origin = InputField(
        prompt="Where are you flying from?",
        description="Departure city or airport code",
    )
    destination = InputField(
        prompt="Where are you flying to?",
        description="Arrival city or airport code",
    )
    departure_date = InputField(
        prompt="When do you want to leave?",
        description="Departure date",
    )
    return_date = InputField(
        prompt="When are you coming back? (say 'one way' if not returning)",
        description="Return date for round trip",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self.access_token = None
        self._default_origin_loaded = False

    def needs_approval(self) -> bool:
        return False

    def _get_default_origin(self) -> Optional[str]:
        """Get default origin city from user's profile"""
        try:
            user_profile = self.context_hints.get("user_profile", {})
            if not user_profile or not user_profile.get("addresses"):
                return None

            addresses = user_profile.get("addresses")

            if isinstance(addresses, str):
                try:
                    addresses = json.loads(addresses)
                except Exception:
                    return None

            if not addresses:
                return None

            home_address = None
            if isinstance(addresses, list) and len(addresses) > 0:
                for addr in addresses:
                    if isinstance(addr, dict) and addr.get("label") == "home":
                        home_address = addr
                        break
                if not home_address:
                    home_address = addresses[0]
            elif isinstance(addresses, dict):
                home_address = (
                    addresses.get("home") or
                    addresses.get("primary") or
                    addresses.get("current") or
                    addresses.get("address")
                )

            if not home_address:
                return None

            city = None
            if isinstance(home_address, dict):
                city = home_address.get("city") or home_address.get("locality")
            elif isinstance(home_address, str):
                city = home_address

            return city if city else None

        except Exception as e:
            logger.error(f"Failed to get default origin from user profile: {e}")
            return None

    async def _get_access_token(self) -> Optional[str]:
        """Get Amadeus API access token"""
        if self.access_token:
            return self.access_token

        try:
            url = "https://test.api.amadeus.com/v1/security/oauth2/token"
            data = {
                "grant_type": "client_credentials",
                "client_id": os.getenv("AMADEUS_API_KEY", ""),
                "client_secret": os.getenv("AMADEUS_API_SECRET", "")
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, timeout=10.0)
                response.raise_for_status()
                result = response.json()

            self.access_token = result.get("access_token")
            logger.info("Got Amadeus access token")
            return self.access_token

        except Exception as e:
            logger.error(f"Failed to get Amadeus token: {e}")
            return None

    async def _convert_to_iata_code(self, location: str) -> str:
        """Convert city name to IATA airport code"""
        if not location:
            return ""

        common_mappings = {
            "seattle": "SEA", "new york": "JFK", "nyc": "JFK",
            "los angeles": "LAX", "la": "LAX", "san francisco": "SFO",
            "chicago": "ORD", "boston": "BOS", "miami": "MIA",
            "atlanta": "ATL", "dallas": "DFW", "denver": "DEN",
            "las vegas": "LAS", "portland": "PDX", "london": "LHR",
            "paris": "CDG", "tokyo": "NRT", "beijing": "PEK",
            "shanghai": "PVG", "hong kong": "HKG", "singapore": "SIN",
            "dubai": "DXB", "sydney": "SYD", "toronto": "YYZ"
        }

        location_lower = location.lower().strip()

        if len(location) == 3 and location.isalpha():
            return location.upper()

        if location_lower in common_mappings:
            return common_mappings[location_lower]

        if self.llm_client:
            try:
                prompt = f"""Convert this location to IATA airport code.
Location: "{location}"
Return ONLY the 3-letter IATA code, nothing else.
If you don't know, return "UNKNOWN".
IATA code:"""

                result = await self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You convert locations to IATA airport codes."},
                        {"role": "user", "content": prompt}
                    ],
                    enable_thinking=False
                )

                code = result.content.strip().upper()
                if code != "UNKNOWN" and len(code) == 3:
                    return code

            except Exception as e:
                logger.error(f"LLM IATA conversion failed: {e}")

        return location.upper()[:3]

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract flight search parameters from user input"""
        if not self.llm_client:
            return {"origin": "", "destination": "", "departure_date": ""}

        try:
            now = datetime.now()
            today_str = now.strftime('%Y-%m-%d')
            day_of_week = now.strftime('%A')

            prompt = f"""Extract flight search details from this message.

User message: "{user_input}"

Current date: {today_str} ({day_of_week})

Extract and return JSON:
{{
  "origin": "city or airport",
  "destination": "city or airport",
  "departure_date": "YYYY-MM-DD",
  "return_date": "YYYY-MM-DD or null"
}}

Return ONLY valid JSON:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract flight search parameters. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            data = json.loads(content)

            extracted = {
                "origin": data.get("origin", ""),
                "destination": data.get("destination", ""),
                "departure_date": data.get("departure_date", ""),
                "return_date": data.get("return_date", "") if data.get("return_date") else ""
            }

            if not extracted.get("origin") and not self._default_origin_loaded:
                self._default_origin_loaded = True
                default_origin = self._get_default_origin()
                if default_origin:
                    extracted["origin"] = default_origin
                    logger.info(f"Using default origin from user profile: {default_origin}")

            return extracted

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search for flights using Amadeus API"""
        origin = self.collected_fields.get("origin", "")
        destination = self.collected_fields.get("destination", "")
        departure_date = self.collected_fields.get("departure_date", "")
        return_date = self.collected_fields.get("return_date", "")

        logger.info(f"Searching flights: {origin} -> {destination} on {departure_date}")

        if not os.getenv("AMADEUS_API_KEY") or not os.getenv("AMADEUS_API_SECRET"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Amadeus API credentials not configured. Please contact support."
            )

        token = await self._get_access_token()
        if not token:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't connect to flight search service. Try again later?"
            )

        try:
            origin_code = await self._convert_to_iata_code(origin)
            dest_code = await self._convert_to_iata_code(destination)

            logger.info(f"Converted: {origin} -> {origin_code}, {destination} -> {dest_code}")

            url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
            params = {
                "originLocationCode": origin_code,
                "destinationLocationCode": dest_code,
                "departureDate": departure_date,
                "adults": "1",
                "max": "5",
                "currencyCode": "USD"
            }

            if return_date:
                params["returnDate"] = return_date

            headers = {"Authorization": f"Bearer {token}"}

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=20.0)
                response.raise_for_status()
                data = response.json()

            offers = data.get("data", [])

            if not offers:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No flights found from {origin} to {destination} on {departure_date}. Try different dates?"
                )

            logger.info(f"Found {len(offers)} flight offers")
            formatted = await self._format_results(offers, origin_code, dest_code, departure_date, return_date)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=formatted
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Amadeus API error: {e.response.status_code}")
            if e.response.status_code == 400:
                error_msg = "Invalid search: please check your dates and locations."
            elif e.response.status_code == 401:
                error_msg = "Flight search authentication failed. Please contact support."
            else:
                error_msg = "Couldn't search flights. Try again later?"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=error_msg
            )

        except Exception as e:
            logger.error(f"Flight search failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't search flights. Try again later?"
            )

    async def _format_results(self, offers: List[Dict], origin_code: str, dest_code: str, departure: str, return_date: str) -> str:
        """Format flight offers into concise SMS message"""
        flights_summary = []
        for i, offer in enumerate(offers[:3], 1):
            price = offer.get("price", {}).get("total", "N/A")
            itineraries = offer.get("itineraries", [])

            if not itineraries:
                continue

            outbound = itineraries[0]
            segments = outbound.get("segments", [])
            if not segments:
                continue

            first_seg = segments[0]
            last_seg = segments[-1]

            carrier = first_seg.get("carrierCode", "")
            departure_time = first_seg.get("departure", {}).get("at", "")
            arrival_time = last_seg.get("arrival", {}).get("at", "")
            stops = len(segments) - 1

            flights_summary.append({
                "rank": i,
                "price": price,
                "carrier": carrier,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "stops": stops
            })

        if self.llm_client:
            try:
                formatting_prompt = f"""Format these flight search results into a concise SMS (max 280 chars).

Flight search:
- Route: {origin_code} -> {dest_code}
- Date: {departure} {f"(return {return_date})" if return_date else "(one-way)"}

Results: {json.dumps(flights_summary)}

Requirements:
1. Show top 3 flights with price, airline NAME, stops
2. Format times as 6:00am, 2:15pm
3. Mark direct flights
4. Keep under 280 chars
5. Note: "Prices may vary"

Return ONLY the formatted message:"""

                result = await self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You format flight search results for SMS."},
                        {"role": "user", "content": formatting_prompt}
                    ],
                    enable_thinking=False
                )

                return result.content.strip()

            except Exception as e:
                logger.error(f"LLM formatting failed: {e}")

        # Fallback formatting
        msg = f"{origin_code}->{dest_code} {departure}:\n\n"
        for f in flights_summary[:3]:
            stops_txt = "direct" if f["stops"] == 0 else f"{f['stops']} stop"
            msg += f"{f['rank']}. {f['carrier']} ${f['price']} ({stops_txt})\n"
        msg += f"\nPrices may vary."
        return msg
