"""
HotelSearchAgent - Search for hotels using Amadeus Hotel Search API
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
class HotelSearchAgent(StandardAgent):
    """Hotel search agent using Amadeus Hotel Search API"""

    location = InputField(
        prompt="Where do you need a hotel?",
        description="City or area name",
    )
    check_in = InputField(
        prompt="When do you want to check in?",
        description="Check-in date",
    )
    check_out = InputField(
        prompt="When are you checking out? (or say how many nights)",
        description="Check-out date",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )
        self.access_token = None

    def needs_approval(self) -> bool:
        return False

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

    async def _geocode_location(self, location: str) -> Optional[Dict[str, float]]:
        """Convert location name to coordinates for hotel search"""
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        if not google_api_key:
            logger.warning("Google Maps API key not found")
            return None

        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "address": location,
                "key": google_api_key
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            if data["status"] != "OK" or not data.get("results"):
                return None

            coords = data["results"][0]["geometry"]["location"]
            return {
                "latitude": coords["lat"],
                "longitude": coords["lng"]
            }

        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
            return None

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract hotel search parameters from user input"""
        if not self.llm_client:
            return {"location": "", "check_in": "", "check_out": ""}

        try:
            today = datetime.now().strftime('%Y-%m-%d')

            prompt = f"""Extract hotel search details from this message.

User message: "{user_input}"

Extract:
1. location: city or area name
2. check_in: check-in date (convert to YYYY-MM-DD, assume current year if not specified)
3. check_out: check-out date (if mentioned) or null
4. nights: number of nights (if mentioned) or null

Today's date: {today}

Return JSON format:
{{
  "location": "city name",
  "check_in": "YYYY-MM-DD",
  "check_out": "YYYY-MM-DD or null",
  "nights": number or null
}}

Return ONLY valid JSON:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract hotel search parameters. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            data = json.loads(content)

            check_out = data.get("check_out", "")
            if not check_out and data.get("nights"):
                check_in_date = datetime.strptime(data["check_in"], "%Y-%m-%d")
                check_out_date = check_in_date + timedelta(days=data["nights"])
                check_out = check_out_date.strftime("%Y-%m-%d")

            return {
                "location": data.get("location", ""),
                "check_in": data.get("check_in", ""),
                "check_out": check_out if check_out else ""
            }

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search for hotels using Amadeus API"""
        location = self.collected_fields.get("location", "")
        check_in = self.collected_fields.get("check_in", "")
        check_out = self.collected_fields.get("check_out", "")

        # Default to 1 night if check_out not specified
        if not check_out and check_in:
            check_in_date = datetime.strptime(check_in, "%Y-%m-%d")
            check_out_date = check_in_date + timedelta(days=1)
            check_out = check_out_date.strftime("%Y-%m-%d")

        logger.info(f"Searching hotels in {location}: {check_in} to {check_out}")

        if not os.getenv("AMADEUS_API_KEY") or not os.getenv("AMADEUS_API_SECRET"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Amadeus API credentials not configured. Please contact support."
            )

        token = await self._get_access_token()
        if not token:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't connect to hotel search service. Try again later?"
            )

        try:
            coords = await self._geocode_location(location)

            if not coords:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"Couldn't find location: {location}. Please be more specific?"
                )

            url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-geocode"
            params = {
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "radius": 5,
                "radiusUnit": "KM"
            }
            headers = {"Authorization": f"Bearer {token}"}

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, timeout=15.0)
                response.raise_for_status()
                hotels_data = response.json()

            hotel_ids = [h.get("hotelId") for h in hotels_data.get("data", [])[:10]]

            if not hotel_ids:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No hotels found in {location}. Try a different area?"
                )

            offers_url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
            offers_params = {
                "hotelIds": ",".join(hotel_ids[:5]),
                "checkInDate": check_in,
                "checkOutDate": check_out,
                "adults": "1",
                "currency": "USD"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(offers_url, params=offers_params, headers=headers, timeout=20.0)
                response.raise_for_status()
                offers_data = response.json()

            offers = offers_data.get("data", [])

            if not offers:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No available hotels in {location} for {check_in} to {check_out}. Try different dates?"
                )

            logger.info(f"Found {len(offers)} hotel offers")
            formatted = await self._format_results(offers, location, check_in, check_out)

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=formatted
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Amadeus API error: {e.response.status_code}")
            if e.response.status_code == 400:
                error_msg = "Invalid search: please check your dates."
            elif e.response.status_code == 401:
                error_msg = "Hotel search authentication failed. Please contact support."
            else:
                error_msg = "Couldn't search hotels. Try again later?"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=error_msg
            )

        except Exception as e:
            logger.error(f"Hotel search failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't search hotels. Try again later?"
            )

    async def _format_results(self, offers: List[Dict], location: str, check_in: str, check_out: str) -> str:
        """Format hotel offers into concise SMS message"""
        hotels_summary = []
        for i, offer in enumerate(offers[:5], 1):
            hotel = offer.get("hotel", {})
            hotel_offers = offer.get("offers", [])

            if not hotel_offers:
                continue

            best_offer = hotel_offers[0]

            name = hotel.get("name", "Unknown Hotel")
            price = best_offer.get("price", {}).get("total", "N/A")
            rating = hotel.get("rating", "N/A")

            hotels_summary.append({
                "rank": i,
                "name": name,
                "price": price,
                "rating": rating
            })

        booking_link = f"https://www.booking.com/searchresults.html?ss={location}&checkin={check_in}&checkout={check_out}"

        if self.llm_client:
            try:
                formatting_prompt = f"""Format these hotel search results into a concise SMS (max 280 chars).

Hotel search:
- Location: {location}
- Check-in: {check_in}
- Check-out: {check_out}

Results: {json.dumps(hotels_summary)}

Booking link: {booking_link}

Requirements:
1. Show top 5 hotels with name, price/night, rating
2. Mark highly rated (4.5+) with star
3. Add booking link at end
4. Keep under 280 chars
5. Note: "Prices may vary"

Return ONLY the formatted message:"""

                result = await self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You format hotel search results for SMS."},
                        {"role": "user", "content": formatting_prompt}
                    ],
                    enable_thinking=False
                )

                return result.content.strip()

            except Exception as e:
                logger.error(f"LLM formatting failed: {e}")

        # Fallback formatting
        msg = f"{location} {check_in} to {check_out}:\n\n"
        for h in hotels_summary[:5]:
            star = "*" if isinstance(h["rating"], (int, float)) and h["rating"] >= 4.5 else ""
            msg += f"{h['rank']}. {h['name'][:20]} ${h['price']}/nt {star}{h['rating']}\n"
        msg += f"\nPrices may vary. Book:\n{booking_link}"
        return msg
