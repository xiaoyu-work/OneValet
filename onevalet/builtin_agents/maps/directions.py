"""
DirectionsAgent - Get directions using Google Directions API
"""
import os
import logging
import json
import httpx
from typing import Dict, Any, List, Optional

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")


@valet(triggers=["directions", "how to get", "navigate", "route"])
class DirectionsAgent(StandardAgent):
    """Directions agent using Google Directions API"""

    destination = InputField(
        prompt="Where do you want to go?",
        description="Where to go (address or place name)",
    )
    origin = InputField(
        prompt="Where are you starting from? (or say 'home')",
        description="Where to start from (defaults to your home address)",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs
        )

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract destination and origin from user input"""
        maps_cache = self.context_hints.get("maps_cache", {})
        cached_places = maps_cache.get("places", [])

        if not self.llm_client:
            return {"destination": user_input}

        try:
            search_context = ""
            if cached_places:
                search_context = f"\nRecent search results (user can reference by number):\n"
                for i, place in enumerate(cached_places[:5], 1):
                    name = place.get("displayName", {}).get("text", "Unknown")
                    address = place.get("formattedAddress", "")
                    search_context += f"{i}. {name} - {address}\n"

            prompt = f"""Extract destination and origin from this directions request.

{search_context}
User message: "{user_input}"

Rules:
1. Destination: Where the user wants to go
   - If user references a number (#1, #2, "first", "second"), use that search result
   - Otherwise extract the address/place name
2. Origin: Where they're starting from
   - If mentioned ("from home", "from work"), extract it
   - If not mentioned, return "unknown" (will use profile default)
3. Return structured JSON

Return JSON format:
{{
  "destination": "place name or address",
  "origin": "start location or unknown",
  "search_result_index": number or null (1-based index if referencing search results)
}}

Return ONLY valid JSON:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract destinations and origins from directions requests. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            data = json.loads(content)

            destination = data.get("destination", "")
            search_index = data.get("search_result_index")

            if search_index and cached_places:
                idx = search_index - 1
                if 0 <= idx < len(cached_places):
                    place = cached_places[idx]
                    destination = place.get("formattedAddress") or place.get("displayName", {}).get("text", "")
                    logger.info(f"Resolved destination from search result #{search_index}: {destination}")

            origin = data.get("origin", "unknown")

            if origin == "unknown" or origin == "home":
                user_profile = self.context_hints.get("user_profile", {})
                if user_profile:
                    addresses = user_profile.get("addresses", [])
                    if isinstance(addresses, str):
                        try:
                            addresses = json.loads(addresses)
                        except Exception:
                            addresses = []

                    home_address = None
                    for addr in addresses:
                        if addr.get("label") == "home":
                            street = addr.get("street", "")
                            city = addr.get("city", "")
                            state = addr.get("state", "")
                            zip_code = addr.get("zip", "")
                            home_address = f"{street}, {city}, {state} {zip_code}".strip(", ")
                            break

                    if home_address:
                        origin = home_address
                        logger.info(f"Using home address as origin: {origin}")
                    else:
                        origin = ""

            return {
                "destination": destination,
                "origin": origin if origin and origin != "unknown" else ""
            }

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"destination": user_input}

    async def on_running(self, msg: Message) -> AgentResult:
        """Get directions between origin and destination"""
        destination = self.collected_fields.get("destination", "")
        origin = self.collected_fields.get("origin", "")

        if not destination:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="I need a destination. Where do you want to go?"
            )

        if not origin:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message="I need a starting point. Where are you starting from?"
            )

        logger.info(f"Getting directions from {origin} to {destination}")

        if not GOOGLE_MAPS_API_KEY:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Google Maps API key not configured. Please contact support."
            )

        try:
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                "origin": origin,
                "destination": destination,
                "mode": "driving",
                "alternatives": "false",
                "key": GOOGLE_MAPS_API_KEY
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=15.0)
                response.raise_for_status()
                data = response.json()

            if data["status"] != "OK":
                logger.error(f"Directions API error: {data['status']}")
                if data["status"] == "ZERO_RESULTS":
                    error_msg = f"Couldn't find a route from {origin} to {destination}."
                elif data["status"] == "NOT_FOUND":
                    error_msg = "One of the locations wasn't found. Please check the addresses."
                else:
                    error_msg = "Couldn't get directions. Please try again."

                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=error_msg
                )

            routes = data.get("routes", [])
            if not routes:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No route found from {origin} to {destination}."
                )

            route = routes[0]
            leg = route["legs"][0]

            distance = leg["distance"]["text"]
            duration = leg["duration"]["text"]
            start_address = leg["start_address"]
            end_address = leg["end_address"]
            steps = leg["steps"]

            maps_link = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"

            directions_data = {
                "start": start_address,
                "end": end_address,
                "distance": distance,
                "duration": duration,
                "steps": steps[:5],
                "maps_link": maps_link
            }

            logger.info(f"Directions: {distance}, {duration}")

            formatting_prompt = f"""Format these directions into a concise SMS message (max 250 chars).

Directions data:
- From: {directions_data['start']}
- To: {directions_data['end']}
- Distance: {directions_data['distance']}
- Duration: {directions_data['duration']}
- Google Maps link: {directions_data['maps_link']}

First few steps:
{json.dumps([s.get('html_instructions', '') for s in directions_data['steps'][:3]])}

Requirements:
1. Simplify addresses
2. Show distance and duration
3. Give 2-3 key initial steps (simplified, no HTML)
4. Include Google Maps link at end
5. Keep under 250 characters

Format the message (ONLY return the formatted message, nothing else):"""

            llm_result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You format driving directions for SMS. Be concise and natural."},
                    {"role": "user", "content": formatting_prompt}
                ],
                enable_thinking=False
            )

            formatted_message = llm_result.content.strip()

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=formatted_message
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Directions API HTTP error: {e.response.status_code}")
            if e.response.status_code == 400:
                error_msg = "Invalid location. Please check the addresses."
            elif e.response.status_code in [401, 403]:
                error_msg = "Google Maps API authentication failed. Please contact support."
            else:
                error_msg = "Couldn't get directions. Try again later?"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=error_msg
            )

        except Exception as e:
            logger.error(f"Directions API call failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Couldn't get directions. Try again later?"
            )
