"""
MapSearchAgent - Search for places using Google Maps Platform
"""
import os
import logging
import json
import httpx
from typing import Dict, Any, List

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet()
class MapSearchAgent(StandardAgent):
    """Map search agent using Google Places API"""

    query = InputField(
        prompt="What are you looking for?",
        description="What to search for (e.g., 'coffee', 'pizza', 'gas station')",
    )
    location = InputField(
        prompt="Where are you looking? (city or neighborhood)",
        description="Where to search (city name or address)",
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
        """Extract search query and location from user input"""
        if not self.llm_client:
            return {"query": user_input, "location": ""}

        try:
            prompt = f"""Extract the search query and location from this message.

User message: "{user_input}"

Extract:
- query: What type of place/business (don't include "address", "location" in query)
- location: City or neighborhood (leave empty if not mentioned)

Return JSON:
{{
  "query": "",
  "location": ""
}}"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Extract search queries and locations. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format="json_object",
                enable_thinking=False
            )

            content = result.content.strip()
            data = json.loads(content)

            return {
                "query": data.get("query", ""),
                "location": data.get("location", "")
            }

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"query": user_input, "location": ""}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search for places using Google Places API"""
        query = self.collected_fields.get("query", "")
        location = self.collected_fields.get("location", "")

        logger.info(f"Searching for '{query}' in '{location}'")

        if not os.getenv("GOOGLE_MAPS_API_KEY"):
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Google Maps API key not configured. Please contact support."
            )

        try:
            url = "https://places.googleapis.com/v1/places:searchText"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": os.getenv("GOOGLE_MAPS_API_KEY", ""),
                "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.types,places.priceLevel,places.businessStatus,places.googleMapsUri,places.internationalPhoneNumber,places.regularOpeningHours,places.websiteUri"
            }

            text_query = f"{query} in {location}"
            request_body = {
                "textQuery": text_query,
                "maxResultCount": 5,
                "languageCode": "en"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=15.0
                )
                response.raise_for_status()
                data = response.json()

            places = data.get("places", [])

            if not places:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"No results found for {query} in {location}."
                )

            result_lines = [f"Found {len(places)} results for \"{query}\" in \"{location}\":\n"]
            for i, place in enumerate(places, 1):
                name = place.get("displayName", {}).get("text", "Unknown")
                address = place.get("formattedAddress", "")
                rating = place.get("rating")
                rating_count = place.get("userRatingCount", 0)
                phone = place.get("internationalPhoneNumber", "")
                maps_uri = place.get("googleMapsUri", "")
                website = place.get("websiteUri", "")
                price_level = place.get("priceLevel", "")
                hours = place.get("regularOpeningHours", {})
                hours_text = "; ".join(hours.get("weekdayDescriptions", [])) if hours else ""

                result_lines.append(f"{i}. {name}")
                if address:
                    result_lines.append(f"   Address: {address}")
                if rating:
                    result_lines.append(f"   Rating: {rating}/5 ({rating_count} reviews)")
                if price_level:
                    result_lines.append(f"   Price: {price_level}")
                if phone:
                    result_lines.append(f"   Phone: {phone}")
                if hours_text:
                    result_lines.append(f"   Hours: {hours_text}")
                if maps_uri:
                    result_lines.append(f"   Google Maps: {maps_uri}")
                if website:
                    result_lines.append(f"   Website: {website}")
                result_lines.append("")

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="\n".join(result_lines).strip()
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Places API HTTP error: {e.response.status_code}")
            if e.response.status_code == 400:
                error_msg = "Invalid search query. Try being more specific?"
            elif e.response.status_code in [401, 403]:
                error_msg = "Google Maps API authentication failed. Please contact support."
            else:
                error_msg = f"Couldn't search for {query}. Try again later?"

            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=error_msg
            )

        except Exception as e:
            logger.error(f"Map search failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=f"Couldn't search for {query}. Try again later?"
            )
