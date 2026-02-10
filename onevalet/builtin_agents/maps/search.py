"""
MapSearchAgent - Search for places using Google Maps Platform
"""
import os
import logging
import json
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message

logger = logging.getLogger(__name__)


@valet(triggers=["find", "search", "where", "near", "nearby", "restaurant", "store", "shop"])
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
        self.search_results = []
        self.last_query = ""

    def needs_approval(self) -> bool:
        return False

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract search query and location from user input"""
        if self.search_results and user_input.strip().isdigit():
            selected_num = int(user_input.strip())
            if 1 <= selected_num <= len(self.search_results):
                return {
                    "detail_request": True,
                    "place_number": selected_num,
                    "query": self.collected_fields.get("query", ""),
                    "location": self.collected_fields.get("location", "")
                }

        if not self.llm_client:
            return {"query": user_input, "location": ""}

        try:
            prompt = f"""Extract the search query and location from this message.

User message: "{user_input}"

Extract:
- query: What type of place/business (don't include "address", "location" in query)
- location: City or neighborhood (leave empty if not mentioned)
- is_followup: true if asking for "more", "details", "directions"

Return JSON:
{{
  "query": "",
  "location": "",
  "is_followup": false
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

            if data.get("is_followup"):
                return {"is_followup": True, "followup_intent": data.get("followup_intent")}

            return {
                "query": data.get("query", ""),
                "location": data.get("location", "")
            }

        except Exception as e:
            logger.error(f"Field extraction failed: {e}")
            return {"query": user_input, "location": ""}

    async def on_running(self, msg: Message) -> AgentResult:
        """Search for places using Google Places API"""
        fields = self.collected_fields

        if fields.get("detail_request"):
            place_number = fields.get("place_number", 1)
            self.collected_fields.pop("detail_request", None)
            self.collected_fields.pop("place_number", None)

            if self.search_results and 1 <= place_number <= len(self.search_results):
                place = self.search_results[place_number - 1]
                formatted = await self._format_results([place], self.last_query)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=formatted
                )
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Couldn't find that option. Try searching again?"
                )

        if fields.get("is_followup"):
            self.collected_fields.pop("is_followup", None)
            self.collected_fields.pop("followup_intent", None)

            result = await self._handle_followup(fields.get("followup_intent"))
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message=result
            )

        self.collected_fields.pop("detail_request", None)
        self.collected_fields.pop("place_number", None)
        self.collected_fields.pop("is_followup", None)
        self.collected_fields.pop("followup_intent", None)

        query = fields.get("query", "")
        location = fields.get("location", "")
        self.last_query = query

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
                    raw_message=f"Couldn't find any {query} in {location}. Try a different search?"
                )

            self.search_results = places

            formatted = await self._format_results(places, query)

            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=formatted
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

    async def _format_results(self, places: List[Dict], query: str) -> str:
        """Format search results into concise SMS message"""
        if not self.llm_client:
            result = f"Found {len(places)} {query}:\n\n"
            for i, place in enumerate(places[:3], 1):
                name = place.get("displayName", {}).get("text", "Unknown")
                address = place.get("formattedAddress", "No address")
                rating = place.get("rating", "N/A")
                result += f"{i}. {name}\n   {address}\n   * {rating}\n\n"
            return result.strip()

        places_summary = []
        for i, place in enumerate(places, 1):
            name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "No address")
            rating = place.get("rating")
            rating_count = place.get("userRatingCount", 0)
            price_level = place.get("priceLevel", "")

            phone = place.get("internationalPhoneNumber", "")
            opening_hours = place.get("regularOpeningHours", {})
            hours_text = ""
            if opening_hours:
                weekday_descriptions = opening_hours.get("weekdayDescriptions", [])
                if weekday_descriptions:
                    hours_text = weekday_descriptions[0] if len(weekday_descriptions) > 0 else ""

            place_info = {
                "number": i,
                "name": name,
                "address": address,
                "rating": rating,
                "rating_count": rating_count,
                "price_level": price_level,
                "phone": phone,
                "hours": hours_text
            }
            places_summary.append(place_info)

        is_detail_view = len(places_summary) == 1

        if is_detail_view:
            formatting_prompt = f"""Format this place information into a concise SMS message (max 300 chars).

Place data: {json.dumps(places_summary[0])}

Requirements:
1. Show name, address, rating with star
2. ONLY show phone number if "phone" field is not empty
3. ONLY show hours if "hours" field is not empty
4. DO NOT make up or invent any information - only use the data provided
5. If phone or hours are missing/empty, skip them entirely
6. Keep it natural and concise

Return ONLY the formatted message:"""
        else:
            formatting_prompt = f"""Format these search results into a concise SMS message (max 300 chars).

Search query: {query}
Results: {json.dumps(places_summary)}

Requirements:
1. Show top 3 results (numbered)
2. For each: name, brief address (shorten if needed), rating with star
3. Use line breaks for readability
4. Keep it under 300 characters
5. Add "Reply with number for details" at the end
6. Be natural and concise
7. DO NOT show phone numbers or hours in list view

Return ONLY the formatted message:"""

        try:
            llm_result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You format search results for SMS. Be concise and natural. Only use the data provided - never make up information."},
                    {"role": "user", "content": formatting_prompt}
                ],
                enable_thinking=False
            )

            return llm_result.content.strip()

        except Exception as e:
            logger.error(f"LLM formatting failed: {e}")
            result = f"Found {query}:\n\n"
            for i, p in enumerate(places_summary[:3], 1):
                result += f"{i}. {p['name']}\n   {p['address'][:30]}... * {p['rating']}\n\n"
            result += "Reply with number for details"
            return result

    async def _handle_followup(self, intent: str) -> str:
        """Handle follow-up queries like show more, details, directions"""
        if not self.search_results:
            return "I don't have any search results to show. Try a new search?"

        if intent == "more":
            if len(self.search_results) > 3:
                return await self._format_results(self.search_results[3:], self.last_query)
            else:
                return "That's all I found. Try a different search?"

        elif intent == "details":
            return "Which one? Reply with the number (1, 2, 3, etc.)"

        elif intent == "directions":
            return "Which place? Reply with the number and I'll get you directions."

        else:
            return "Not sure what you mean. Try asking for a new search?"
