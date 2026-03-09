"""
Google Calendar Provider - Implementation for Google Calendar API

Implements BaseCalendarProvider for Google Calendar.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import httpx

from .base import BaseCalendarProvider
from ..http_mixin import OAuthHTTPMixin

logger = logging.getLogger(__name__)


class GoogleCalendarProvider(BaseCalendarProvider, OAuthHTTPMixin):
    """Google Calendar provider using Google Calendar API v3."""

    def __init__(
        self,
        credentials: dict,
        on_token_refreshed: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(credentials, on_token_refreshed)
        self.api_base_url = "https://www.googleapis.com/calendar/v3"

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    async def list_events(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10,
        query: Optional[str] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List Google Calendar events."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"
        if not time_min:
            time_min = datetime.now(timezone.utc)
        if not time_max:
            time_max = time_min + timedelta(days=7)

        try:
            params: Dict[str, Any] = {
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "maxResults": max_results,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if query:
                params["q"] = query

            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/calendars/{calendar_id}/events",
                params=params,
            )

            response.raise_for_status()
            data = response.json()

            events = []
            for item in data.get("items", []):
                start_data = item.get("start", {})
                end_data = item.get("end", {})
                start_str = start_data.get("dateTime") or start_data.get("date")
                end_str = end_data.get("dateTime") or end_data.get("date")

                from dateutil import parser as date_parser
                start_dt = date_parser.parse(start_str) if start_str else None
                end_dt = date_parser.parse(end_str) if end_str else None

                attendees = [a.get("email", "") for a in item.get("attendees", [])]

                events.append({
                    "event_id": item.get("id"),
                    "summary": item.get("summary", "No title"),
                    "description": item.get("description", ""),
                    "start": start_dt,
                    "end": end_dt,
                    "location": item.get("location", ""),
                    "attendees": attendees,
                    "organizer": item.get("organizer", {}).get("email", ""),
                    "status": item.get("status", "confirmed"),
                    "html_link": item.get("htmlLink", ""),
                })

            logger.info(f"Retrieved {len(events)} events from Google Calendar")
            return {"success": True, "data": events, "count": len(events)}

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Calendar API error: {e.response.status_code} - {e.response.text}")
            return {"success": False, "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to list events: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Google Calendar event."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"

        try:
            event_body: Dict[str, Any] = {
                "summary": summary,
                "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
            }
            if description:
                event_body["description"] = description
            if location:
                event_body["location"] = location
            if attendees:
                event_body["attendees"] = [{"email": email} for email in attendees]

            response = await self._oauth_request(
                "POST",
                f"{self.api_base_url}/calendars/{calendar_id}/events",
                json=event_body,
            )

            response.raise_for_status()
            data = response.json()
            logger.info(f"Created event: {summary}")
            return {
                "success": True,
                "event_id": data.get("id"),
                "html_link": data.get("htmlLink", ""),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Calendar API error: {e.response.status_code} - {e.response.text}")
            return {"success": False, "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to create event: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def update_event(
        self,
        event_id: str,
        summary: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a Google Calendar event."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"

        try:
            update_body: Dict[str, Any] = {}
            if summary is not None:
                update_body["summary"] = summary
            if start is not None:
                update_body["start"] = {"dateTime": start.isoformat(), "timeZone": "UTC"}
            if end is not None:
                update_body["end"] = {"dateTime": end.isoformat(), "timeZone": "UTC"}
            if description is not None:
                update_body["description"] = description
            if location is not None:
                update_body["location"] = location
            if attendees is not None:
                update_body["attendees"] = [{"email": email} for email in attendees]

            response = await self._oauth_request(
                "PATCH",
                f"{self.api_base_url}/calendars/{calendar_id}/events/{event_id}",
                json=update_body,
            )

            response.raise_for_status()
            data = response.json()
            logger.info(f"Updated event: {event_id}")
            return {
                "success": True,
                "event_id": data.get("id"),
                "html_link": data.get("htmlLink", ""),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Calendar API error: {e.response.status_code} - {e.response.text}")
            return {"success": False, "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to update event: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a Google Calendar event."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"

        try:
            response = await self._oauth_request(
                "DELETE",
                f"{self.api_base_url}/calendars/{calendar_id}/events/{event_id}",
            )

            if response.status_code == 204:
                logger.info(f"Deleted event: {event_id}")
                return {"success": True}

            response.raise_for_status()
            return {"success": True}

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Calendar API error: {e.response.status_code} - {e.response.text}")
            return {"success": False, "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to delete event: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

