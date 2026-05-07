"""
Google Calendar Provider - Implementation for Google Calendar API

Implements BaseCalendarProvider for Google Calendar.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import httpx

from ..http_mixin import OAuthHTTPMixin
from .base import BaseCalendarProvider

logger = logging.getLogger(__name__)


def _format_google_endpoint(dt: datetime, *, all_day: bool) -> Dict[str, str]:
    """Translate our internal datetime into Google's ``start``/``end`` shape.

    For all-day events Google expects ``{"date": "YYYY-MM-DD"}`` with the
    end date being **exclusive**. Our internal representation already
    stores end as an exclusive local midnight, so we just emit
    ``date = dt.date()``. For timed events we send ``dateTime`` in UTC.
    """
    if all_day:
        return {"date": dt.date().isoformat()}
    return {"dateTime": dt.isoformat(), "timeZone": "UTC"}


def _parse_google_event(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Google API event payload into the shape expected by
    ``CalendarSyncService._upsert_event``.

    All-day events come back as ``{"date": "YYYY-MM-DD"}`` (end exclusive).
    We promote them to a tz-aware UTC midnight datetime so downstream
    storage treats them like every other row, while flagging
    ``all_day=True`` for the upsert.
    """
    from dateutil import parser as date_parser

    start_data = item.get("start") or {}
    end_data = item.get("end") or {}
    start_str = start_data.get("dateTime") or start_data.get("date")
    end_str = end_data.get("dateTime") or end_data.get("date")
    all_day = bool(start_data.get("date") and not start_data.get("dateTime"))

    start_dt = date_parser.parse(start_str) if start_str else None
    end_dt = date_parser.parse(end_str) if end_str else None

    attendees = [a.get("email", "") for a in item.get("attendees", []) if a.get("email")]

    return {
        "event_id": item.get("id"),
        "summary": item.get("summary", "No title"),
        "description": item.get("description", ""),
        "start": start_dt,
        "end": end_dt,
        "all_day": all_day,
        "location": item.get("location", ""),
        "attendees": attendees,
        "organizer": item.get("organizer", {}).get("email", ""),
        "status": item.get("status", "confirmed"),
        "html_link": item.get("htmlLink", ""),
        "ical_uid": item.get("iCalUID"),
    }


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
        if time_min is None or time_max is None:
            return {
                "success": False,
                "error": (
                    "time_min and time_max are required (no implicit window). "
                    "Callers must compute an explicit ISO-8601 range."
                ),
            }

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

            events = [_parse_google_event(item) for item in data.get("items", [])]

            logger.info(f"Retrieved {len(events)} events from Google Calendar")
            return {"success": True, "data": events, "count": len(events)}

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Calendar API error: {e.response.status_code} - {e.response.text}")
            return {"success": False, "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to list events: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a single event from Google Calendar.

        Returns the same normalized shape as ``list_events`` rows so the
        result can be passed directly into ``CalendarSyncService._upsert_event``.
        """
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"

        try:
            response = await self._oauth_request(
                "GET",
                f"{self.api_base_url}/calendars/{calendar_id}/events/{event_id}",
            )
            response.raise_for_status()
            item = response.json()
            return {"success": True, "data": _parse_google_event(item)}
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Google Calendar API error (get_event): {e.response.status_code} - {e.response.text}"
            )
            return {"success": False, "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Failed to get event: {e}", exc_info=True)
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
        all_day: bool = False,
    ) -> Dict[str, Any]:
        """Create a Google Calendar event.

        ``all_day=True`` switches the start/end shape to Google's date-only
        form ``{"date": "YYYY-MM-DD"}``. Per Google's all-day convention,
        ``end.date`` is **exclusive** (so a single-day all-day event has
        ``end.date == start.date + 1``). Callers pass our internal
        end-as-exclusive-midnight datetime and we convert.
        """
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"

        try:
            event_body: Dict[str, Any] = {
                "summary": summary,
                "start": _format_google_endpoint(start, all_day=all_day),
                "end": _format_google_endpoint(end, all_day=all_day),
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
        all_day: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update a Google Calendar event.

        When ``all_day`` is provided we use Google's date-only shape;
        otherwise we default to ``dateTime``. ``all_day=True`` callers
        must pass our internal end-as-exclusive-midnight datetime; this
        method converts to Google's ``{"date": "YYYY-MM-DD"}`` form.
        """
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        if not calendar_id:
            calendar_id = "primary"

        try:
            update_body: Dict[str, Any] = {}
            if summary is not None:
                update_body["summary"] = summary
            if start is not None:
                update_body["start"] = _format_google_endpoint(start, all_day=bool(all_day))
            if end is not None:
                update_body["end"] = _format_google_endpoint(end, all_day=bool(all_day))
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

    # ── Push Notifications (Watch) ──

    async def setup_watch(
        self,
        webhook_url: str,
        channel_id: Optional[str] = None,
        calendar_id: str = "primary",
    ) -> Dict[str, Any]:
        """Set up push notifications for calendar event changes.

        Google Calendar watches expire after ~7 days. Call periodically to renew.

        Args:
            webhook_url: HTTPS callback URL for notifications.
            channel_id: Unique channel ID (auto-generated if not provided).
            calendar_id: Calendar to watch (default: primary).

        Returns:
            {"success": True, "channel_id": ..., "resource_id": ..., "expiration": ...}
        """
        import uuid

        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        cid = channel_id or str(uuid.uuid4())

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.api_base_url}/calendars/{calendar_id}/events/watch",
                    headers=self._get_headers(),
                    json={
                        "id": cid,
                        "type": "web_hook",
                        "address": webhook_url,
                    },
                    timeout=30.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(f"✅ Google Calendar watch set up (channel={cid})")
                    return {
                        "success": True,
                        "channel_id": data.get("id"),
                        "resource_id": data.get("resourceId"),
                        "expiration": int(data.get("expiration", 0)),
                    }
                else:
                    logger.error(f"❌ Calendar watch failed: {resp.text}")
                    return {"success": False, "error": f"Watch failed: {resp.status_code}"}

        except Exception as e:
            logger.error(f"❌ Calendar watch error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def stop_watch(
        self,
        channel_id: str,
        resource_id: str,
    ) -> Dict[str, Any]:
        """Stop a push notification channel."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://www.googleapis.com/calendar/v3/channels/stop",
                    headers=self._get_headers(),
                    json={"id": channel_id, "resourceId": resource_id},
                    timeout=15.0,
                )
                return {"success": resp.status_code in (200, 204)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_events_sync_token(
        self,
        calendar_id: str = "primary",
    ) -> Dict[str, Any]:
        """Get a sync token for incremental event sync."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base_url}/calendars/{calendar_id}/events",
                    headers=self._get_headers(),
                    params={"maxResults": 1, "showDeleted": False},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return {"success": True, "sync_token": data.get("nextSyncToken")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_events_incremental(
        self,
        sync_token: str,
        calendar_id: str = "primary",
    ) -> Dict[str, Any]:
        """Fetch events changed since the last sync token."""
        if not await self.ensure_valid_token():
            return {"success": False, "error": "Failed to refresh access token"}

        try:
            all_events: List[Dict[str, Any]] = []
            page_token: Optional[str] = None

            while True:
                params: Dict[str, Any] = {"syncToken": sync_token, "showDeleted": True}
                if page_token:
                    params["pageToken"] = page_token

                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self.api_base_url}/calendars/{calendar_id}/events",
                        headers=self._get_headers(),
                        params=params,
                        timeout=30.0,
                    )

                if resp.status_code == 410:
                    # Sync token expired, full sync needed
                    return {
                        "success": False,
                        "error": "sync_token_expired",
                        "full_sync_required": True,
                    }

                resp.raise_for_status()
                data = resp.json()
                all_events.extend(data.get("items", []))

                page_token = data.get("nextPageToken")
                if not page_token:
                    return {
                        "success": True,
                        "events": all_events,
                        "next_sync_token": data.get("nextSyncToken"),
                    }

        except Exception as e:
            logger.error(f"Incremental sync failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
