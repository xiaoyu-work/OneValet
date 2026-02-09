"""
TripRepository - Data access for the trips table.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from onevalet.db import Repository

logger = logging.getLogger(__name__)


class TripRepository(Repository):

    TABLE_NAME = "trips"

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS trips (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id         TEXT NOT NULL,
        title           TEXT,
        trip_type       TEXT,
        carrier         TEXT,
        carrier_code    TEXT,
        trip_number     TEXT,
        booking_reference TEXT,
        origin_city     TEXT,
        origin_code     TEXT,
        destination_city TEXT,
        destination_code TEXT,
        departure_time  TIMESTAMPTZ,
        departure_local_time TEXT,
        departure_terminal TEXT,
        departure_gate  TEXT,
        arrival_time    TIMESTAMPTZ,
        arrival_local_time TEXT,
        arrival_terminal TEXT,
        arrival_gate    TEXT,
        hotel_name      TEXT,
        hotel_address   TEXT,
        check_in_date   DATE,
        check_out_date  DATE,
        rental_company  TEXT,
        pickup_time     TIMESTAMPTZ,
        dropoff_time    TIMESTAMPTZ,
        status          TEXT DEFAULT 'upcoming',
        source          TEXT,
        source_id       TEXT,
        source_account  TEXT,
        raw_data        JSONB,
        created_at      TIMESTAMPTZ DEFAULT now(),
        updated_at      TIMESTAMPTZ DEFAULT now()
    );
    """

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_user_trips(
        self,
        user_id: str,
        status: str = "upcoming",
        search_term: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return trips for a user with optional status / search filters."""
        conditions = ["user_id = $1", "status = $2"]
        args: list[Any] = [user_id, status]
        idx = 3

        if search_term:
            conditions.append(
                f"(title ILIKE ${idx} OR destination_city ILIKE ${idx} "
                f"OR destination_code ILIKE ${idx} OR trip_number ILIKE ${idx} "
                f"OR booking_reference ILIKE ${idx})"
            )
            args.append(f"%{search_term}%")
            idx += 1

        where = " AND ".join(conditions)
        return await self._fetch_many(
            where=where,
            args=tuple(args),
            order_by="departure_time ASC NULLS LAST",
            limit=limit,
        )

    async def get_today_trips(self, user_id: str) -> List[Dict[str, Any]]:
        """Return trips departing today for a user."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)

        return await self._fetch_many(
            where="user_id = $1 AND departure_time >= $2 AND departure_time < $3 AND status = 'upcoming'",
            args=(user_id, today, tomorrow),
            order_by="departure_time ASC",
        )

    async def insert_trip(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert a trip row and return it."""
        if "raw_data" in data and not isinstance(data["raw_data"], str):
            data = {**data, "raw_data": json.dumps(data["raw_data"])}
        return await self._insert(data)

    async def update_trip(
        self, trip_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a trip by id and return it."""
        if "raw_data" in data and not isinstance(data["raw_data"], str):
            data = {**data, "raw_data": json.dumps(data["raw_data"])}
        return await self._update("id", trip_id, data)

    async def soft_delete_trip(self, trip_id: str) -> Optional[Dict[str, Any]]:
        """Mark a trip as cancelled (soft delete)."""
        return await self._update("id", trip_id, {"status": "cancelled"})

    # ------------------------------------------------------------------
    # Deduplication finders
    # ------------------------------------------------------------------

    async def find_by_source_id(
        self, user_id: str, source_id: str
    ) -> Optional[Dict[str, Any]]:
        row = await self.db.fetchrow(
            "SELECT * FROM trips WHERE user_id = $1 AND source_id = $2",
            user_id,
            source_id,
        )
        return dict(row) if row else None

    async def find_by_booking_reference(
        self, user_id: str, booking_ref: str
    ) -> Optional[Dict[str, Any]]:
        row = await self.db.fetchrow(
            "SELECT * FROM trips WHERE user_id = $1 AND booking_reference = $2",
            user_id,
            booking_ref,
        )
        return dict(row) if row else None

    async def find_by_trip_number_and_date(
        self, user_id: str, trip_number: str, departure_date: datetime
    ) -> Optional[Dict[str, Any]]:
        date_start = departure_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)

        row = await self.db.fetchrow(
            "SELECT * FROM trips WHERE user_id = $1 AND trip_number = $2 "
            "AND departure_time >= $3 AND departure_time < $4",
            user_id,
            trip_number,
            date_start,
            date_end,
        )
        return dict(row) if row else None

    async def find_by_route_and_date(
        self,
        user_id: str,
        origin_code: str,
        dest_code: str,
        departure_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        date_start = departure_date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)

        row = await self.db.fetchrow(
            "SELECT * FROM trips WHERE user_id = $1 AND origin_code = $2 "
            "AND destination_code = $3 AND departure_time >= $4 AND departure_time < $5",
            user_id,
            origin_code,
            dest_code,
            date_start,
            date_end,
        )
        return dict(row) if row else None

    async def find_by_hotel(
        self, user_id: str, hotel_name: str, check_in_date: str
    ) -> Optional[Dict[str, Any]]:
        row = await self.db.fetchrow(
            "SELECT * FROM trips WHERE user_id = $1 AND hotel_name = $2 AND check_in_date = $3",
            user_id,
            hotel_name,
            check_in_date,
        )
        return dict(row) if row else None
