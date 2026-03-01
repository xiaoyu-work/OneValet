"""
TripRepository - Data access for the trips table.

Used by TripPlannerAgent to save, list, update, and delete user trips.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from onevalet.db import Repository

logger = logging.getLogger(__name__)


class TripRepository(Repository):
    TABLE_NAME = "trips"

    async def get_tenant_trips(
        self, tenant_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trips for a tenant, optionally filtered by status."""
        if status:
            rows = await self.db.fetch(
                "SELECT * FROM trips WHERE tenant_id = $1 AND status = $2 "
                "ORDER BY departure_time ASC NULLS LAST",
                tenant_id, status,
            )
        else:
            rows = await self.db.fetch(
                "SELECT * FROM trips WHERE tenant_id = $1 "
                "ORDER BY departure_time ASC NULLS LAST",
                tenant_id,
            )
        return [self._row_to_dict(r) for r in rows]

    async def upsert_trip(
        self, tenant_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Insert a new trip. Returns the created row."""
        insert_data = {"tenant_id": tenant_id, **data}
        if "raw_data" in insert_data and isinstance(insert_data["raw_data"], dict):
            insert_data["raw_data"] = json.dumps(insert_data["raw_data"])
        return await self._insert(insert_data)

    async def update_trip(
        self, trip_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a trip by id. Returns the updated row."""
        if "raw_data" in data and isinstance(data["raw_data"], dict):
            data["raw_data"] = json.dumps(data["raw_data"])
        return await self._update("id", trip_id, data)

    async def delete_trip(self, trip_id: str) -> bool:
        """Delete a trip by id."""
        return await self._delete("id", trip_id)

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        """Convert a row to dict, deserializing JSONB fields."""
        d = dict(row)
        if isinstance(d.get("raw_data"), str):
            try:
                d["raw_data"] = json.loads(d["raw_data"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
