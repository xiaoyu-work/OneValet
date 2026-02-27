"""
Shipment Repository - Data access for shipments table.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from onevalet.db import Repository

logger = logging.getLogger(__name__)


class ShipmentRepository(Repository):
    TABLE_NAME = "shipments"
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS shipments (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id         TEXT NOT NULL,
        tracking_number TEXT NOT NULL,
        carrier         TEXT,
        tracking_url    TEXT,
        status          TEXT DEFAULT 'unknown',
        description     TEXT,
        last_update     TEXT,
        estimated_delivery TEXT,
        tracking_history JSONB DEFAULT '[]',
        delivered_notified BOOLEAN,
        is_active       BOOLEAN DEFAULT TRUE,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        updated_at      TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (tenant_id, tracking_number)
    );
    """
    SETUP_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_shipments_tenant_id ON shipments (tenant_id)",
    ]

    async def get_user_shipments(
        self, tenant_id: str, is_active: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all shipments for a user filtered by active status."""
        rows = await self.db.fetch(
            "SELECT * FROM shipments WHERE tenant_id = $1 AND is_active = $2 "
            "ORDER BY updated_at DESC",
            tenant_id,
            is_active,
        )
        return [dict(r) for r in rows]

    async def upsert_shipment(
        self,
        tenant_id: str,
        tracking_number: str,
        carrier: str,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Insert or update a shipment keyed on (tenant_id, tracking_number)."""
        # Build the full data dict
        data: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "tracking_number": tracking_number,
            "carrier": carrier,
        }

        allowed = {
            "tracking_url",
            "status",
            "description",
            "last_update",
            "estimated_delivery",
            "tracking_history",
            "delivered_notified",
            "is_active",
        }

        for key, value in kwargs.items():
            if key in allowed and value is not None:
                data[key] = value

        # Serialize tracking_history to JSON string for JSONB column
        if "tracking_history" in data:
            data["tracking_history"] = json.dumps(data["tracking_history"])

        columns = list(data.keys())
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        values = list(data.values())

        # Build SET clause for ON CONFLICT — update everything except the key columns
        update_cols = [c for c in columns if c not in ("tenant_id", "tracking_number")]
        set_clause = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in update_cols
        )
        set_clause += ", updated_at = NOW()"

        query = (
            f"INSERT INTO shipments ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT (tenant_id, tracking_number) DO UPDATE SET {set_clause} "
            f"RETURNING *"
        )

        row = await self.db.fetchrow(query, *values)
        return dict(row) if row else None

    async def update_shipment(
        self, shipment_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a shipment by its id."""
        if "tracking_history" in data:
            data["tracking_history"] = json.dumps(data["tracking_history"])
        data["updated_at"] = "NOW()"
        # Use _update helper but handle updated_at specially
        # Build manually so we can use NOW() as a SQL expression
        set_clauses = []
        values = []
        idx = 1
        for col, val in data.items():
            if col == "updated_at":
                set_clauses.append("updated_at = NOW()")
            else:
                set_clauses.append(f"{col} = ${idx}")
                values.append(val)
                idx += 1

        values.append(shipment_id)

        query = (
            f"UPDATE shipments SET {', '.join(set_clauses)} "
            f"WHERE id = ${idx}::uuid RETURNING *"
        )
        row = await self.db.fetchrow(query, *values)
        return dict(row) if row else None

    async def archive_shipment(self, shipment_id: str) -> Optional[Dict[str, Any]]:
        """Set is_active=FALSE for a shipment by id."""
        row = await self.db.fetchrow(
            "UPDATE shipments SET is_active = FALSE, updated_at = NOW() "
            "WHERE id = $1::uuid RETURNING *",
            shipment_id,
        )
        return dict(row) if row else None

    async def archive_shipment_by_tracking(
        self, tenant_id: str, tracking_number: str
    ) -> Optional[Dict[str, Any]]:
        """Set is_active=FALSE for a shipment by tenant_id + tracking_number."""
        row = await self.db.fetchrow(
            "UPDATE shipments SET is_active = FALSE, updated_at = NOW() "
            "WHERE tenant_id = $1 AND tracking_number = $2 RETURNING *",
            tenant_id,
            tracking_number,
        )
        return dict(row) if row else None
