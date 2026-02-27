"""
ImportantDatesRepository - Data access for the important_dates table.

Used by ImportantDateDigestAgent and the important_dates tools.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from onevalet.db import Repository

logger = logging.getLogger(__name__)


class ImportantDatesRepository(Repository):
    TABLE_NAME = "important_dates"
    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS important_dates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            date DATE NOT NULL,
            date_type TEXT DEFAULT 'custom',
            person_name TEXT,
            relationship TEXT,
            recurring BOOLEAN DEFAULT TRUE,
            remind_days_before JSONB DEFAULT '[0, 1, 7]',
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            last_reminded_year INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """

    SETUP_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_important_dates_tenant_id ON important_dates (tenant_id)",
        "CREATE INDEX IF NOT EXISTS idx_important_dates_date ON important_dates (date)",
        "CREATE INDEX IF NOT EXISTS idx_important_dates_tenant_date ON important_dates (tenant_id, date)",
        "CREATE INDEX IF NOT EXISTS idx_important_dates_is_active ON important_dates (is_active)",
        """
        CREATE OR REPLACE FUNCTION get_upcoming_important_dates(
            p_tenant_id TEXT,
            p_days_ahead INTEGER DEFAULT 7
        )
        RETURNS TABLE (
            id UUID,
            title TEXT,
            description TEXT,
            original_date DATE,
            upcoming_date DATE,
            days_until INTEGER,
            date_type TEXT,
            person_name TEXT,
            relationship TEXT,
            remind_days_before JSONB
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                d.id,
                d.title,
                d.description,
                d.date AS original_date,
                CASE
                    WHEN d.recurring THEN
                        CASE
                            WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) >= CURRENT_DATE
                            THEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 )
                            ELSE make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int + 1,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 )
                        END
                    ELSE d.date
                END AS upcoming_date,
                CASE
                    WHEN d.recurring THEN
                        CASE
                            WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) >= CURRENT_DATE
                            THEN (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                            ELSE (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int + 1,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                        END
                    ELSE (d.date - CURRENT_DATE)::int
                END AS days_until,
                d.date_type,
                d.person_name,
                d.relationship,
                d.remind_days_before
            FROM important_dates d
            WHERE d.tenant_id = p_tenant_id
              AND d.is_active = TRUE
              AND CASE
                    WHEN d.recurring THEN
                        CASE
                            WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) >= CURRENT_DATE
                            THEN (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                            ELSE (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int + 1,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                        END
                    ELSE (d.date - CURRENT_DATE)::int
                  END <= p_days_ahead
            ORDER BY days_until ASC;
        $$
        """,
    ]

    async def get_today_important_dates(
        self, tenant_id: str
    ) -> List[Dict[str, Any]]:
        """Get dates that need reminding today.

        For recurring dates, calculates the next occurrence this year
        (or next year if already passed), then checks if any value in
        remind_days_before matches the number of days until that occurrence.
        """
        query = """
            WITH date_calc AS (
                SELECT *,
                    CASE
                        WHEN recurring THEN
                            CASE
                                WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM date)::int,
                                    EXTRACT(DAY FROM date)::int
                                ) >= CURRENT_DATE
                                THEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM date)::int,
                                    EXTRACT(DAY FROM date)::int
                                )
                                ELSE make_date(
                                    (EXTRACT(YEAR FROM CURRENT_DATE) + 1)::int,
                                    EXTRACT(MONTH FROM date)::int,
                                    EXTRACT(DAY FROM date)::int
                                )
                            END
                        ELSE date
                    END AS upcoming_date
                FROM important_dates
                WHERE tenant_id = $1
            )
            SELECT *,
                (upcoming_date - CURRENT_DATE) AS days_until
            FROM date_calc
            WHERE EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(remind_days_before) AS r
                WHERE r::int = (upcoming_date - CURRENT_DATE)
            )
            ORDER BY days_until ASC
        """
        rows = await self.db.fetch(query, tenant_id)
        return [dict(r) for r in rows]

    async def get_important_dates(
        self, tenant_id: str, days_ahead: int = 60
    ) -> List[Dict[str, Any]]:
        """Get upcoming dates within N days."""
        query = """
            WITH date_calc AS (
                SELECT *,
                    CASE
                        WHEN recurring THEN
                            CASE
                                WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM date)::int,
                                    EXTRACT(DAY FROM date)::int
                                ) >= CURRENT_DATE
                                THEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM date)::int,
                                    EXTRACT(DAY FROM date)::int
                                )
                                ELSE make_date(
                                    (EXTRACT(YEAR FROM CURRENT_DATE) + 1)::int,
                                    EXTRACT(MONTH FROM date)::int,
                                    EXTRACT(DAY FROM date)::int
                                )
                            END
                        ELSE date
                    END AS upcoming_date
                FROM important_dates
                WHERE tenant_id = $1
            )
            SELECT *,
                (upcoming_date - CURRENT_DATE) AS days_until
            FROM date_calc
            WHERE (upcoming_date - CURRENT_DATE) BETWEEN 0 AND $2
            ORDER BY days_until ASC
        """
        rows = await self.db.fetch(query, tenant_id, days_ahead)
        return [dict(r) for r in rows]

    async def search_important_dates(
        self, tenant_id: str, search_term: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search dates by title or person_name (ILIKE)."""
        query = """
            SELECT * FROM important_dates
            WHERE tenant_id = $1
              AND (title ILIKE $2 OR person_name ILIKE $2)
            ORDER BY date ASC
            LIMIT $3
        """
        pattern = f"%{search_term}%"
        rows = await self.db.fetch(query, tenant_id, pattern, limit)
        return [dict(r) for r in rows]

    async def create_important_date(
        self, tenant_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Insert a new important date. Returns the created row."""
        insert_data = {"tenant_id": tenant_id, **data}
        # Serialize remind_days_before to JSONB if it's a list
        if "remind_days_before" in insert_data and isinstance(
            insert_data["remind_days_before"], list
        ):
            insert_data["remind_days_before"] = json.dumps(
                insert_data["remind_days_before"]
            )
        return await self._insert(insert_data)

    async def update_important_date(
        self, tenant_id: str, date_id: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an important date. Returns the updated row or None."""
        # Verify ownership
        row = await self.db.fetchrow(
            "SELECT id FROM important_dates WHERE id = $1 AND tenant_id = $2",
            date_id,
            tenant_id,
        )
        if not row:
            return None
        if "remind_days_before" in updates and isinstance(
            updates["remind_days_before"], list
        ):
            updates["remind_days_before"] = json.dumps(
                updates["remind_days_before"]
            )
        return await self._update("id", date_id, updates)

    async def delete_important_date(
        self, tenant_id: str, date_id: str
    ) -> bool:
        """Delete an important date. Returns True if deleted."""
        result = await self.db.execute(
            "DELETE FROM important_dates WHERE id = $1 AND tenant_id = $2",
            date_id,
            tenant_id,
        )
        return result == "DELETE 1"
