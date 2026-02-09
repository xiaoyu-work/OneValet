"""
OneValet Repository - Base class for domain-specific data access.

Each agent/domain creates a subclass that defines:
- TABLE_NAME: the table it owns
- CREATE_TABLE_SQL: DDL for creating the table
- Domain-specific query methods

Adding a new agent with a new table = one new Repository subclass file.
No changes to existing code.

Usage:
    class TripRepository(Repository):
        TABLE_NAME = "trips"
        CREATE_TABLE_SQL = '''CREATE TABLE IF NOT EXISTS trips (...)'''

        async def get_user_trips(self, user_id: str) -> list[dict]:
            rows = await self.db.fetch(
                "SELECT * FROM trips WHERE user_id = $1", user_id
            )
            return [dict(r) for r in rows]
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Repository:
    """
    Base class for domain data access.

    Subclasses define TABLE_NAME, CREATE_TABLE_SQL, and domain methods.
    """

    TABLE_NAME: str = ""
    CREATE_TABLE_SQL: str = ""

    def __init__(self, db: "Database"):
        self._db = db

    @property
    def db(self) -> "Database":
        return self._db

    async def ensure_table(self) -> None:
        """Create the table if it does not exist. Call once at startup."""
        if self.CREATE_TABLE_SQL:
            await self._db.execute(self.CREATE_TABLE_SQL)
            logger.debug(f"Ensured table: {self.TABLE_NAME}")

    # -- Generic CRUD helpers (subclasses can use or ignore) --

    async def _insert(
        self,
        data: Dict[str, Any],
        returning: str = "*",
    ) -> Optional[Dict[str, Any]]:
        """Insert a row and return it."""
        columns = list(data.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(data.values())

        query = (
            f"INSERT INTO {self.TABLE_NAME} ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"RETURNING {returning}"
        )
        row = await self._db.fetchrow(query, *values)
        return dict(row) if row else None

    async def _update(
        self,
        id_column: str,
        id_value: Any,
        data: Dict[str, Any],
        returning: str = "*",
    ) -> Optional[Dict[str, Any]]:
        """Update a row by ID and return it."""
        set_clauses = []
        values = []
        for i, (col, val) in enumerate(data.items(), 1):
            set_clauses.append(f"{col} = ${i}")
            values.append(val)

        values.append(id_value)
        where_idx = len(values)

        query = (
            f"UPDATE {self.TABLE_NAME} "
            f"SET {', '.join(set_clauses)} "
            f"WHERE {id_column} = ${where_idx} "
            f"RETURNING {returning}"
        )
        row = await self._db.fetchrow(query, *values)
        return dict(row) if row else None

    async def _delete(
        self,
        id_column: str,
        id_value: Any,
    ) -> bool:
        """Delete a row by ID. Returns True if deleted."""
        result = await self._db.execute(
            f"DELETE FROM {self.TABLE_NAME} WHERE {id_column} = $1",
            id_value,
        )
        return result == "DELETE 1"

    async def _fetch_many(
        self,
        where: str = "",
        args: tuple = (),
        order_by: str = "",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch multiple rows with optional WHERE, ORDER BY, LIMIT."""
        query = f"SELECT * FROM {self.TABLE_NAME}"
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"

        rows = await self._db.fetch(query, *args)
        return [dict(r) for r in rows]
