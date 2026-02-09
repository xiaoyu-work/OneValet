"""
OneValet Database - Shared asyncpg connection pool manager.

A single Database instance is created per application and shared across
all repositories. Any Postgres-compatible database works (Supabase, RDS, local, etc.)
— just provide a DSN connection string.

Usage:
    db = Database(dsn="postgresql://user:pass@host:5432/db")
    await db.initialize()

    # Pass to repositories
    trip_repo = TripRepository(db)
    cred_store = CredentialStore(db=db)

    await db.close()
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class Database:
    """
    Manages a shared asyncpg connection pool.

    One instance per application. All Repository instances share this pool.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
    ):
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool = None
        self._initialized = False

    @property
    def pool(self):
        """Access the raw asyncpg pool. Raises if not initialized."""
        if self._pool is None:
            raise RuntimeError(
                "Database not initialized. Call await db.initialize() first."
            )
        return self._pool

    async def initialize(self) -> None:
        """Create the asyncpg connection pool."""
        if self._initialized:
            return
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for Database. "
                "Install with: pip install asyncpg"
            )
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        self._initialized = True
        logger.info("Database pool initialized")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("Database pool closed")

    def acquire(self):
        """Acquire a connection from the pool. Use as async context manager."""
        return self.pool.acquire()

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query and return status string."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[Any]:
        """Execute a query and return all rows."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[Any]:
        """Execute a query and return first row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a query and return first column of first row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)
