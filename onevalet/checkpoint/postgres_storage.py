"""
PostgreSQL checkpoint storage backend.

Uses asyncpg via the shared Database pool for production-grade
checkpoint persistence with JSONB storage and indexed queries.
"""

import json
import logging
from typing import List, Optional

from ..db.database import Database
from .models import Checkpoint, CheckpointMetadata, CheckpointTree
from .storage import CheckpointStorage

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    data JSONB NOT NULL,
    parent_checkpoint_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
)
"""

SETUP_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_id ON checkpoints(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_user_id ON checkpoints(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp ON checkpoints(timestamp)",
]


class PostgreSQLStorage(CheckpointStorage):
    """
    PostgreSQL checkpoint storage for production.

    Stores full checkpoint data as JSONB for flexible querying.
    Uses indexed columns for fast lookups by agent_id, user_id, and timestamp.

    Usage with shared Database pool (recommended):
        db = Database(dsn="postgresql://...")
        await db.initialize()
        storage = PostgreSQLStorage(db=db)
        await storage.initialize()

    Usage standalone:
        storage = PostgreSQLStorage(dsn="postgresql://...")
        await storage.initialize()
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        dsn: Optional[str] = None,
    ):
        if db is None and dsn is None:
            raise ValueError("Either db or dsn must be provided")
        self._db = db
        self._dsn = dsn
        self._owns_db = db is None
        self._initialized = False

    async def initialize(self) -> None:
        """Create table and indexes. Must be called before use."""
        if self._initialized:
            return

        if self._db is None:
            self._db = Database(dsn=self._dsn)
            await self._db.initialize()

        await self._db.execute(CREATE_TABLE_SQL)
        for sql in SETUP_SQL:
            await self._db.execute(sql)

        self._initialized = True
        logger.info("PostgreSQL checkpoint storage initialized")

    async def close(self) -> None:
        """Close database connection if we own it."""
        if self._owns_db and self._db is not None:
            await self._db.close()

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "PostgreSQLStorage not initialized. Call await storage.initialize() first."
            )

    # -- CheckpointStorage interface ------------------------------------------

    async def save(self, checkpoint: Checkpoint) -> str:
        self._ensure_initialized()
        await self._db.execute(
            """
            INSERT INTO checkpoints (id, agent_id, agent_type, user_id, status, data, parent_checkpoint_id, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                data = EXCLUDED.data,
                timestamp = EXCLUDED.timestamp
            """,
            checkpoint.id,
            checkpoint.agent_id,
            checkpoint.agent_type,
            checkpoint.user_id,
            checkpoint.status,
            checkpoint.to_json(),
            checkpoint.parent_checkpoint_id,
            checkpoint.timestamp,
        )
        return checkpoint.id

    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        self._ensure_initialized()
        row = await self._db.fetchrow(
            "SELECT data FROM checkpoints WHERE id = $1",
            checkpoint_id,
        )
        if row is None:
            return None
        return self._parse_checkpoint(row["data"])

    async def delete(self, checkpoint_id: str) -> bool:
        self._ensure_initialized()
        result = await self._db.execute(
            "DELETE FROM checkpoints WHERE id = $1",
            checkpoint_id,
        )
        # asyncpg returns "DELETE N"
        return result == "DELETE 1"

    async def list_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CheckpointMetadata]:
        self._ensure_initialized()
        rows = await self._db.fetch(
            """
            SELECT data FROM checkpoints
            WHERE agent_id = $1
            ORDER BY timestamp DESC
            LIMIT $2 OFFSET $3
            """,
            agent_id,
            limit,
            offset,
        )
        return [
            CheckpointMetadata.from_checkpoint(self._parse_checkpoint(r["data"]))
            for r in rows
        ]

    async def list_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CheckpointMetadata]:
        self._ensure_initialized()
        rows = await self._db.fetch(
            """
            SELECT data FROM checkpoints
            WHERE user_id = $1
            ORDER BY timestamp DESC
            LIMIT $2 OFFSET $3
            """,
            user_id,
            limit,
            offset,
        )
        return [
            CheckpointMetadata.from_checkpoint(self._parse_checkpoint(r["data"]))
            for r in rows
        ]

    async def get_tree(self, agent_id: str) -> Optional[CheckpointTree]:
        self._ensure_initialized()
        rows = await self._db.fetch(
            """
            SELECT data FROM checkpoints
            WHERE agent_id = $1
            ORDER BY timestamp ASC
            """,
            agent_id,
        )
        if not rows:
            return None

        checkpoints = [self._parse_checkpoint(r["data"]) for r in rows]
        tree = CheckpointTree(root_id=checkpoints[0].id)
        for checkpoint in checkpoints:
            tree.add_checkpoint(checkpoint)
        return tree

    async def get_latest(self, agent_id: str) -> Optional[Checkpoint]:
        self._ensure_initialized()
        row = await self._db.fetchrow(
            """
            SELECT data FROM checkpoints
            WHERE agent_id = $1
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            agent_id,
        )
        if row is None:
            return None
        return self._parse_checkpoint(row["data"])

    async def clear_agent(self, agent_id: str) -> int:
        self._ensure_initialized()
        result = await self._db.execute(
            "DELETE FROM checkpoints WHERE agent_id = $1",
            agent_id,
        )
        return self._parse_delete_count(result)

    async def clear_user(self, user_id: str) -> int:
        self._ensure_initialized()
        result = await self._db.execute(
            "DELETE FROM checkpoints WHERE user_id = $1",
            user_id,
        )
        return self._parse_delete_count(result)

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _parse_checkpoint(data) -> Checkpoint:
        """Parse JSONB data (returned as dict or str) into a Checkpoint."""
        if isinstance(data, str):
            return Checkpoint.from_json(data)
        return Checkpoint.from_dict(data)

    @staticmethod
    def _parse_delete_count(result: str) -> int:
        """Extract row count from asyncpg 'DELETE N' status string."""
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0
