"""
PostgreSQL pool backend for agent session persistence.

Uses the shared Database (asyncpg) pool. Sessions are stored as JSONB
with an expires_at column for TTL-based filtering.
"""

import json
import logging
from typing import List, Optional

from ..db.database import Database
from .models import AgentPoolEntry
from .pool import PoolBackend

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    tenant_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, agent_id)
)
"""

SETUP_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_tenant ON agent_sessions(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_expires ON agent_sessions(expires_at)",
]


class PostgresPoolBackend(PoolBackend):
    """
    PostgreSQL pool backend for production.

    Stores agent session entries as JSONB with TTL via expires_at column.
    Expired rows are filtered out in queries and cleaned up periodically.

    Usage:
        db = Database(dsn="postgresql://...")
        await db.initialize()
        backend = PostgresPoolBackend(db=db, session_ttl=86400)
    """

    def __init__(
        self,
        db: Database,
        session_ttl: int = 86400,
    ):
        self._db = db
        self._session_ttl = session_ttl
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create table and indexes on first use."""
        if self._initialized:
            return
        await self._db.execute(CREATE_TABLE_SQL)
        for sql in SETUP_SQL:
            await self._db.execute(sql)
        self._initialized = True
        logger.info("PostgreSQL pool backend initialized")

    async def save_agent(self, tenant_id: str, entry: AgentPoolEntry) -> None:
        await self._ensure_initialized()
        await self._db.execute(
            """
            INSERT INTO agent_sessions (tenant_id, agent_id, data, expires_at, updated_at)
            VALUES ($1, $2, $3::jsonb, NOW() + make_interval(secs => $4), NOW())
            ON CONFLICT (tenant_id, agent_id) DO UPDATE SET
                data = EXCLUDED.data,
                expires_at = NOW() + make_interval(secs => $4),
                updated_at = NOW()
            """,
            tenant_id,
            entry.agent_id,
            json.dumps(entry.to_dict()),
            float(self._session_ttl),
        )

    async def get_agent(self, tenant_id: str, agent_id: str) -> Optional[AgentPoolEntry]:
        await self._ensure_initialized()
        row = await self._db.fetchrow(
            """
            SELECT data FROM agent_sessions
            WHERE tenant_id = $1 AND agent_id = $2 AND expires_at > NOW()
            """,
            tenant_id,
            agent_id,
        )
        if row is None:
            return None
        return self._parse_entry(row["data"])

    async def list_agents(self, tenant_id: str) -> List[AgentPoolEntry]:
        await self._ensure_initialized()
        rows = await self._db.fetch(
            """
            SELECT data FROM agent_sessions
            WHERE tenant_id = $1 AND expires_at > NOW()
            """,
            tenant_id,
        )
        return [self._parse_entry(r["data"]) for r in rows]

    async def remove_agent(self, tenant_id: str, agent_id: str) -> None:
        await self._ensure_initialized()
        await self._db.execute(
            "DELETE FROM agent_sessions WHERE tenant_id = $1 AND agent_id = $2",
            tenant_id,
            agent_id,
        )

    async def clear_tenant(self, tenant_id: str) -> None:
        await self._ensure_initialized()
        await self._db.execute(
            "DELETE FROM agent_sessions WHERE tenant_id = $1",
            tenant_id,
        )

    async def get_active_tenants(self) -> List[str]:
        await self._ensure_initialized()
        rows = await self._db.fetch(
            "SELECT DISTINCT tenant_id FROM agent_sessions WHERE expires_at > NOW()"
        )
        return [r["tenant_id"] for r in rows]

    async def cleanup_expired(self) -> int:
        """Delete expired sessions. Returns number of rows removed."""
        await self._ensure_initialized()
        result = await self._db.execute(
            "DELETE FROM agent_sessions WHERE expires_at <= NOW()"
        )
        try:
            count = int(result.split()[-1])
        except (ValueError, IndexError):
            count = 0
        if count > 0:
            logger.info(f"Cleaned up {count} expired agent sessions")
        return count

    async def close(self) -> None:
        """No-op — the shared Database pool is managed externally."""
        pass

    @staticmethod
    def _parse_entry(data) -> AgentPoolEntry:
        if isinstance(data, str):
            data = json.loads(data)
        return AgentPoolEntry.from_dict(data)
