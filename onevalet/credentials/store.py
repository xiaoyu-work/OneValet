"""
OneValet CredentialStore - Per-tenant credential storage with Postgres backend

Direct implementation, no abstract interface.
The framework stores and retrieves dict. Each agent/provider knows what's inside.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CredentialStore:
    """
    Per-tenant credential storage and retrieval. Postgres backend, direct implementation.
    Data is isolated by tenant_id, naturally supporting multi-tenancy.

    Table: credentials
    Primary key: (tenant_id, service, account_name)
    Columns: tenant_id, service, account_name, credentials_json, created_at, updated_at

    Usage:
        store = CredentialStore(dsn="postgresql://...")
        await store.initialize()

        await store.save("user_123", "google", {"access_token": "...", "refresh_token": "..."})
        creds = await store.get("user_123", "google")
        accounts = await store.list("user_123", service="google")
        await store.delete("user_123", "google")

        await store.close()
    """

    _CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS credentials (
        tenant_id TEXT NOT NULL,
        service TEXT NOT NULL,
        account_name TEXT NOT NULL DEFAULT 'primary',
        credentials_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (tenant_id, service, account_name)
    )
    """

    def __init__(self, dsn: str):
        """
        Args:
            dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@host:5432/db")
        """
        self._dsn = dsn
        self._pool = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool and create table if needed."""
        if self._initialized:
            return
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for CredentialStore. "
                "Install with: pip install asyncpg"
            )
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute(self._CREATE_TABLE_SQL)
        self._initialized = True
        logger.info("CredentialStore initialized")

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def save(
        self,
        tenant_id: str,
        service: str,
        credentials: dict,
        account_name: str = "primary",
    ) -> None:
        """Save credentials. Upserts on conflict."""
        await self._ensure_initialized()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO credentials (tenant_id, service, account_name, credentials_json, updated_at)
                VALUES ($1, $2, $3, $4::jsonb, NOW())
                ON CONFLICT (tenant_id, service, account_name)
                DO UPDATE SET credentials_json = $4::jsonb, updated_at = NOW()
                """,
                tenant_id, service, account_name, json.dumps(credentials),
            )

    async def get(
        self,
        tenant_id: str,
        service: str,
        account_name: str = "primary",
    ) -> Optional[dict]:
        """Retrieve credentials. Returns None if not found."""
        await self._ensure_initialized()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT credentials_json FROM credentials
                WHERE tenant_id = $1 AND service = $2 AND account_name = $3
                """,
                tenant_id, service, account_name,
            )
        if row:
            val = row["credentials_json"]
            return json.loads(val) if isinstance(val, str) else val
        return None

    async def list(
        self,
        tenant_id: str,
        service: Optional[str] = None,
    ) -> List[dict]:
        """List all connected accounts for a user, optionally filtered by service."""
        await self._ensure_initialized()
        async with self._pool.acquire() as conn:
            if service:
                rows = await conn.fetch(
                    """
                    SELECT service, account_name, credentials_json, created_at, updated_at
                    FROM credentials WHERE tenant_id = $1 AND service = $2
                    ORDER BY service, account_name
                    """,
                    tenant_id, service,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT service, account_name, credentials_json, created_at, updated_at
                    FROM credentials WHERE tenant_id = $1
                    ORDER BY service, account_name
                    """,
                    tenant_id,
                )
        results = []
        for row in rows:
            val = row["credentials_json"]
            creds = json.loads(val) if isinstance(val, str) else val
            results.append({
                "service": row["service"],
                "account_name": row["account_name"],
                "credentials": creds,
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            })
        return results

    async def delete(
        self,
        tenant_id: str,
        service: str,
        account_name: str = "primary",
    ) -> bool:
        """Delete credentials. Returns True if deleted, False if not found."""
        await self._ensure_initialized()
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM credentials
                WHERE tenant_id = $1 AND service = $2 AND account_name = $3
                """,
                tenant_id, service, account_name,
            )
        return result == "DELETE 1"

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("CredentialStore closed")
