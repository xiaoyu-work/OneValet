"""
OneValet CredentialStore - Per-tenant credential storage with Postgres backend.

Extends Repository for shared connection pool. Backward compatible with
standalone DSN construction.

Usage (shared pool — recommended):
    db = Database(dsn="postgresql://...")
    await db.initialize()
    store = CredentialStore(db=db)

Usage (standalone — backward compatible):
    store = CredentialStore(dsn="postgresql://...")
    await store.initialize()

    await store.save("user_123", "google", {"access_token": "...", "refresh_token": "..."})
    creds = await store.get("user_123", "google")
    accounts = await store.list("user_123", service="google")
    await store.delete("user_123", "google")

    await store.close()
"""

import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..db import Database, Repository

logger = logging.getLogger(__name__)


class CredentialStore(Repository):
    """
    Per-tenant credential storage and retrieval.

    Table: credentials
    Primary key: (tenant_id, service, account_name)
    """

    TABLE_NAME = "credentials"
    CREATE_TABLE_SQL = """
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
    SETUP_SQL = [
        # Email lookup index for webhook handlers
        "CREATE INDEX IF NOT EXISTS idx_credentials_email ON credentials ((credentials_json->>'email'))",
        # OAuth state tokens with automatic expiration
        """
        CREATE TABLE IF NOT EXISTS oauth_states (
            state TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            service TEXT NOT NULL,
            redirect_after TEXT,
            account_name TEXT NOT NULL DEFAULT 'primary',
            extra_data JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '10 minutes'
        )
        """,
    ]

    def __init__(self, db: Database = None, dsn: str = None):
        """
        Two construction modes:

        1. Shared pool: CredentialStore(db=database_instance)
        2. Standalone:  CredentialStore(dsn="postgresql://...")
        """
        if db:
            super().__init__(db)
            self._standalone = False
        elif dsn:
            self._standalone_db = Database(dsn=dsn, min_size=1, max_size=5)
            super().__init__(self._standalone_db)
            self._standalone = True
        else:
            raise ValueError("Either db or dsn must be provided")

    async def initialize(self) -> None:
        """Initialize pool and ensure table. For standalone mode or first-time setup."""
        if self._standalone:
            await self._db.initialize()
        await self.ensure_table()
        logger.info("CredentialStore initialized")

    async def close(self) -> None:
        """Close pool. Only closes if standalone (owns its own pool)."""
        if self._standalone:
            await self._db.close()
        logger.info("CredentialStore closed")

    async def save(
        self,
        tenant_id: str,
        service: str,
        credentials: dict,
        account_name: str = "primary",
    ) -> None:
        """Save credentials. Upserts on conflict."""
        await self.db.execute(
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
        row = await self.db.fetchrow(
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
        """List all connected accounts, optionally filtered by service."""
        if service:
            rows = await self.db.fetch(
                """
                SELECT service, account_name, credentials_json, created_at, updated_at
                FROM credentials WHERE tenant_id = $1 AND service = $2
                ORDER BY service, account_name
                """,
                tenant_id, service,
            )
        else:
            rows = await self.db.fetch(
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
        result = await self.db.execute(
            """
            DELETE FROM credentials
            WHERE tenant_id = $1 AND service = $2 AND account_name = $3
            """,
            tenant_id, service, account_name,
        )
        return result == "DELETE 1"

    async def find_by_email(
        self,
        email: str,
        service: Optional[str] = None,
    ) -> Optional[dict]:
        """Find credentials by email across all tenants.

        Used by webhook handlers to map email → tenant_id + credentials.
        """
        if service:
            row = await self.db.fetchrow(
                """
                SELECT tenant_id, service, account_name, credentials_json
                FROM credentials
                WHERE credentials_json->>'email' = $1 AND service = $2
                LIMIT 1
                """,
                email, service,
            )
        else:
            row = await self.db.fetchrow(
                """
                SELECT tenant_id, service, account_name, credentials_json
                FROM credentials
                WHERE credentials_json->>'email' = $1
                LIMIT 1
                """,
                email,
            )
        if not row:
            return None
        val = row["credentials_json"]
        creds = json.loads(val) if isinstance(val, str) else val
        return {
            "tenant_id": row["tenant_id"],
            "service": row["service"],
            "account_name": row["account_name"],
            "credentials": creds,
        }

    # ─── OAuth State Management ───

    async def save_oauth_state(
        self,
        tenant_id: str,
        service: str,
        redirect_after: Optional[str] = None,
        account_name: str = "primary",
        extra_data: Optional[dict] = None,
    ) -> str:
        """Generate and persist an OAuth state token. Returns the token."""
        state = secrets.token_urlsafe(32)
        await self.db.execute(
            """
            INSERT INTO oauth_states (state, tenant_id, service, redirect_after, account_name, extra_data)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            state, tenant_id, service, redirect_after, account_name,
            json.dumps(extra_data) if extra_data else None,
        )
        # Garbage-collect expired states
        await self.db.execute(
            "DELETE FROM oauth_states WHERE expires_at < NOW()"
        )
        return state

    async def consume_oauth_state(self, state: str) -> Optional[dict]:
        """Validate and consume (delete) an OAuth state token.

        Returns {tenant_id, service, redirect_after, account_name, extra_data}
        or None if invalid/expired.
        """
        row = await self.db.fetchrow(
            """
            DELETE FROM oauth_states
            WHERE state = $1 AND expires_at > NOW()
            RETURNING tenant_id, service, redirect_after, account_name, extra_data
            """,
            state,
        )
        if not row:
            return None
        extra = row["extra_data"]
        if isinstance(extra, str):
            extra = json.loads(extra)
        return {
            "tenant_id": row["tenant_id"],
            "service": row["service"],
            "redirect_after": row["redirect_after"],
            "account_name": row["account_name"],
            "extra_data": extra,
        }
