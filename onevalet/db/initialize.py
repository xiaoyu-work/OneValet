"""
OneValet Database Schema Management.

Lightweight migration system:
- Tracks current schema version in a `schema_version` table
- Only runs migrations that haven't been applied yet
- Each migration is a (version, description, SQL) tuple
- Safe for concurrent startup (uses advisory lock)

Usage:
    db = Database(dsn="postgresql://...")
    await db.initialize()
    await ensure_schema(db)
"""

import logging
from typing import List, Tuple

from .database import Database

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Migration registry
#
# Append-only. Never modify or delete existing entries.
# Each entry: (version, description, sql)
# ──────────────────────────────────────────────────────────────
MIGRATIONS: List[Tuple[int, str, str]] = [
    (
        1,
        "Create credentials table",
        """
        CREATE TABLE IF NOT EXISTS credentials (
            tenant_id TEXT NOT NULL,
            service TEXT NOT NULL,
            account_name TEXT NOT NULL DEFAULT 'primary',
            credentials_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, service, account_name)
        );
        """,
    ),
    (
        2,
        "Create trips table",
        """
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
        """,
    ),
    (
        3,
        "Create shipments table",
        """
        CREATE TABLE IF NOT EXISTS shipments (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         TEXT NOT NULL,
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
            UNIQUE (user_id, tracking_number)
        );
        """,
    ),
    (
        4,
        "Create tasks table",
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            name TEXT,
            description TEXT,
            status TEXT DEFAULT 'active',
            trigger_type TEXT,
            trigger_config JSONB,
            action_type TEXT,
            action_config JSONB,
            run_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
    ),
    (
        5,
        "Create important_dates table",
        """
        CREATE TABLE IF NOT EXISTS important_dates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            date DATE NOT NULL,
            date_type TEXT DEFAULT 'custom',
            person_name TEXT,
            relationship TEXT,
            recurring BOOLEAN DEFAULT TRUE,
            remind_days_before JSONB DEFAULT '[0, 1, 7]',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
    ),
    # ── Future migrations go here ──
    # (6, "Add index on trips.user_id", "CREATE INDEX IF NOT EXISTS ..."),
    # (7, "Add column trips.notes",     "ALTER TABLE trips ADD COLUMN IF NOT EXISTS notes TEXT"),
]


# ──────────────────────────────────────────────────────────────
# Schema management
# ──────────────────────────────────────────────────────────────

_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# PostgreSQL advisory lock ID (arbitrary constant, unique to this app)
_LOCK_ID = 7_2658_1001


async def ensure_schema(db: Database) -> None:
    """Apply any pending migrations.

    - Creates the ``schema_version`` tracking table if needed
    - Uses a PostgreSQL advisory lock to prevent concurrent migration runs
    - Skips migrations that have already been applied
    - Each migration runs in its own transaction

    Args:
        db: Initialized Database instance.
    """
    async with db.pool.acquire() as conn:
        # Advisory lock — only one process migrates at a time
        await conn.execute("SELECT pg_advisory_lock($1)", _LOCK_ID)
        try:
            # Bootstrap the version table
            await conn.execute(_BOOTSTRAP_SQL)

            # Get current version
            row = await conn.fetchrow(
                "SELECT COALESCE(MAX(version), 0) AS v FROM schema_version"
            )
            current = row["v"]

            pending = [(v, d, s) for v, d, s in MIGRATIONS if v > current]
            if not pending:
                logger.debug(f"Schema up to date (version {current})")
                return

            # Apply each migration in order
            for version, description, sql in pending:
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute(
                        "INSERT INTO schema_version (version, description) VALUES ($1, $2)",
                        version,
                        description,
                    )
                logger.info(f"Migration {version}: {description}")

            logger.info(
                f"Schema migrated {current} -> {pending[-1][0]} "
                f"({len(pending)} migration(s))"
            )
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_ID)
