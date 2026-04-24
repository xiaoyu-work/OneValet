"""Sensing foundation: raw device-sensor tables.

Adds the 5 raw sensor tables that iOS sends via koi-backend's proxy.
Lives in tenant_default (same schema as Momex) so aggregator/reflector
can read alongside long-term memory.

No pgvector here — these are plain structured rows keyed by
(user_id, ts). Episodes themselves are stored in Momex (see
koa/memory/lifecycle/episode_memory.py) rather than in their own table.

Column names match the existing sensing / calendar / todo agent code
(started_at/ended_at, starts_at/ends_at, due_at/completed_at).

Revision ID: 011
Revises: 010
"""

from typing import Sequence, Union

from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "tenant_default"


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";')
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')

    # 1. HealthKit samples
    op.execute("""
        CREATE TABLE IF NOT EXISTS health_samples (
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            ended_at TIMESTAMPTZ NULL,
            value DOUBLE PRECISION NULL,
            unit TEXT NULL,
            metadata JSONB NULL,
            source TEXT NULL,
            inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, type, started_at)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_health_samples_user_time "
        "ON health_samples(user_id, started_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_health_samples_type "
        "ON health_samples(user_id, type, started_at DESC);"
    )

    # 2. Motion segments (CoreMotion activity rollups)
    op.execute("""
        CREATE TABLE IF NOT EXISTS motion_segments (
            user_id TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            ended_at TIMESTAMPTZ NOT NULL,
            activity TEXT NOT NULL,
            confidence DOUBLE PRECISION NULL,
            metadata JSONB NULL,
            inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, started_at, activity)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_motion_segments_user_time "
        "ON motion_segments(user_id, started_at DESC);"
    )

    # 3. Device contacts (frequency/recency signals; NOT full address book)
    op.execute("""
        CREATE TABLE IF NOT EXISTS device_contacts (
            user_id TEXT NOT NULL,
            contact_hash TEXT NOT NULL,
            display_name TEXT NULL,
            channel_hints JSONB NULL,
            last_interaction_at TIMESTAMPTZ NULL,
            interaction_count INTEGER NOT NULL DEFAULT 0,
            metadata JSONB NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, contact_hash)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_device_contacts_user_recent "
        "ON device_contacts(user_id, last_interaction_at DESC);"
    )

    # 4. Local EventKit calendar events
    op.execute("""
        CREATE TABLE IF NOT EXISTS local_calendar_events (
            user_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            calendar_name TEXT NULL,
            title TEXT NULL,
            starts_at TIMESTAMPTZ NULL,
            ends_at TIMESTAMPTZ NULL,
            all_day BOOLEAN NOT NULL DEFAULT FALSE,
            location TEXT NULL,
            notes TEXT NULL,
            attendees JSONB NULL,
            metadata JSONB NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, event_id)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_local_calendar_events_user_time "
        "ON local_calendar_events(user_id, starts_at DESC);"
    )

    # 5. Local EventKit reminders
    op.execute("""
        CREATE TABLE IF NOT EXISTS local_reminders (
            user_id TEXT NOT NULL,
            reminder_id TEXT NOT NULL,
            list_name TEXT NULL,
            title TEXT NULL,
            notes TEXT NULL,
            due_at TIMESTAMPTZ NULL,
            completed BOOLEAN NOT NULL DEFAULT FALSE,
            completed_at TIMESTAMPTZ NULL,
            priority INTEGER NULL,
            metadata JSONB NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, reminder_id)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_local_reminders_user_due "
        "ON local_reminders(user_id, due_at);"
    )

    # 6. user_state — per-day wide row of derived scalars (sensing agents
    #    write here; aggregator reads).
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_state (
            user_id TEXT NOT NULL,
            local_date DATE NOT NULL,
            timezone TEXT NULL,
            sleep_minutes INTEGER NULL,
            sleep_score DOUBLE PRECISION NULL,
            hrv_ms DOUBLE PRECISION NULL,
            resting_hr DOUBLE PRECISION NULL,
            steps INTEGER NULL,
            activity_minutes INTEGER NULL,
            stress_score DOUBLE PRECISION NULL,
            mood TEXT NULL,
            primary_location TEXT NULL,
            focus_mode TEXT NULL,
            flags TEXT[] NULL,
            source_data JSONB NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, local_date)
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_state_user_date "
        "ON user_state(user_id, local_date DESC);"
    )


def downgrade() -> None:
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')
    for t in (
        "user_state",
        "local_reminders",
        "local_calendar_events",
        "device_contacts",
        "motion_segments",
        "health_samples",
    ):
        op.execute(f"DROP TABLE IF EXISTS {t};")
