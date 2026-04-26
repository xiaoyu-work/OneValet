"""Routing preferences + source semantics for local_calendar_events.

PR1 of the koa-as-source-of-truth refactor:

1. Adds ``user_routing_preferences`` table so the calendar/todo agent's
   ``resolve_surface_target`` can persist the per-tenant default provider
   (local vs google etc.) directly in koa's own DB. Previously the resolver
   HTTP-called koi-backend for an endpoint that never existed; every call
   404'd and the resolver always fell back to ``provider="local"``.

2. Adds a ``source`` column to ``local_calendar_events`` so we can tell
   apart EventKit-ingested rows, Google-mirror rows, and AI-created
   "true local" rows. The events CRUD route added in this PR refuses
   to PATCH/DELETE rows whose source is not ``local`` — without this,
   an agent could "successfully" delete a Google-mirrored event from
   the cache while the real Google calendar still has it (the next
   sync would reincarnate it).

   Existing rows default to ``eventkit`` since migration 011 was created
   to ingest iOS EventKit data. CalendarSyncService is updated in the
   same PR to set source='google' explicitly.

Revision ID: 012
Revises: 011
"""

from typing import Sequence, Union

from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "tenant_default"


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";')
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')

    # 1. Source column on local_calendar_events
    op.execute("""
        ALTER TABLE local_calendar_events
        ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'eventkit';
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_local_calendar_events_user_source "
        "ON local_calendar_events(user_id, source);"
    )

    # 2. user_routing_preferences
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_routing_preferences (
            tenant_id TEXT NOT NULL,
            surface TEXT NOT NULL,
            default_provider TEXT NOT NULL,
            default_account TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, surface)
        );
    """)


def downgrade() -> None:
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')
    op.execute("DROP TABLE IF EXISTS user_routing_preferences;")
    op.execute("DROP INDEX IF EXISTS idx_local_calendar_events_user_source;")
    op.execute("ALTER TABLE local_calendar_events DROP COLUMN IF EXISTS source;")
