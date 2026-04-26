"""Add iOS-friendly fields to local_calendar_events.

PR2 of the koa-as-source-of-truth refactor: koi-backend's
calendar/events routes are being rewritten as thin proxies to koa.
The Supabase-backed ``koi_events`` table that the iOS app used to
read carried three iOS-specific fields that koa's
``local_calendar_events`` doesn't have yet:

  * ``color``            — UI accent color the user picked for the event.
  * ``recurrence_rule``  — RFC 5545 RRULE string for repeating events.
  * ``reminder_minutes`` — array of "minutes before start" alarm offsets.

Adding them here so the proxy can round-trip cleanly without losing
the user's UX choices when an event flows app → koi-backend → koa.

All three default to NULL/empty — existing rows (Google-mirrored,
EventKit-ingested) don't carry these and shouldn't be forced to.

Revision ID: 013
Revises: 012
"""

from typing import Sequence, Union

from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "tenant_default"


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";')
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')

    op.execute("""
        ALTER TABLE local_calendar_events
        ADD COLUMN IF NOT EXISTS color TEXT NULL,
        ADD COLUMN IF NOT EXISTS recurrence_rule TEXT NULL,
        ADD COLUMN IF NOT EXISTS reminder_minutes INTEGER[] NULL;
    """)


def downgrade() -> None:
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')
    op.execute("""
        ALTER TABLE local_calendar_events
        DROP COLUMN IF EXISTS color,
        DROP COLUMN IF EXISTS recurrence_rule,
        DROP COLUMN IF EXISTS reminder_minutes;
    """)
