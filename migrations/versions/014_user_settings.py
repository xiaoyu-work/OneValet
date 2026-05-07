"""Two-way calendar sync settings.

Adds ``tenant_default.user_settings`` — a key/value bag for per-user
preferences that need to be readable both from the AI engine and the
HTTP gateway. Initial use case is the ``two_way_calendar_sync`` boolean
that gates reverse edits on Google / EventKit calendar rows from the
iOS app.

Revision ID: 014
Revises: 013
"""

from typing import Sequence, Union

from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "tenant_default"


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";')
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')

    op.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id    TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, key)
        );
    """)


def downgrade() -> None:
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')
    op.execute("DROP TABLE IF EXISTS user_settings;")
