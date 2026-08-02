"""The human-attention queue.

When an agent needs a decision only a person can make, it records the request
here and the run stops. The person answers whenever they get to it -- from the
app, a text message, or anywhere else the item was mirrored -- and the run
picks up where it left off.

Two properties matter at scale, and both are enforced by the database rather
than by process-local locks:

- An item is unique per (run_id, tool_call_id), so a retried or replayed run
  asks once rather than filling someone's phone with duplicates.
- Resolution is a compare-and-swap on state, so when the same question is
  answered from two devices at once exactly one write wins.

Revision ID: 017
Revises: 016
"""

from typing import Sequence, Union

from alembic import op

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS pending_asks (
            id           TEXT PRIMARY KEY,
            tenant_id    TEXT NOT NULL,
            run_id       TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            kind         TEXT NOT NULL,
            title        TEXT NOT NULL DEFAULT '',
            body         TEXT NOT NULL DEFAULT '',
            options      JSONB NOT NULL DEFAULT '[]'::jsonb,
            state        TEXT NOT NULL DEFAULT 'pending',
            resolution   TEXT,
            resolved_by  TEXT,
            data         JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            resolved_at  TIMESTAMPTZ
        );
    """)
    # One ask per tool call: a replayed run must not re-ask what is already
    # pending or answered.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_asks_call
            ON pending_asks (run_id, tool_call_id);
    """)
    # "What is waiting for me?" -- the query every surface runs.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pending_asks_open
            ON pending_asks (tenant_id, created_at DESC)
            WHERE state = 'pending';
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pending_asks;")
