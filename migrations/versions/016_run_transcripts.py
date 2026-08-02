"""Persist ReAct transcripts so a run can be rebuilt.

A run's message list lives only in memory today: if the process restarts, or
the agent needs a human decision it cannot get right now, everything the run
had gathered is lost and the user must start over.

Storing the transcript makes a run resumable. The rows are keyed by the
request id the audit log already threads through every request, and hold the
message list verbatim -- the same shape the model is sent -- so resuming means
replaying it rather than reconstructing intent.

Revision ID: 016
Revises: 015
"""

from typing import Sequence, Union

from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS run_transcripts (
            run_id       TEXT PRIMARY KEY,
            tenant_id    TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'running',
            messages     JSONB NOT NULL,
            user_message TEXT NOT NULL DEFAULT '',
            metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
            turn         INTEGER NOT NULL DEFAULT 0,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    # Finding a tenant's resumable runs is the hot query; completed runs are
    # only ever read by id, so keep the index narrow.
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_transcripts_resumable
            ON run_transcripts (tenant_id, updated_at DESC)
            WHERE status IN ('running', 'suspended');
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS run_transcripts;")
