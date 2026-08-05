"""Bound automatic recovery attempts across processes and restarts.

A permanently unroutable approval (agent removed, tenant at its agent limit)
remains a valid suspended run. The maintenance worker used to rediscover it
every five minutes forever, replaying the model and growing the transcript on
each pass. Process-local handoff counts cannot bound a database worker running
on many instances, and reset on every deploy.

The retry budget belongs beside the durable run. Maintenance reserves an
attempt atomically and moves next_recovery_at forward with exponential
backoff. A new user answer resets the budget; manual operator recovery is not
blocked by it.

Revision ID: 020
Revises: 019
"""

from typing import Sequence, Union

from alembic import op

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE run_transcripts
            ADD COLUMN IF NOT EXISTS recovery_attempts INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS next_recovery_at TIMESTAMPTZ;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_run_transcripts_recovery
            ON run_transcripts (next_recovery_at, updated_at)
            WHERE status IN ('running', 'suspended');
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_run_transcripts_recovery;")
    op.execute(
        """
        ALTER TABLE run_transcripts
            DROP COLUMN IF EXISTS next_recovery_at,
            DROP COLUMN IF EXISTS recovery_attempts;
        """
    )
