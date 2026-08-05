"""Index and parallelize durable Inbox retention.

The maintenance worker orders terminal rows by age, but neither table had an
index on the matching predicate. At scale every app instance repeatedly
sequential-scanned and sorted the whole history while deleting only 1000 rows.

Revision ID: 022
Revises: 021
"""

from typing import Sequence, Union

from alembic import op

revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # These tables are precisely the ones this migration expects to be large.
    # Building their indexes inside Alembic's transaction would block writes
    # for the duration, so use PostgreSQL's concurrent build in an explicit
    # autocommit block.
    with op.get_context().autocommit_block():
        # A failed concurrent build leaves an INVALID index with the same
        # name. IF NOT EXISTS would silently accept it on retry, so remove any
        # previous/partial object and create without that footgun.
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_run_transcripts_terminal_prune;"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_run_transcripts_running_age;"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_terminal_prune;"
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_run_transcripts_terminal_prune
                ON run_transcripts (updated_at)
                WHERE status IN ('completed', 'failed');
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_run_transcripts_running_age
                ON run_transcripts (updated_at)
                WHERE status = 'running';
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_pending_asks_terminal_prune
                ON pending_asks (resolved_at)
                WHERE (state = 'resolved' AND executed_at IS NOT NULL)
                   OR state = 'expired';
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_terminal_prune;")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_run_transcripts_running_age;")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_run_transcripts_terminal_prune;")
