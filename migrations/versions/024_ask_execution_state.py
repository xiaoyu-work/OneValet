"""Separate execution ownership from execution completion.

executed_at used to be stamped before the tool was called. A process crash
between that write and the first line of the executor permanently consumed
the approval while doing nothing, and every recovery path reported the action
already done.

Execution now has a durable state machine:

    pending -> claimed -> started -> completed

A claim abandoned before `started` is safe to return to `pending`. A process
lost after `started` is ambiguous -- the provider may have acted -- so the
ask is reopened as a new explicit "retry?" decision rather than being
automatically executed twice.

Revision ID: 024
Revises: 023
"""

from typing import Sequence, Union

from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pending_asks
            ADD COLUMN IF NOT EXISTS execution_state TEXT NOT NULL DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS execution_claim_token TEXT,
            ADD COLUMN IF NOT EXISTS execution_claimed_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS execution_started_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS execution_finished_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS execution_outcome TEXT;
        """
    )
    op.execute(
        """
        UPDATE pending_asks
           SET execution_state = CASE
                   -- Before 024, executed_at was stamped before the executor
                   -- started. It proves an attempt was claimed, not that the
                   -- action completed, so migrate it as ambiguous.
                   WHEN executed_at IS NULL THEN 'pending'
                   WHEN LOWER(COALESCE(resolution, '')) IN
                        ('approve', 'approved', 'yes', 'y', 'ok', 'okay',
                         'confirm', 'allow', 'accept')
                       THEN 'started'
                   ELSE 'completed'
               END,
               execution_started_at = CASE
                   WHEN executed_at IS NOT NULL
                    AND LOWER(COALESCE(resolution, '')) IN
                        ('approve', 'approved', 'yes', 'y', 'ok', 'okay',
                         'confirm', 'allow', 'accept')
                       THEN executed_at
                   ELSE NULL
               END,
               execution_finished_at = CASE
                   WHEN executed_at IS NOT NULL
                    AND LOWER(COALESCE(resolution, '')) NOT IN
                        ('approve', 'approved', 'yes', 'y', 'ok', 'okay',
                         'confirm', 'allow', 'accept')
                       THEN executed_at
                   ELSE NULL
               END
         WHERE state = 'resolved';
        """
    )

    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_unexecuted;")
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_pending_asks_unexecuted
                ON pending_asks (run_id)
                WHERE state = 'resolved'
                  AND execution_state IN ('pending', 'claimed', 'started');
            """
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_terminal_prune;"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_recent_outcomes;"
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_pending_asks_terminal_prune
                ON pending_asks (resolved_at)
                WHERE (state = 'resolved' AND execution_state = 'completed')
                   OR state = 'expired';
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_pending_asks_recent_outcomes
                ON pending_asks (tenant_id, execution_finished_at DESC)
                WHERE state = 'resolved' AND execution_state = 'completed';
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_recent_outcomes;"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_unexecuted;")
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_pending_asks_unexecuted
                ON pending_asks (run_id)
                WHERE state = 'resolved' AND executed_at IS NULL;
            """
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_pending_asks_terminal_prune;"
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY idx_pending_asks_terminal_prune
                ON pending_asks (resolved_at)
                WHERE (state = 'resolved' AND executed_at IS NOT NULL)
                   OR state = 'expired';
            """
        )
    op.execute(
        """
        ALTER TABLE pending_asks
            DROP COLUMN IF EXISTS execution_outcome,
            DROP COLUMN IF EXISTS execution_finished_at,
            DROP COLUMN IF EXISTS execution_started_at,
            DROP COLUMN IF EXISTS execution_claimed_at,
            DROP COLUMN IF EXISTS execution_claim_token,
            DROP COLUMN IF EXISTS execution_state;
        """
    )
