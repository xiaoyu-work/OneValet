"""Expire approval requests that are no longer safe to act on.

An unanswered ask previously lived forever, together with its suspended
transcript, and could be approved months after the context that justified it
had gone stale. At scale, ignored approvals also make both tables grow without
bound because open work is correctly excluded from retention pruning.

Revision ID: 021
Revises: 020
"""

from typing import Sequence, Union

from alembic import op

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pending_asks
            ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
        """
    )
    op.execute(
        """
        UPDATE pending_asks
           SET expires_at = created_at + INTERVAL '7 days'
         WHERE expires_at IS NULL;
        """
    )
    op.execute("ALTER TABLE pending_asks ALTER COLUMN expires_at SET NOT NULL;")
    op.execute(
        """
        ALTER TABLE pending_asks
            ALTER COLUMN expires_at SET DEFAULT (NOW() + INTERVAL '7 days');
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pending_asks_expiry
            ON pending_asks (expires_at)
            WHERE state = 'pending';
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pending_asks_expiry;")
    op.execute("ALTER TABLE pending_asks DROP COLUMN IF EXISTS expires_at;")
