"""Add deleted_at column to cron_jobs for soft-delete support.

Instead of hard-deleting cron jobs, we now set deleted_at to the
current timestamp and enabled=false. This preserves history and
allows potential restoration of deleted reminders.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE cron_jobs
        ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cron_jobs_not_deleted
        ON cron_jobs (enabled) WHERE deleted_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_cron_jobs_not_deleted")
    op.execute("ALTER TABLE cron_jobs DROP COLUMN IF EXISTS deleted_at")
