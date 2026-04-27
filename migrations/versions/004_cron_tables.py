"""Create cron_jobs and cron_runs tables for persistent cron storage.

Replaces the file-based JSON store with PostgreSQL so data survives
container restarts and supports multi-user isolation.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS cron_jobs (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            name            TEXT NOT NULL DEFAULT '',
            enabled         BOOLEAN NOT NULL DEFAULT TRUE,
            data            JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cron_jobs_user_id ON cron_jobs (user_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cron_jobs_enabled ON cron_jobs (enabled) WHERE enabled = TRUE
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS cron_runs (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id          TEXT NOT NULL REFERENCES cron_jobs(id) ON DELETE CASCADE,
            status          TEXT,
            summary         TEXT,
            error           TEXT,
            delivered       BOOLEAN,
            duration_ms     INTEGER,
            data            JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cron_runs_job_id ON cron_runs (job_id, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS cron_runs")
    op.execute("DROP TABLE IF EXISTS cron_jobs")
