"""Persist a multi-intent DAG while a sub-task waits for approval.

A waiting child used to block its dependents correctly, but the parent DAG
then disappeared. Resuming the child executed the approved action and never
returned to the skipped dependency levels.

Revision ID: 025
Revises: 024
"""

from typing import Sequence, Union

from alembic import op

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS dag_continuations (
            parent_run_id TEXT PRIMARY KEY,
            tenant_id     TEXT NOT NULL,
            intent        JSONB NOT NULL,
            metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
            results       JSONB NOT NULL DEFAULT '{}'::jsonb,
            status        TEXT NOT NULL DEFAULT 'running',
            claim_token   TEXT,
            recovery_attempts INTEGER NOT NULL DEFAULT 0,
            next_recovery_at TIMESTAMPTZ,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS dag_waiting_subruns (
            sub_run_id    TEXT PRIMARY KEY,
            parent_run_id TEXT NOT NULL REFERENCES dag_continuations(parent_run_id)
                ON DELETE CASCADE,
            sub_task_id   INTEGER NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dag_waiting_parent
            ON dag_waiting_subruns (parent_run_id);
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dag_continuations_waiting
            ON dag_continuations (next_recovery_at, updated_at)
            WHERE status IN ('waiting', 'running');
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dag_waiting_subruns;")
    op.execute("DROP TABLE IF EXISTS dag_continuations;")
