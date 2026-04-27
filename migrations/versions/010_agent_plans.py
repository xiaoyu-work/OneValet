"""Add agent_plans table for plan state persistence.

Revision ID: 010
Revises: 009
Create Date: 2026-04-03
"""

from typing import Sequence, Union

from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_plans (
            tenant_id TEXT NOT NULL,
            plan_data JSONB NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_plans_status ON agent_plans(status)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_plans")
