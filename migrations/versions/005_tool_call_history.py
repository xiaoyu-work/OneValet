"""Add tool_call_history table for persistent action tracking.

Revision ID: 005
Revises: 004
"""

from typing import Sequence, Union

from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS tool_call_history (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            agent_name TEXT,
            summary TEXT,
            args_summary JSONB DEFAULT '{}',
            success BOOLEAN DEFAULT TRUE,
            result_status TEXT,
            result_chars INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_call_history_tenant_created
        ON tool_call_history (tenant_id, created_at DESC);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_tool_call_history_tenant_created;")
    op.execute("DROP TABLE IF EXISTS tool_call_history;")
