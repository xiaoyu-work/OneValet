"""Add tenant_extractions table for per-account raw extraction history.

Revision ID: 003
Revises: 002
Create Date: 2026-03-05
"""
from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE tenant_extractions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       TEXT NOT NULL,
            email_account   TEXT NOT NULL,
            raw_profile     JSONB NOT NULL DEFAULT '{}',
            extracted_at    TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute(
        "CREATE INDEX idx_tenant_extractions_tenant "
        "ON tenant_extractions (tenant_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tenant_extractions")
