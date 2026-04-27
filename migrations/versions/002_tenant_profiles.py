"""Add tenant_profiles table for storing extracted user profiles.

Revision ID: 002
Revises: 001
Create Date: 2026-03-05
"""

from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE tenant_profiles (
            tenant_id    TEXT PRIMARY KEY,
            profile      JSONB NOT NULL DEFAULT '{}',
            extracted_at TIMESTAMPTZ,
            created_at   TIMESTAMPTZ DEFAULT NOW(),
            updated_at   TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tenant_profiles")
