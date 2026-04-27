"""Add subscriptions table for tracking user subscription services.

Detected from email receipts/invoices (Netflix, Spotify, iCloud, etc.)
and stored per-tenant for the AI to reference.

Revision ID: 009
Revises: 008
"""

from alembic import op

revision: str = "009"
down_revision: str = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id         TEXT NOT NULL,
            service_name      TEXT NOT NULL,
            category          TEXT DEFAULT 'other',
            amount            NUMERIC(10,2),
            currency          TEXT DEFAULT 'USD',
            billing_cycle     TEXT,
            next_billing_date DATE,
            last_charged_date DATE,
            status            TEXT DEFAULT 'active',
            detected_from     TEXT DEFAULT 'email',
            source_email      TEXT,
            is_active         BOOLEAN DEFAULT TRUE,
            created_at        TIMESTAMPTZ DEFAULT NOW(),
            updated_at        TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (tenant_id, service_name)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_subscriptions_tenant_id ON subscriptions (tenant_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions (tenant_id, status)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS subscriptions")
