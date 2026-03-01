"""Initial schema: all tables, indexes, and stored functions.

Revision ID: 001
Revises: None
Create Date: 2026-03-01
"""
from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. credentials ──
    op.execute("""
        CREATE TABLE credentials (
            tenant_id TEXT NOT NULL,
            service TEXT NOT NULL,
            account_name TEXT NOT NULL DEFAULT 'primary',
            credentials_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (tenant_id, service, account_name)
        )
    """)
    op.execute(
        "CREATE INDEX idx_credentials_email "
        "ON credentials ((credentials_json->>'email'))"
    )

    # ── 2. oauth_states ──
    op.execute("""
        CREATE TABLE oauth_states (
            state TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            service TEXT NOT NULL,
            redirect_after TEXT,
            account_name TEXT NOT NULL DEFAULT 'primary',
            extra_data JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '10 minutes'
        )
    """)

    # ── 3. checkpoints ──
    op.execute("""
        CREATE TABLE checkpoints (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            data JSONB NOT NULL,
            parent_checkpoint_id TEXT,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_checkpoints_agent_id ON checkpoints(agent_id)")
    op.execute("CREATE INDEX idx_checkpoints_user_id ON checkpoints(user_id)")
    op.execute("CREATE INDEX idx_checkpoints_timestamp ON checkpoints(timestamp)")

    # ── 4. agent_sessions ──
    op.execute("""
        CREATE TABLE agent_sessions (
            tenant_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            data JSONB NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (tenant_id, agent_id)
        )
    """)
    op.execute("CREATE INDEX idx_agent_sessions_tenant ON agent_sessions(tenant_id)")
    op.execute("CREATE INDEX idx_agent_sessions_expires ON agent_sessions(expires_at)")

    # ── 5. expenses ──
    op.execute("""
        CREATE TABLE expenses (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       TEXT NOT NULL,
            amount          NUMERIC(12,2) NOT NULL,
            currency        TEXT DEFAULT 'USD',
            category        TEXT NOT NULL,
            description     TEXT DEFAULT '',
            merchant        TEXT DEFAULT '',
            date            DATE NOT NULL DEFAULT CURRENT_DATE,
            receipt_id      UUID,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_expenses_tenant_date ON expenses (tenant_id, date)")
    op.execute("CREATE INDEX idx_expenses_tenant_category ON expenses (tenant_id, category)")
    op.execute("CREATE INDEX idx_expenses_tenant_merchant ON expenses (tenant_id, merchant)")

    # ── 6. budgets ──
    op.execute("""
        CREATE TABLE budgets (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       TEXT NOT NULL,
            category        TEXT DEFAULT '_total',
            monthly_limit   NUMERIC(12,2) NOT NULL,
            currency        TEXT DEFAULT 'USD',
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(tenant_id, category)
        )
    """)

    # ── 7. receipts ──
    op.execute("""
        CREATE TABLE receipts (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id         TEXT NOT NULL,
            expense_id        UUID,
            file_name         TEXT NOT NULL,
            storage_provider  TEXT DEFAULT '',
            storage_file_id   TEXT,
            storage_url       TEXT,
            thumbnail_base64  TEXT,
            ocr_text          TEXT DEFAULT '',
            created_at        TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_receipts_tenant_id ON receipts (tenant_id)")
    op.execute("CREATE INDEX idx_receipts_expense_id ON receipts (expense_id)")

    # ── 8. trips ──
    op.execute("""
        CREATE TABLE trips (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       TEXT NOT NULL,
            title           TEXT,
            trip_type       TEXT,
            carrier         TEXT,
            carrier_code    TEXT,
            trip_number     TEXT,
            booking_reference TEXT,
            origin_city     TEXT,
            origin_code     TEXT,
            destination_city TEXT,
            destination_code TEXT,
            departure_time  TIMESTAMPTZ,
            departure_local_time TEXT,
            departure_terminal TEXT,
            departure_gate  TEXT,
            arrival_time    TIMESTAMPTZ,
            arrival_local_time TEXT,
            arrival_terminal TEXT,
            arrival_gate    TEXT,
            hotel_name      TEXT,
            hotel_address   TEXT,
            check_in_date   DATE,
            check_out_date  DATE,
            rental_company  TEXT,
            pickup_time     TIMESTAMPTZ,
            dropoff_time    TIMESTAMPTZ,
            status          TEXT DEFAULT 'upcoming',
            source          TEXT,
            source_id       TEXT,
            source_account  TEXT,
            raw_data        JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_trips_tenant_id ON trips (tenant_id)")
    op.execute("CREATE INDEX idx_trips_status ON trips (status)")

    # ── 9. shipments ──
    op.execute("""
        CREATE TABLE shipments (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       TEXT NOT NULL,
            tracking_number TEXT NOT NULL,
            carrier         TEXT,
            tracking_url    TEXT,
            status          TEXT DEFAULT 'unknown',
            description     TEXT,
            last_update     TEXT,
            estimated_delivery TEXT,
            tracking_history JSONB DEFAULT '[]',
            delivered_notified BOOLEAN,
            is_active       BOOLEAN DEFAULT TRUE,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (tenant_id, tracking_number)
        )
    """)
    op.execute("CREATE INDEX idx_shipments_tenant_id ON shipments (tenant_id)")

    # ── 10. important_dates ──
    op.execute("""
        CREATE TABLE important_dates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            date DATE NOT NULL,
            date_type TEXT DEFAULT 'custom',
            person_name TEXT,
            relationship TEXT,
            recurring BOOLEAN DEFAULT TRUE,
            remind_days_before JSONB DEFAULT '[0, 1, 7]',
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            last_reminded_year INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX idx_important_dates_tenant_id ON important_dates (tenant_id)")
    op.execute("CREATE INDEX idx_important_dates_date ON important_dates (date)")
    op.execute("CREATE INDEX idx_important_dates_tenant_date ON important_dates (tenant_id, date)")
    op.execute("CREATE INDEX idx_important_dates_is_active ON important_dates (is_active)")

    # ── Stored function: get_upcoming_important_dates ──
    op.execute("""
        CREATE OR REPLACE FUNCTION get_upcoming_important_dates(
            p_tenant_id TEXT,
            p_days_ahead INTEGER DEFAULT 7
        )
        RETURNS TABLE (
            id UUID,
            title TEXT,
            description TEXT,
            original_date DATE,
            upcoming_date DATE,
            days_until INTEGER,
            date_type TEXT,
            person_name TEXT,
            relationship TEXT,
            remind_days_before JSONB
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                d.id,
                d.title,
                d.description,
                d.date AS original_date,
                CASE
                    WHEN d.recurring THEN
                        CASE
                            WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) >= CURRENT_DATE
                            THEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 )
                            ELSE make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int + 1,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 )
                        END
                    ELSE d.date
                END AS upcoming_date,
                CASE
                    WHEN d.recurring THEN
                        CASE
                            WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) >= CURRENT_DATE
                            THEN (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                            ELSE (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int + 1,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                        END
                    ELSE (d.date - CURRENT_DATE)::int
                END AS days_until,
                d.date_type,
                d.person_name,
                d.relationship,
                d.remind_days_before
            FROM important_dates d
            WHERE d.tenant_id = p_tenant_id
              AND d.is_active = TRUE
              AND CASE
                    WHEN d.recurring THEN
                        CASE
                            WHEN make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) >= CURRENT_DATE
                            THEN (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                            ELSE (make_date(
                                    EXTRACT(YEAR FROM CURRENT_DATE)::int + 1,
                                    EXTRACT(MONTH FROM d.date)::int,
                                    EXTRACT(DAY FROM d.date)::int
                                 ) - CURRENT_DATE)::int
                        END
                    ELSE (d.date - CURRENT_DATE)::int
                  END <= p_days_ahead
            ORDER BY days_until ASC;
        $$
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS get_upcoming_important_dates(TEXT, INTEGER)")
    op.execute("DROP TABLE IF EXISTS important_dates")
    op.execute("DROP TABLE IF EXISTS shipments")
    op.execute("DROP TABLE IF EXISTS trips")
    op.execute("DROP TABLE IF EXISTS receipts")
    op.execute("DROP TABLE IF EXISTS budgets")
    op.execute("DROP TABLE IF EXISTS expenses")
    op.execute("DROP TABLE IF EXISTS agent_sessions")
    op.execute("DROP TABLE IF EXISTS checkpoints")
    op.execute("DROP TABLE IF EXISTS oauth_states")
    op.execute("DROP TABLE IF EXISTS credentials")
