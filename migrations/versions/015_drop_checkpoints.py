"""Drop the unused checkpoints table.

The checkpoint system it backed has been removed. It was built for the
InputField state machine, where a checkpoint captured half-collected fields
so a paused agent could resume filling them in. That state machine is gone,
and with it the two fields the snapshot actually carried -- collected_fields
and execution_state are now permanently empty -- so the table could only ever
have stored empty dicts. Nothing ever wrote to it in production.

Crash recovery, if it is wanted later, needs transcript replay rather than
per-agent state snapshots, and that is a different schema.

Revision ID: 015
Revises: 014
"""

from typing import Sequence, Union

from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS checkpoints;")


def downgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            data JSONB NOT NULL,
            parent_checkpoint_id TEXT,
            timestamp TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_id ON checkpoints(agent_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_user_id ON checkpoints(user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp ON checkpoints(timestamp);")
