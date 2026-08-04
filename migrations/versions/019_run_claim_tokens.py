"""Make a run lease belong to the process that claimed it.

A stale run can be taken over while the old process is still alive but
paused. Status='running' alone does not distinguish those processes: when the
old one wakes, its touch keeps the new lease alive, its save overwrites the
new transcript, and its release can suspend the run the new process just
claimed.

The claim token is the fencing token. Every write made by a resumed run must
carry the token it received from claim(), so a process whose lease was
replaced can no longer change the row. Its next heartbeat sees that it lost
ownership and stops before another model or tool call.

Revision ID: 019
Revises: 018
"""

from typing import Sequence, Union

from alembic import op

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE run_transcripts ADD COLUMN IF NOT EXISTS claim_token TEXT;")


def downgrade() -> None:
    op.execute("ALTER TABLE run_transcripts DROP COLUMN IF EXISTS claim_token;")
