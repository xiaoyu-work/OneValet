"""Identify an ask by the action it is about, and record when it was carried out.

Two corrections to 017, both from the same mistake: an ask was identified by
the tool_call_id that happened to be in play when it was raised.

Those ids are minted fresh by the model on every run. The run that asks and
the run that acts on the answer are different processes, so the id recorded
here means nothing to the one that has to honour it -- and a replay that
re-derived the same action generated a new id, which the old index did not
recognise as a duplicate. The user could be asked the same question forever.

An action is now identified by what it *is*: this run, this tool, these
arguments. That is what survives a replay, and it is what the code looks up
by, so uniqueness and lookup finally agree.

executed_at closes the other half. An approval is a contract to carry out one
specific action, and the run that honours it may not be the only one trying:
whoever stamps this column first owns the execution, so an approved action
happens once even if two instances resume the same run together.

Revision ID: 018
Revises: 017
"""

from typing import Sequence, Union

from alembic import op

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pending_asks ADD COLUMN IF NOT EXISTS action_key TEXT;")
    op.execute("ALTER TABLE pending_asks ADD COLUMN IF NOT EXISTS executed_at TIMESTAMPTZ;")

    # Existing rows predate the column. Their tool_call_id is still unique
    # within the run, so it is a safe stand-in that cannot collide with a real
    # action key (which has a versioned "v1:" prefix).
    op.execute("UPDATE pending_asks SET action_key = tool_call_id WHERE action_key IS NULL;")
    op.execute("ALTER TABLE pending_asks ALTER COLUMN action_key SET NOT NULL;")

    # The old index keyed idempotency on an id the next run does not have.
    op.execute("DROP INDEX IF EXISTS idx_pending_asks_call;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_asks_action
            ON pending_asks (run_id, action_key);
    """)
    # Resuming asks "what did they approve that I have not done yet".
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pending_asks_unexecuted
            ON pending_asks (run_id)
            WHERE state = 'resolved' AND executed_at IS NULL;
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pending_asks_unexecuted;")
    op.execute("DROP INDEX IF EXISTS idx_pending_asks_action;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_asks_call
            ON pending_asks (run_id, tool_call_id);
    """)
    op.execute("ALTER TABLE pending_asks DROP COLUMN IF EXISTS executed_at;")
    op.execute("ALTER TABLE pending_asks DROP COLUMN IF EXISTS action_key;")
