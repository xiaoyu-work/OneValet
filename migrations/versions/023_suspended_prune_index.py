"""Add the suspended-retention index for databases that already ran 022.

Migration 022 gained this index while the feature was still being hardened.
An environment that applied an earlier 022 is already stamped at that
revision and will not re-run the edited file, so add the missing index in a
new revision as well. A database created from the latest chain already has
the index and skips it.

Revision ID: 023
Revises: 022
"""

from typing import Sequence, Union

from alembic import op

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_run_transcripts_suspended_age
                ON run_transcripts (updated_at)
                WHERE status = 'suspended';
            """
        )


def downgrade() -> None:
    # 022 in the latest chain also owns this index; leave it in place when
    # stepping back to 022 and let 022's own downgrade remove it.
    pass
