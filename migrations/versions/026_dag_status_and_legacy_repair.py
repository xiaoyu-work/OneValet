"""Repair databases that already applied earlier revisions 024 and 025.

Revision 025 originally lacked the active/waiting child status. Revision 024
also originally reopened every affirmative legacy row, including ones whose
transcript was terminal or already pruned. Editing old migration files does
not update stamped databases, so make both repairs in a new revision.

Revision ID: 026
Revises: 025
"""

from typing import Sequence, Union

from alembic import op

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE dag_waiting_subruns
            ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'waiting';
        """
    )

    op.execute(
        """
        UPDATE pending_asks AS ask
           SET state = 'resolved',
               resolution = COALESCE(ask.resolution, 'legacy_unknown'),
               execution_state = 'completed',
               execution_finished_at = COALESCE(
                   ask.execution_finished_at,
                   ask.execution_started_at,
                   ask.executed_at,
                   NOW()
               ),
               execution_outcome =
                   'Legacy approved attempt has an unknown outcome; '
                   || 'its run is no longer resumable.'
         WHERE (
                   ask.execution_state = 'started'
                   OR (
                       ask.state = 'pending'
                       AND ask.execution_state = 'pending'
                       AND ask.execution_outcome =
                           'Previous attempt started; outcome is uncertain.'
                   )
               )
           AND NOT EXISTS (
               SELECT 1 FROM run_transcripts AS rt
               WHERE rt.run_id = ask.run_id
                 AND rt.status IN ('running', 'suspended')
           );
        """
    )


def downgrade() -> None:
    # The current revision-025 schema owns this column. 026 exists only for
    # databases stamped by an older copy of 025, so stepping back to the
    # current 025 must leave it in place.
    pass
