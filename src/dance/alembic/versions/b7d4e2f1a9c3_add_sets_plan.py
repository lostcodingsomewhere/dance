"""add sets.plan (per-role queue plan)

Revision ID: b7d4e2f1a9c3
Revises: 2c7f5d237935
Create Date: 2026-06-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d4e2f1a9c3'
down_revision: Union[str, Sequence[str], None] = '2c7f5d237935'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a nullable ``plan`` (JSON-as-Text) column to ``sets``.

    A Set's plan lives *inside* the Booth rec grid: each role column holds the
    DJ's queued picks on top, live recs below. We store the queues as
    ``{role: [track_id, ...]}`` (roles = ``PlanRole`` values: drums/bass/
    vocals/other/song). NULL means an empty plan. JSON-in-Text follows the
    project convention (``TrackEdge.meta``, ``SetTrack.stem_kinds``).

    Re-parented onto main's head (``2c7f5d237935``) — main never received the
    abandoned ``set_moves``/``set_scenes`` migrations, so this is a single,
    additive column with no table drops.
    """
    with op.batch_alter_table("sets") as batch_op:
        batch_op.add_column(sa.Column("plan", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("sets") as batch_op:
        batch_op.drop_column("plan")
