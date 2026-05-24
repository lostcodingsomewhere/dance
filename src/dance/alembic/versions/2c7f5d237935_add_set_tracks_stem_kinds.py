"""add set_tracks.stem_kinds

Revision ID: 2c7f5d237935
Revises: c6e3ba31c1f0
Create Date: 2026-05-23 20:55:03.195768

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2c7f5d237935'
down_revision: Union[str, Sequence[str], None] = 'c6e3ba31c1f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable ``stem_kinds`` (JSON-as-Text) to ``set_tracks``.

    NULL means "load all stems" (the existing behavior); a JSON list like
    ``["drums","vocals"]`` means "only load these stems when this slot is
    force-loaded." JSON-in-Text follows the project's pattern (see
    ``TrackEdge.meta``)."""
    with op.batch_alter_table("set_tracks") as batch_op:
        batch_op.add_column(sa.Column("stem_kinds", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("set_tracks") as batch_op:
        batch_op.drop_column("stem_kinds")
