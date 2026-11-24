"""add regions.slot (which of a song's five pads a pad region is)

Revision ID: a1c5d9e3f7b2
Revises: e4a91c7b2d05
Create Date: 2026-11-23 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1c5d9e3f7b2"
down_revision: str | Sequence[str] | None = "e4a91c7b2d05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("regions") as batch_op:
        batch_op.add_column(sa.Column("slot", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("regions") as batch_op:
        batch_op.drop_column("slot")
