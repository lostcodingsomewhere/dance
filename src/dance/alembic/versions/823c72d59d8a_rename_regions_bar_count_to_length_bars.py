"""rename regions.bar_count to length_bars

Revision ID: 823c72d59d8a
Revises: 95ce5599f3d5
Create Date: 2026-05-10 13:41:12.002264

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '823c72d59d8a'
down_revision: Union[str, Sequence[str], None] = '95ce5599f3d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename ``regions.bar_count`` → ``regions.length_bars`` to disambiguate
    from ``phrases.bar_count`` which has different semantics."""
    with op.batch_alter_table("regions") as batch_op:
        batch_op.alter_column("bar_count", new_column_name="length_bars")


def downgrade() -> None:
    with op.batch_alter_table("regions") as batch_op:
        batch_op.alter_column("length_bars", new_column_name="bar_count")
