"""add sets and set_tracks tables

Revision ID: c6e3ba31c1f0
Revises: 823c72d59d8a
Create Date: 2026-05-23 18:44:54.023241

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c6e3ba31c1f0'
down_revision: Union[str, Sequence[str], None] = '823c72d59d8a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ``sets`` and ``set_tracks`` tables, plus the partial unique index
    enforcing exactly-one active Set (created via raw DDL — SQLAlchemy /
    autogenerate can't model ``UNIQUE ... WHERE`` portably; same pattern as
    the ``audio_analysis`` partial indexes)."""
    op.create_table(
        "sets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "set_tracks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "set_id",
            sa.Integer(),
            sa.ForeignKey("sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "track_id",
            sa.Integer(),
            sa.ForeignKey("tracks.id"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("added_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("set_id", "position", name="uq_set_tracks_set_position"),
    )
    op.create_index(
        "ix_set_tracks_set_position",
        "set_tracks",
        ["set_id", "position"],
    )

    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_sets_one_active "
        "ON sets(is_active) WHERE is_active"
    )


def downgrade() -> None:
    """Drop ``set_tracks`` then ``sets`` (FK cascade order)."""
    op.execute("DROP INDEX IF EXISTS uq_sets_one_active")
    op.drop_index("ix_set_tracks_set_position", table_name="set_tracks")
    op.drop_table("set_tracks")
    op.drop_table("sets")
