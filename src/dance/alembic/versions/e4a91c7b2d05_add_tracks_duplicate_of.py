"""add tracks.duplicate_of (mark redundant copies, don't delete them)

Revision ID: e4a91c7b2d05
Revises: b7d4e2f1a9c3
Create Date: 2026-08-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e4a91c7b2d05"
down_revision: Union[str, Sequence[str], None] = "b7d4e2f1a9c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a nullable self-referencing ``duplicate_of`` to ``tracks``.

    The library was assembled over three ingest runs (2026-05-17 / 05-26 /
    06-13) that pulled overlapping songs under slightly different filenames.
    Different bytes → different ``file_hash`` → ingest's hash-based dedup
    never fired, so each copy became its own row:

        #67  Dot Major;Kitty Amor - Navi - Kitty Amor Remix.mp3   211.8s
        #246 Dot Major, Kitty Amor - Navi - Kitty Amor Remix.mp3  211.8s
        #374 Dot Major - Navi - Kitty Amor Remix.mp3              211.8s

    Measured: 82 groups, 145 redundant rows of 353 (41%). See
    docs/proposals/library-duplicates.md.

    A MARKER rather than a delete, deliberately: ``session_plays`` and
    ``track_edges`` reference these rows (61,508 edges were built over a
    library that is 41% redundant), and the audio is still on disk. Marking
    is fully reversible — ``UPDATE tracks SET duplicate_of = NULL`` restores
    the previous state exactly.

    Purely additive: nullable column + index, no backfill here. The backfill
    is a separate, re-runnable CLI step (``dance dedupe``) so the schema
    change and the data change can be reviewed and reverted independently.
    """
    with op.batch_alter_table("tracks") as batch_op:
        batch_op.add_column(sa.Column("duplicate_of", sa.Integer(), nullable=True))
        batch_op.create_index(
            "ix_tracks_duplicate_of", ["duplicate_of"], unique=False
        )
        batch_op.create_foreign_key(
            "fk_tracks_duplicate_of",
            "tracks",
            ["duplicate_of"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("tracks") as batch_op:
        batch_op.drop_constraint("fk_tracks_duplicate_of", type_="foreignkey")
        batch_op.drop_index("ix_tracks_duplicate_of")
        batch_op.drop_column("duplicate_of")
