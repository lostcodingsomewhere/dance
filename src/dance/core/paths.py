"""Filesystem-path helpers shared across the pipeline and API.

The one job of this module is Unicode normalization. macOS's HFS+/APFS will
happily store a filename in either NFC or NFD form, and the two are *different
byte strings* even though they render identically. When a library is rsynced
between machines the on-disk filenames can land as NFC while a DB row recorded
at ingest time still holds the NFD form (or vice-versa). ``os.path.exists`` then
returns ``False`` for any track whose name contains a non-ASCII character —
silently dropping accented-title tracks from the whole pipeline.

We pick NFC as the canonical form (it matches what most tooling and rsync emit)
and normalize *every* path string before it is persisted or compared against the
filesystem, so the DB and disk always agree.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path


def nfc_path(path: str | os.PathLike[str] | Path) -> str:
    """Return ``path`` as a string in Unicode NFC form.

    Use this anywhere a path is about to be stored in the DB or compared
    against a value that may have come from the filesystem.
    """
    return unicodedata.normalize("NFC", os.fspath(path))


__all__ = ["nfc_path"]
