"""
Fast environment preflight checks.

These run *before* any real work so a missing native dependency fails loudly
and actionably instead of silently breaking a stage deep in the pipeline.

``ffmpeg`` is a hard, currently-undeclared dependency: spotDL shells out to it
for downloads, and demucs / librosa / soundfile rely on it (or its libs) for
decoding. It was silently missing on this machine once and broke everything
with opaque errors, so we check for it up front.

Mirrors the style of ``SpotifyDownloader._check_spotdl_installed`` — a tiny,
side-effect-free check that callers turn into a clear message.
"""

from __future__ import annotations

import shutil


class PreflightError(RuntimeError):
    """A required external tool is missing. Message is user-facing."""


def ffmpeg_available() -> bool:
    """Return True iff an ``ffmpeg`` executable is on PATH."""
    return shutil.which("ffmpeg") is not None


def require_ffmpeg() -> None:
    """Raise :class:`PreflightError` with an actionable message if ffmpeg is
    missing.

    ffmpeg is needed by both ``dance sync`` (spotDL downloads) and
    ``dance process`` (demucs / librosa / soundfile decoding). Call this at
    the very start of those commands, before doing any work.
    """
    if not ffmpeg_available():
        raise PreflightError(
            "ffmpeg not found — install with: brew install ffmpeg\n"
            "(ffmpeg is required for spotDL downloads and audio decoding "
            "during processing.)"
        )
