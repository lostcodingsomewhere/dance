"""Exportify-CSV → yt-dlp downloads. Shared by the CLI script and API.

Two halves:

- :func:`parse_csv` extracts ``(artist, title, album, duration_s)`` rows
  from raw CSV text. Pure stdlib so it works in tests.
- :func:`download_track` finds, downloads and verifies one row (see
  :mod:`dance.spotify.acquire`). Out of the request thread.

The CLI script (``scripts/yt_dlp_csv_import.py``) still owns the entry
point, but uses these helpers so behavior stays identical between
"paste CSV into the UI" and "run the script from the terminal".
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from dance.spotify.acquire import (
    Runner,
    TagValues,
    Tools,
    acquire_track,
    audio_format_spec,
    target_extension,
)
from dance.spotify.matching import Target, split_artists


def _resolve_yt_dlp() -> str | None:
    """Find yt-dlp, preferring the active venv's copy.

    spotDL installs yt-dlp as a pip dependency, so the venv's bin is the
    most reliable location. If the venv isn't the calling process, fall
    back to ``shutil.which`` for a system-wide install. Returns None if
    neither path exists — callers should surface a helpful error then."""
    venv_yt = Path(sys.executable).parent / "yt-dlp"
    if venv_yt.exists():
        return str(venv_yt)
    return shutil.which("yt-dlp")


def _resolve_ffmpeg(explicit: str | None) -> str:
    """ffmpeg lives system-wide (brew /opt/homebrew/bin/ffmpeg on Apple
    Silicon, /usr/local/bin on Intel). Honor an explicit override first,
    then try the standard Homebrew prefixes, then fall back to the bare
    name so yt-dlp's own PATH lookup gets a shot."""
    if explicit and explicit != "ffmpeg":
        return explicit
    candidates = [
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return "ffmpeg"


@dataclass(frozen=True)
class CsvRow:
    artist: str
    title: str
    album: str
    duration_s: int  # 0 if unknown


def _primary_artist(field: str) -> str:
    """Exportify joins multiple artists with comma OR semicolon."""
    for sep in (",", ";"):
        if sep in field:
            return field.split(sep)[0].strip()
    return field.strip()


def sanitize_filename(s: str) -> str:
    """Strip characters that would corrupt the path or trip Live's parser."""
    s = re.sub(r"[/\\:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]


def parse_csv(csv_text: str) -> tuple[list[CsvRow], list[str]]:
    """Parse Exportify CSV text. Returns ``(rows, parse_errors)``.

    Tolerant: a malformed row is logged in ``parse_errors`` but doesn't
    abort the whole import. UTF-8 BOMs (which Exportify includes) are
    handled by the ``csv`` module via the dict-key value, not key normalization.
    """
    rows: list[CsvRow] = []
    errors: list[str] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        errors.append("CSV has no header row.")
        return rows, errors

    # Find the column names tolerantly — Exportify variants exist.
    def _find(needle: str) -> str | None:
        for col in reader.fieldnames or []:
            # Strip BOM and case-fold for comparison.
            stripped = col.lstrip("﻿").strip().lower()
            if needle in stripped:
                return col
        return None

    artist_col = _find("artist name") or _find("artist")
    title_col = _find("track name") or _find("title")
    album_col = _find("album name") or _find("album")
    duration_col = _find("duration (ms)") or _find("duration")

    if not artist_col or not title_col:
        errors.append(
            f"CSV missing required columns. "
            f"Found: {reader.fieldnames!r}. Need 'Artist Name(s)' and 'Track Name'."
        )
        return rows, errors

    for i, raw in enumerate(reader, start=2):  # row 1 is header
        artist = _primary_artist(raw.get(artist_col, "") or "")
        title = (raw.get(title_col, "") or "").strip()
        if not artist or not title:
            errors.append(f"row {i}: missing artist or title")
            continue
        album = (raw.get(album_col, "") or "").strip() if album_col else ""
        duration_s = 0
        if duration_col:
            try:
                duration_s = int(raw.get(duration_col, "0") or 0) // 1000
            except ValueError:
                pass
        rows.append(CsvRow(artist=artist, title=title, album=album, duration_s=duration_s))

    return rows, errors


def expected_target(library: Path, row: CsvRow, audio_format: str = "mp3") -> Path:
    """The path we'd write to for this CSV row.

    The extension follows ``audio_format`` (``settings.audio_format``): an
    ``m4a`` library keeps YouTube's AAC stream rather than re-encoding it.
    """
    safe = f"{sanitize_filename(row.artist)} - {sanitize_filename(row.title)}"
    return library / f"{safe}.{target_extension(audio_format)}"


def _resolve_ffprobe(ffmpeg_path: str) -> str:
    """ffprobe ships beside ffmpeg in every install we support."""
    sibling = Path(ffmpeg_path).with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"


def download_track(
    row: CsvRow,
    library: Path,
    cookies_file: Path | None = None,
    ffmpeg_location: str | None = None,
    *,
    timeout_s: int = 180,
    audio_format: str = "mp3",
    cookies_from_browser: str | None = None,
    runner: Runner | None = None,
    dest: Path | None = None,
) -> tuple[str, str]:
    """Find, download and verify one row. Returns ``(status, message)``.

    ``status`` is one of:
      - ``"ok"``: new file downloaded and verified; ``message`` names the
        chosen YouTube id + match score (also written to the comment tag)
      - ``"skip"``: file already existed
      - ``"fail"``: no candidate qualified, or every eligible one failed
        download/verification — ``message`` says which and why

    Candidate selection lives in :mod:`dance.spotify.matching`; the
    download/verify loop in :mod:`dance.spotify.acquire`. There is **no
    unfiltered retry**: the old "retry without --match-filter" fallback
    accepted whatever YouTube ranked first (live versions, extended mixes,
    hour-long sets) and then stamped the Spotify tags on it.

    Idempotent: if ``expected_target(...)`` (or ``dest``, when the caller
    picked the path) already exists with size >100 KB, returns
    ``("skip", "already downloaded")``.

    Auth: pass ``cookies_file`` (typically ``~/.dance/cookies.txt`` exported
    via the "Get cookies.txt LOCALLY" Chrome extension) to bypass YouTube's
    bot-detection. Defaults to looking for the file at the standard path
    and to ``None`` (no cookies) if it's missing — yt-dlp will still try
    the request but is much more likely to hit a 403 or sign-in prompt.
    ``cookies_from_browser`` (e.g. ``"chrome:Profile 2"``) is the CLI
    script's alternative.

    ``ffmpeg_location`` defaults to whatever ``ffmpeg`` is on PATH. Override
    only when you have a known-good static build you want to pin to.
    ``runner`` replaces ``subprocess.run`` (tests fake yt-dlp with it).
    """
    # Default lookup for cookies.txt — caller can override or pass None.
    if cookies_file is None and cookies_from_browser is None:
        default_cookies = Path.home() / ".dance" / "cookies.txt"
        if default_cookies.exists():
            cookies_file = default_cookies
    ffmpeg_location = _resolve_ffmpeg(ffmpeg_location)

    try:
        spec = audio_format_spec(audio_format)
    except ValueError as exc:
        return ("fail", str(exc))

    target = dest or expected_target(library, row, spec.ext)
    if target.exists() and target.stat().st_size > 100 * 1024:
        return ("skip", "already downloaded")

    # Prefer the venv's yt-dlp (installed via pip alongside spotdl) — this
    # keeps the backend working even when launched without the venv on PATH.
    yt_dlp_bin = _resolve_yt_dlp()
    if yt_dlp_bin is None:
        return (
            "fail",
            "yt-dlp not found — install with `pip install yt-dlp` into the "
            "venv or `brew install yt-dlp` on the host",
        )

    tools = Tools(
        yt_dlp=yt_dlp_bin,
        ffmpeg=ffmpeg_location,
        ffprobe=_resolve_ffprobe(ffmpeg_location),
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )
    match_target = Target(
        title=row.title,
        artists=split_artists(row.artist) or (row.artist,),
        duration_s=float(row.duration_s) if row.duration_s else None,
    )
    result = acquire_track(
        match_target,
        TagValues(title=row.title, artist=row.artist, album=row.album),
        target,
        spec,
        tools,
        runner=runner,
        timeout_s=timeout_s,
    )
    return (result.status, result.message)


__all__ = [
    "CsvRow",
    "download_track",
    "expected_target",
    "parse_csv",
    "sanitize_filename",
]
