"""Exportify-CSV → yt-dlp downloads. Shared by the CLI script and API.

Two halves:

- :func:`parse_csv` extracts ``(artist, title, album, duration_s)`` rows
  from raw CSV text. Pure stdlib so it works in tests.
- :func:`download_track` runs the actual yt-dlp call. Out of the request
  thread; library calls into :func:`subprocess.run`.

The CLI script (``scripts/yt_dlp_csv_import.py``) still owns the entry
point, but uses these helpers so behavior stays identical between
"paste CSV into the UI" and "run the script from the terminal".
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def expected_target(library: Path, row: CsvRow) -> Path:
    """The path we'd write to for this CSV row."""
    safe = f"{sanitize_filename(row.artist)} - {sanitize_filename(row.title)}"
    return library / f"{safe}.mp3"


def download_track(
    row: CsvRow,
    library: Path,
    cookies_file: Path | None = None,
    ffmpeg_location: str | None = None,
    duration_tolerance_s: int = 30,
    timeout_s: int = 180,
) -> tuple[str, str]:
    """Run yt-dlp for one row. Returns ``(status, message)``.

    ``status`` is one of:
      - ``"ok"``: new file downloaded
      - ``"skip"``: file already existed
      - ``"fail"``: yt-dlp failed both attempts

    Idempotent: if ``expected_target(...)`` already exists with size >100 KB,
    returns ``("skip", "already downloaded")``.

    Auth: pass ``cookies_file`` (typically ``~/.dance/cookies.txt`` exported
    via the "Get cookies.txt LOCALLY" Chrome extension) to bypass YouTube's
    bot-detection. Defaults to looking for the file at the standard path
    and to ``None`` (no cookies) if it's missing — yt-dlp will still try
    the request but is much more likely to hit a 403 or sign-in prompt.

    ``ffmpeg_location`` defaults to whatever ``ffmpeg`` is on PATH. Override
    only when you have a known-good static build you want to pin to.
    """
    # Default lookup for cookies.txt — caller can override or pass None.
    if cookies_file is None:
        default_cookies = Path.home() / ".dance" / "cookies.txt"
        if default_cookies.exists():
            cookies_file = default_cookies
    if ffmpeg_location is None:
        ffmpeg_location = "ffmpeg"

    target = expected_target(library, row)
    if target.exists() and target.stat().st_size > 100 * 1024:
        return ("skip", "already downloaded")

    # Fail fast with a helpful message when yt-dlp isn't installed.
    if shutil.which("yt-dlp") is None:
        return (
            "fail",
            "yt-dlp not found on PATH — install with `brew install yt-dlp` "
            "(macOS) or `pipx install yt-dlp`",
        )

    output_template = str(library / f"{target.stem}.%(ext)s")
    query = f"ytsearch1:{row.artist} {row.title}"

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--ffmpeg-location", ffmpeg_location,
        "--default-search", "ytsearch",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-b:a 320k",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--output", output_template,
    ]
    if cookies_file is not None:
        # --cookies path is portable across machines (unlike
        # --cookies-from-browser, which assumes a specific Chrome profile).
        cmd[1:1] = ["--cookies", str(cookies_file)]
    if row.duration_s:
        cmd += [
            "--match-filter",
            (f"duration >= {max(row.duration_s - duration_tolerance_s, 30)} & "
             f"duration <= {row.duration_s + duration_tolerance_s}"),
        ]
    cmd.append(query)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return ("fail", f"yt-dlp timeout after {timeout_s}s")

    if proc.returncode != 0 or not target.exists():
        # Retry once without duration filter.
        retry = [c for c in cmd if c != "--match-filter"]
        retry = [c for c in retry if not c.startswith("duration ")]
        try:
            proc = subprocess.run(retry, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return ("fail", f"yt-dlp timeout (retry) after {timeout_s}s")
        if proc.returncode != 0 or not target.exists():
            tail = (proc.stderr or proc.stdout or "")[-150:].replace("\n", " | ")
            return ("fail", f"yt-dlp rc={proc.returncode}: {tail}")

    # Overwrite yt-dlp's noisy ID3 with clean values from CSV.
    try:
        from mutagen.easyid3 import EasyID3
        from mutagen.id3 import ID3NoHeaderError
        from mutagen.mp3 import MP3

        try:
            tags = EasyID3(target)
        except ID3NoHeaderError:
            audio = MP3(target)
            audio.add_tags()
            audio.save()
            tags = EasyID3(target)
        tags["artist"] = row.artist
        tags["title"] = row.title
        if row.album:
            tags["album"] = row.album
        tags.save()
    except Exception as e:  # noqa: BLE001
        return ("ok", f"{target.stat().st_size // 1024} KB (tag write warn: {e})")

    return ("ok", f"{target.stat().st_size // 1024} KB")


__all__ = [
    "CsvRow",
    "download_track",
    "expected_target",
    "parse_csv",
    "sanitize_filename",
]
