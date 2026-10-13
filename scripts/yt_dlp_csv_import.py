#!/usr/bin/env python3
"""
Download a playlist (exported as Exportify CSV) via yt-dlp into the dance library.

Bypasses Spotify's Web API entirely (which deprecated /v1/playlists/{id}/tracks
for new dev apps on 2024-11-27, see docs/troubleshooting.md). Use:

    1. Export your Spotify playlist to CSV via https://exportify.net
    2. Drop the CSV anywhere
    3. python scripts/yt_dlp_csv_import.py path/to/playlist.csv

Searches YouTube for ten candidates per row, scores them against the CSV's
artist / title / duration, downloads the best eligible one by video id,
verifies the file (duration + full decode), and re-tags it with the CSV
identity plus a provenance comment naming the YouTube URL. Rows where no
candidate qualifies fail with the best near-miss in the message — there is
no "take whatever ranked first" fallback. See dance.spotify.matching.

REQUIRED — see docs/troubleshooting.md "spotdl rate limiting / Spotify Web API
restrictions" for the full YouTube anti-bot stack you need installed:

    pip install -U yt-dlp yt-dlp-ejs bgutil-ytdlp-pot-provider
    brew install node ffmpeg
    docker run --name bgutil-provider -d --restart unless-stopped \
        -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider:latest

The bgutil HTTP server must be running at http://127.0.0.1:4416 throughout.
You also need to be logged into YouTube on Chrome (any profile).

USAGE:
    python scripts/yt_dlp_csv_import.py CSV_PATH [--library DIR] [--workers N]
                                                 [--chrome-profile NAME] [--format mp3|m4a]

Resumable: skips files that already exist with size > 100 KB.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dance.spotify.csv_importer import CsvRow, download_track, expected_target
    from dance.spotify.download_pool import DEFAULT_DOWNLOAD_WORKERS
except ImportError:
    sys.exit("dance not importable. Run from the repo venv: pip install -e '.[dev]'")


DEFAULT_LIBRARY = Path.home() / "Music" / "DJ" / "library"
DEFAULT_WORKERS = DEFAULT_DOWNLOAD_WORKERS
DEFAULT_CHROME_PROFILE = "Profile 2"
FFMPEG = "/Users/arya/.spotdl/ffmpeg"  # bundled by spotDL; fall through to PATH if missing


def primary_artist(artist_field: str) -> str:
    """Exportify joins multiple artists with comma+space OR semicolon."""
    for sep in (",", ";"):
        if sep in artist_field:
            return artist_field.split(sep)[0].strip()
    return artist_field.strip()


def download_one(
    row: dict, library: Path, chrome_profile: str, audio_format: str = "mp3"
) -> tuple[str, str, str]:
    """Returns (status, artist_title, message).

    Delegates to ``dance.spotify.csv_importer.download_track`` — the same
    scored search, hard duration gate, post-download verification and
    provenance tagging the API ingest uses. (This script used to carry its
    own copy of the old ``ytsearch1`` + unfiltered-retry logic.)
    """
    artist = primary_artist(row.get("Artist Name(s)") or row.get("Artist Name", ""))
    title = (row.get("Track Name") or "").strip()
    if not artist or not title:
        return ("skip", "<missing>", "no artist/title in CSV")

    try:
        duration_s = int(row.get("Duration (ms)", "0") or 0) // 1000
    except ValueError:
        duration_s = 0
    csv_row = CsvRow(
        artist=artist,
        title=title,
        album=(row.get("Album Name") or "").strip(),
        duration_s=duration_s,
    )
    name = expected_target(library, csv_row, audio_format).stem
    status, msg = download_track(
        csv_row,
        library,
        ffmpeg_location=FFMPEG if Path(FFMPEG).exists() else None,
        cookies_from_browser=f"chrome:{chrome_profile}",
        audio_format=audio_format,
    )
    return (status, name, msg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download an Exportify CSV playlist via yt-dlp into the dance library.",
    )
    parser.add_argument("csv_path", type=Path, help="Path to Exportify CSV")
    parser.add_argument(
        "--library", type=Path, default=DEFAULT_LIBRARY,
        help=f"Output directory (default: {DEFAULT_LIBRARY})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel yt-dlp processes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--chrome-profile", default=DEFAULT_CHROME_PROFILE,
        help=f"Chrome profile to read cookies from (default: '{DEFAULT_CHROME_PROFILE}')",
    )
    parser.add_argument(
        "--format", default="mp3", choices=("mp3", "m4a"),
        help="mp3 (re-encode to 320k, default) or m4a (keep YouTube's AAC stream)",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"CSV not found: {args.csv_path}", file=sys.stderr)
        return 2

    args.library.mkdir(parents=True, exist_ok=True)
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[dl] {len(rows)} tracks in CSV, {args.workers} workers, "
          f"output: {args.library}")
    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "warn": 0, "fail": 0}
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, row, args.library, args.chrome_profile, args.format): row
            for row in rows
        }
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                status, name, msg = fut.result()
            except Exception as e:
                status, name, msg = "fail", "<exc>", str(e)
            counts[status] = counts.get(status, 0) + 1
            mark = {"ok": "OK", "skip": "--", "warn": "!!", "fail": "XX"}.get(status, "??")
            print(f"[{i:3d}/{len(rows)}] {mark} {name[:60]:60s} {msg[:60]}",
                  flush=True)
            if status == "fail":
                failures.append((name, msg))

    elapsed = time.time() - t0
    print(f"\n[dl] done in {elapsed:.0f}s — ok={counts['ok']} skip={counts['skip']} "
          f"warn={counts.get('warn',0)} fail={counts.get('fail',0)}")
    if failures:
        print("\n[dl] failures:")
        for name, msg in failures:
            print(f"  XX {name}: {msg}")
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
