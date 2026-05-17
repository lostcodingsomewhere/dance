#!/usr/bin/env python3
"""
Download a playlist (exported as Exportify CSV) via yt-dlp into the dance library.

Bypasses Spotify's Web API entirely (which deprecated /v1/playlists/{id}/tracks
for new dev apps on 2024-11-27, see docs/troubleshooting.md). Use:

    1. Export your Spotify playlist to CSV via https://exportify.net
    2. Drop the CSV anywhere
    3. python scripts/yt_dlp_csv_import.py path/to/playlist.csv

Uses YouTube Music search ("ytsearch1:Artist Title") then re-tags the resulting
MP3 with clean ID3 from the CSV (yt-dlp's default tags pick up YouTube's noisy
video titles like "[Official Audio]" suffixes).

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
                                                 [--chrome-profile NAME]

Resumable: skips files that already exist with size > 100 KB.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3NoHeaderError
    from mutagen.mp3 import MP3
except ImportError:
    sys.exit("mutagen not installed. Run: pip install mutagen")


DEFAULT_LIBRARY = Path.home() / "Music" / "DJ" / "library"
DEFAULT_WORKERS = 4
DEFAULT_CHROME_PROFILE = "Profile 2"
DURATION_TOLERANCE_S = 30  # reject yt-dlp pick if duration is more than 30 s off
FFMPEG = "/Users/arya/.spotdl/ffmpeg"  # bundled by spotDL; fall through to PATH if missing


def sanitize_filename(s: str) -> str:
    """Strip characters that would corrupt the path or trip Live's parser later."""
    s = re.sub(r"[/\\:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]


def primary_artist(artist_field: str) -> str:
    """Exportify joins multiple artists with comma+space OR semicolon."""
    for sep in (",", ";"):
        if sep in artist_field:
            return artist_field.split(sep)[0].strip()
    return artist_field.strip()


def already_downloaded(target: Path) -> bool:
    return target.exists() and target.stat().st_size > 100 * 1024


def download_one(row: dict, library: Path, chrome_profile: str) -> tuple[str, str, str]:
    """Returns (status, artist_title, message)."""
    artist = primary_artist(row.get("Artist Name(s)") or row.get("Artist Name", ""))
    title = (row.get("Track Name") or "").strip()
    if not artist or not title:
        return ("skip", "<missing>", "no artist/title in CSV")

    safe_name = f"{sanitize_filename(artist)} - {sanitize_filename(title)}"
    target = library / f"{safe_name}.mp3"

    if already_downloaded(target):
        return ("skip", safe_name, "already downloaded")

    duration_ms = int(row.get("Duration (ms)", "0") or 0)
    duration_s = duration_ms // 1000 if duration_ms else 0

    query = f"ytsearch1:{artist} {title}"
    output_template = str(library / f"{safe_name}.%(ext)s")

    cmd = [
        "yt-dlp",
        # YouTube 2025-2026 anti-bot stack — ALL flags load-bearing.
        # See docs/troubleshooting.md.
        "--cookies-from-browser", f"chrome:{chrome_profile}",
        "--js-runtimes", "node",
        "--ffmpeg-location", FFMPEG if Path(FFMPEG).exists() else "ffmpeg",
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
    if duration_s:
        # Reject candidates whose duration is way off (avoids 10-min compilations).
        cmd += [
            "--match-filter",
            f"duration >= {max(duration_s - DURATION_TOLERANCE_S, 30)} & "
            f"duration <= {duration_s + DURATION_TOLERANCE_S}",
        ]
    cmd.append(query)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return ("fail", safe_name, "yt-dlp timeout after 180s")

    if proc.returncode != 0 or not target.exists():
        # Retry once without duration filter — better a wrong-length pick than nothing.
        retry = [c for c in cmd if not c.startswith("duration")]
        retry = [c for c in retry if c != "--match-filter"]
        try:
            proc = subprocess.run(retry, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return ("fail", safe_name, "yt-dlp timeout (retry)")
        if proc.returncode != 0 or not target.exists():
            err = (proc.stderr or proc.stdout or "")[-200:].replace("\n", " | ")
            return ("fail", safe_name, f"yt-dlp rc={proc.returncode}: {err}")

    # Overwrite yt-dlp's noisy ID3 with clean values from CSV so dance's ingest
    # stage picks up the right artist/title rather than YouTube's video title.
    try:
        try:
            tags = EasyID3(target)
        except ID3NoHeaderError:
            audio = MP3(target)
            audio.add_tags()
            audio.save()
            tags = EasyID3(target)
        tags["artist"] = artist
        tags["title"] = title
        album = (row.get("Album Name") or "").strip()
        if album:
            tags["album"] = album
        tags.save()
    except Exception as e:
        return ("warn", safe_name, f"downloaded but tag write failed: {e}")

    return ("ok", safe_name, f"{target.stat().st_size // 1024} KB")


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
            ex.submit(download_one, row, args.library, args.chrome_profile): row
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
