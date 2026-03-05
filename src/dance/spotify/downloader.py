"""
Spotify playlist downloader using spotDL.

Wraps spotDL to download tracks from a Spotify playlist,
preserving metadata and supporting idempotent syncing.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dance.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""

    downloaded: int  # Number of newly downloaded tracks
    skipped: int  # Number of already existing tracks
    failed: int  # Number of failed downloads
    errors: list[str]  # Error messages for failed downloads


class SpotifyDownloader:
    """
    Downloads tracks from Spotify playlists using spotDL.

    spotDL is idempotent - running multiple times only downloads new tracks.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.output_dir = settings.library_dir
        self.audio_format = settings.audio_format
        self.audio_quality = settings.audio_quality

    def _check_spotdl_installed(self) -> bool:
        """Check if spotDL is installed and accessible."""
        try:
            result = subprocess.run(
                ["spotdl", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def sync_playlist(
        self,
        playlist_url: Optional[str] = None,
        dry_run: bool = False,
    ) -> DownloadResult:
        """
        Sync tracks from a Spotify playlist.

        Args:
            playlist_url: Spotify playlist URL. Uses settings if not provided.
            dry_run: If True, only report what would be downloaded.

        Returns:
            DownloadResult with counts of downloaded/skipped/failed tracks.
        """
        url = playlist_url or self.settings.spotify_playlist_url

        if not url:
            raise ValueError(
                "No Spotify playlist URL configured. "
                "Run 'dance config --spotify-playlist <url>' first."
            )

        if not self._check_spotdl_installed():
            raise RuntimeError(
                "spotDL is not installed. Install with: pip install spotdl"
            )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build spotDL command
        cmd = [
            "spotdl",
            "sync",
            url,
            "--output", str(self.output_dir),
            "--format", self.audio_format,
            "--bitrate", self.audio_quality,
            # Output template: Artist - Title
            "--output", str(self.output_dir / "{artist} - {title}.{output-ext}"),
            # Save sync state for idempotency
            "--save-file", str(self.settings.data_dir / "spotdl_sync.spotdl"),
            # Don't re-download existing files
            "--overwrite", "skip",
        ]

        if dry_run:
            cmd.append("--print-errors")
            logger.info(f"Dry run - would execute: {' '.join(cmd)}")
            return DownloadResult(downloaded=0, skipped=0, failed=0, errors=[])

        logger.info(f"Starting playlist sync: {url}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Run spotDL
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for large playlists
                cwd=str(self.output_dir),
            )

            # Parse output to get counts
            stdout = result.stdout
            stderr = result.stderr

            logger.debug(f"spotDL stdout: {stdout}")
            if stderr:
                logger.warning(f"spotDL stderr: {stderr}")

            # Parse results from spotDL output
            downloaded = self._parse_download_count(stdout)
            skipped = self._parse_skipped_count(stdout)
            failed, errors = self._parse_errors(stdout, stderr)

            if result.returncode != 0 and downloaded == 0:
                logger.error(f"spotDL failed with return code {result.returncode}")
                errors.append(f"spotDL exited with code {result.returncode}")

            return DownloadResult(
                downloaded=downloaded,
                skipped=skipped,
                failed=failed,
                errors=errors,
            )

        except subprocess.TimeoutExpired:
            logger.error("spotDL timed out after 1 hour")
            return DownloadResult(
                downloaded=0, skipped=0, failed=1,
                errors=["Download timed out after 1 hour"],
            )

    def download_track(self, track_url: str) -> Optional[Path]:
        """
        Download a single track from Spotify.

        Args:
            track_url: Spotify track URL.

        Returns:
            Path to downloaded file, or None if failed.
        """
        if not self._check_spotdl_installed():
            raise RuntimeError("spotDL is not installed")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "spotdl",
            "download",
            track_url,
            "--output", str(self.output_dir / "{artist} - {title}.{output-ext}"),
            "--format", self.audio_format,
            "--bitrate", self.audio_quality,
            "--overwrite", "skip",
        ]

        logger.info(f"Downloading track: {track_url}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per track
                cwd=str(self.output_dir),
            )

            if result.returncode != 0:
                logger.error(f"Failed to download track: {result.stderr}")
                return None

            # Find the downloaded file (most recently modified)
            files = sorted(
                self.output_dir.glob(f"*.{self.audio_format}"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

            if files:
                return files[0]

            return None

        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {track_url}")
            return None

    def _parse_download_count(self, output: str) -> int:
        """Parse number of downloaded tracks from spotDL output."""
        # spotDL output varies, try to find download count
        import re

        # Look for patterns like "Downloaded 5 songs"
        match = re.search(r"Downloaded\s+(\d+)", output, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Count "Downloaded:" lines
        downloaded_lines = output.count("Downloaded:")
        if downloaded_lines > 0:
            return downloaded_lines

        return 0

    def _parse_skipped_count(self, output: str) -> int:
        """Parse number of skipped tracks from spotDL output."""
        import re

        # Look for patterns like "Skipped 10 songs"
        match = re.search(r"Skipped\s+(\d+)", output, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Count "Skipping" lines
        skipped_lines = output.count("Skipping")
        if skipped_lines > 0:
            return skipped_lines

        return 0

    def _parse_errors(self, stdout: str, stderr: str) -> tuple[int, list[str]]:
        """Parse errors from spotDL output."""
        errors = []

        # Look for error patterns
        for line in (stdout + "\n" + stderr).split("\n"):
            line = line.strip()
            if any(word in line.lower() for word in ["error", "failed", "couldn't"]):
                if line and line not in errors:
                    errors.append(line)

        return len(errors), errors


def sync_playlist(settings: Optional[Settings] = None, dry_run: bool = False) -> DownloadResult:
    """
    Convenience function to sync the configured playlist.

    Args:
        settings: Settings to use. Uses global settings if not provided.
        dry_run: If True, only report what would be downloaded.

    Returns:
        DownloadResult with sync statistics.
    """
    if settings is None:
        from dance.config import get_settings
        settings = get_settings()

    downloader = SpotifyDownloader(settings)
    return downloader.sync_playlist(dry_run=dry_run)
