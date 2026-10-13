"""Search → pick → download → verify → tag one track with yt-dlp.

The decisions live in :mod:`dance.spotify.matching` (pure, unit-testable);
this module is the side-effectful half that runs yt-dlp / ffprobe / ffmpeg
and moves files. Every subprocess goes through an injectable ``runner``
(``subprocess.run`` by default) so tests can fake yt-dlp without touching
the network.

One attempt per eligible candidate, best first:

1. ``yt-dlp`` downloads **that video id** (never a free-text query) into a
   private scratch directory.
2. ``ffprobe`` measures the duration; it must be within max(10 s, 3.5 %) of
   the Spotify duration when one is known.
3. ``ffmpeg -v error -i F -f null -`` decodes every packet; any error output
   rejects the file, and the *decoded* length must agree with the header —
   a truncated MP3 keeps its Xing header and probes at full length.
4. Tags are rewritten with the Spotify identity, plus a provenance comment
   naming the YouTube URL and match score — so a wrong pick can be traced
   back without a schema change.
5. The file is moved into the library atomically.

Any failure deletes that attempt's scratch directory — only files this
attempt created — and moves on to the next candidate. Nothing in the
library is touched until a file has passed every check.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dance.spotify.matching import (
    MAX_DOWNLOAD_ATTEMPTS,
    SEARCH_LIMIT,
    UNKNOWN_TARGET_MAX_S,
    Candidate,
    ScoredCandidate,
    Target,
    eligible,
    no_candidate_message,
    parse_search_results,
    passes_verification,
    rank_candidates,
    search_queries,
    verify_tolerance_s,
)

logger = logging.getLogger(__name__)

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

#: Politeness towards YouTube, on every yt-dlp call. Each flag verified
#: against the installed yt-dlp (2026.03.17) ``--help``:
#: ``--sleep-requests`` sleeps between extraction requests, the
#: ``--sleep-interval``/``--max-sleep-interval`` pair sleeps a random 2-5 s
#: before each download, and ``--retries 3 --retry-sleep exp=1:20`` backs off
#: exponentially on HTTP errors instead of retrying 10x immediately.
POLITE_FLAGS: tuple[str, ...] = (
    "--sleep-requests",
    "1",
    "--sleep-interval",
    "2",
    "--max-sleep-interval",
    "5",
    "--retries",
    "3",
    "--retry-sleep",
    "exp=1:20",
)

SEARCH_TIMEOUT_S = 90


class AttemptError(RuntimeError):
    """One candidate failed download or verification; try the next."""


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioFormatSpec:
    """How to ask yt-dlp for one ``settings.audio_format``."""

    name: str
    ext: str
    yt_dlp_args: tuple[str, ...]
    #: ffprobe ``codec_name`` the result must have, or None to skip the check.
    codec: str | None


def audio_format_spec(audio_format: str | None) -> AudioFormatSpec:
    """Translate ``settings.audio_format`` into yt-dlp arguments.

    - ``mp3`` (default; the user's ``.env`` pins it): best audio re-encoded
      to 320 kb/s MP3 — the historical behaviour.
    - ``m4a``: YouTube's AAC stream (``bestaudio[ext=m4a]``) kept as-is. No
      re-encode, so no second lossy generation on top of YouTube's. Uploads
      without an m4a stream fail that attempt and the next candidate runs.
    - ``flac``/``wav``/``opus``/``ogg``: yt-dlp ``--extract-audio`` to that
      container (lossless containers of lossy audio — supported, not advised).
    """
    fmt = (audio_format or "mp3").strip().lower().lstrip(".")
    if fmt == "mp3":
        return AudioFormatSpec(
            name="mp3",
            ext="mp3",
            yt_dlp_args=(
                "-f",
                "bestaudio/best",
                "--extract-audio",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "0",
                "--postprocessor-args",
                "ffmpeg:-b:a 320k",
            ),
            codec="mp3",
        )
    if fmt in ("m4a", "aac"):
        return AudioFormatSpec(
            name="m4a",
            ext="m4a",
            yt_dlp_args=("-f", "bestaudio[ext=m4a]"),
            codec="aac",
        )
    extract = {
        "flac": ("flac", "flac"),
        "wav": ("wav", "wav"),
        "opus": ("opus", "opus"),
        "ogg": ("vorbis", "ogg"),
    }
    if fmt in extract:
        yt_fmt, ext = extract[fmt]
        return AudioFormatSpec(
            name=fmt,
            ext=ext,
            yt_dlp_args=(
                "-f",
                "bestaudio/best",
                "--extract-audio",
                "--audio-format",
                yt_fmt,
                "--audio-quality",
                "0",
            ),
            codec=None,
        )
    raise ValueError(f"unsupported audio_format {audio_format!r} (use mp3 or m4a)")


def target_extension(audio_format: str | None) -> str:
    """File extension (no dot) a download in ``audio_format`` ends up with."""
    try:
        return audio_format_spec(audio_format).ext
    except ValueError:
        return "mp3"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tools:
    yt_dlp: str
    ffmpeg: str
    ffprobe: str
    cookies_file: Path | None = None
    #: e.g. ``"chrome:Profile 2"`` — the CLI script's auth mode.
    cookies_from_browser: str | None = None

    def base_args(self) -> list[str]:
        args: list[str] = []
        if self.cookies_file is not None:
            # --cookies is portable across machines (unlike
            # --cookies-from-browser, which assumes a Chrome profile).
            args += ["--cookies", str(self.cookies_file)]
        if self.cookies_from_browser:
            args += ["--cookies-from-browser", self.cookies_from_browser]
        # Load-bearing for YouTube's 2026 JS challenge; see
        # docs/troubleshooting.md.
        args += ["--js-runtimes", "node", "--ffmpeg-location", self.ffmpeg]
        return args


def _tail(proc: subprocess.CompletedProcess[str], n: int = 150) -> str:
    return (proc.stderr or proc.stdout or "")[-n:].replace("\n", " | ").strip()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_candidates(
    target: Target,
    tools: Tools,
    runner: Runner,
    timeout_s: int = SEARCH_TIMEOUT_S,
) -> tuple[list[ScoredCandidate], list[str]]:
    """Metadata-only search; returns ``(ranked, errors)``.

    Runs the all-artists query first and the primary-artist query only when
    the first produced nothing eligible. Candidates from both are ranked
    together against the full target identity.
    """
    collected: list[Candidate] = []
    errors: list[str] = []
    ranked: list[ScoredCandidate] = []
    for query_index, query in enumerate(search_queries(target)):
        cmd = [
            tools.yt_dlp,
            *tools.base_args(),
            *POLITE_FLAGS,
            "--flat-playlist",
            "-J",
            "--no-warnings",
            f"ytsearch{SEARCH_LIMIT}:{query}",
        ]
        try:
            proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            errors.append(f"search timed out after {timeout_s}s")
            continue
        if proc.returncode != 0:
            errors.append(f"search rc={proc.returncode}: {_tail(proc)}")
            continue
        try:
            payload: Any = json.loads(proc.stdout or "")
        except ValueError:
            errors.append("search returned invalid JSON")
            continue
        collected.extend(parse_search_results(payload, rank_offset=query_index * SEARCH_LIMIT))
        ranked = rank_candidates(target, collected)
        if eligible(ranked):
            break
    return ranked, errors


# ---------------------------------------------------------------------------
# Download + verify
# ---------------------------------------------------------------------------


def download_candidate(
    scored: ScoredCandidate,
    spec: AudioFormatSpec,
    work_dir: Path,
    tools: Tools,
    runner: Runner,
    timeout_s: int,
) -> Path:
    """Download one video id into ``work_dir``. Raises :class:`AttemptError`."""
    cand = scored.candidate
    cmd = [
        tools.yt_dlp,
        *tools.base_args(),
        *POLITE_FLAGS,
        *spec.yt_dlp_args,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--output",
        str(work_dir / f"{cand.video_id}.%(ext)s"),
        cand.url,
    ]
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise AttemptError(f"yt-dlp timeout after {timeout_s}s") from exc
    produced = work_dir / f"{cand.video_id}.{spec.ext}"
    if proc.returncode != 0 or not produced.exists():
        raise AttemptError(f"yt-dlp rc={proc.returncode}: {_tail(proc) or 'no file produced'}")
    return produced


@dataclass(frozen=True)
class Probe:
    duration_s: float
    codec: str | None


def probe_audio(path: Path, tools: Tools, runner: Runner) -> Probe:
    cmd = [
        tools.ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_name,codec_type",
        "-of",
        "json",
        str(path),
    ]
    proc = runner(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise AttemptError(f"ffprobe rc={proc.returncode}: {_tail(proc)}")
    try:
        body = json.loads(proc.stdout or "{}")
        duration = float(body["format"]["duration"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AttemptError("ffprobe reported no duration") from exc
    codec = next(
        (
            s.get("codec_name")
            for s in body.get("streams") or []
            if isinstance(s, dict) and s.get("codec_type") == "audio"
        ),
        None,
    )
    if codec is None:
        raise AttemptError("no audio stream")
    return Probe(duration_s=duration, codec=str(codec))


def decode_check(path: Path, tools: Tools, runner: Runner) -> float | None:
    """Full decode (``ffmpeg -v error -i F -f null -``); returns decoded seconds.

    Any error output rejects the file: a damaged MP3 frame decodes with
    exit 0 but logs "Header missing". The decoded length comes from
    ``-progress`` — the header duration alone cannot be trusted, because a
    truncated MP3 keeps its Xing header and still *probes* at full length.
    """
    cmd = [
        tools.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-progress",
        "pipe:1",
        "-nostats",
        "-",
    ]
    proc = runner(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or (proc.stderr or "").strip():
        err = (proc.stderr or "").strip()[-150:].replace("\n", " | ")
        raise AttemptError(f"decode check failed: {err or f'rc={proc.returncode}'}")
    decoded: float | None = None
    for line in (proc.stdout or "").splitlines():
        key, _, value = line.partition("=")
        if key == "out_time_us":
            try:
                decoded = int(value) / 1_000_000
            except ValueError:
                continue
    return decoded


def _truncation_tolerance_s(header_s: float) -> float:
    return max(2.0, header_s * 0.02)


def _check_expected(measured_s: float, expected_s: float | None, what: str) -> None:
    if passes_verification(expected_s, measured_s):
        return
    if expected_s is None or expected_s <= 0:
        raise AttemptError(
            f"{what} {measured_s:.1f}s over the {UNKNOWN_TARGET_MAX_S:.0f}s cap "
            "(no expected duration)"
        )
    raise AttemptError(
        f"{what} {measured_s:.1f}s vs {expected_s:.1f}s expected "
        f"(tolerance ±{verify_tolerance_s(expected_s):.0f}s)"
    )


def verify_audio(
    path: Path,
    expected_s: float | None,
    spec: AudioFormatSpec,
    tools: Tools,
    runner: Runner,
) -> Probe:
    """Probe, then decode every packet. Raises :class:`AttemptError`.

    Checks, cheapest first: header duration vs Spotify (max(10 s, 3.5 %)),
    codec, full decode with no errors, decoded length vs header (truncation),
    decoded length vs Spotify. The returned duration is the decoded one.
    """
    probe = probe_audio(path, tools, runner)
    _check_expected(probe.duration_s, expected_s, "duration")
    if spec.codec is not None and probe.codec != spec.codec:
        raise AttemptError(f"codec {probe.codec} (expected {spec.codec})")
    decoded = decode_check(path, tools, runner)
    if decoded is None:
        return probe
    if abs(decoded - probe.duration_s) > _truncation_tolerance_s(probe.duration_s):
        raise AttemptError(
            f"decoded {decoded:.1f}s of a {probe.duration_s:.1f}s header (truncated?)"
        )
    _check_expected(decoded, expected_s, "decoded duration")
    return Probe(duration_s=decoded, codec=probe.codec)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagValues:
    title: str
    artist: str
    album: str = ""
    comment: str = ""


def write_tags(path: Path, tags: TagValues) -> None:
    """Overwrite identity tags + provenance comment, whatever the container.

    MP3 → ID3v2 frames (TIT2/TPE1/TALB/COMM), M4A → MP4 atoms
    (©nam/©ART/©alb/©cmt), anything else → mutagen's easy interface.
    """
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        from mutagen.id3 import COMM, ID3, TALB, TIT2, TPE1, ID3NoHeaderError

        try:
            id3 = ID3(path)
        except ID3NoHeaderError:
            id3 = ID3()
        id3.setall("TIT2", [TIT2(encoding=3, text=tags.title)])
        id3.setall("TPE1", [TPE1(encoding=3, text=tags.artist)])
        if tags.album:
            id3.setall("TALB", [TALB(encoding=3, text=tags.album)])
        # yt-dlp/ffmpeg may leave a description in COMM; replace, don't append.
        id3.delall("COMM")
        if tags.comment:
            id3.add(COMM(encoding=3, lang="eng", desc="", text=tags.comment))
        id3.save(path)
        return
    if suffix in (".m4a", ".mp4", ".aac"):
        from mutagen.mp4 import MP4

        mp4 = MP4(path)
        if mp4.tags is None:
            mp4.add_tags()
        assert mp4.tags is not None
        mp4.tags["\xa9nam"] = [tags.title]
        mp4.tags["\xa9ART"] = [tags.artist]
        if tags.album:
            mp4.tags["\xa9alb"] = [tags.album]
        if tags.comment:
            mp4.tags["\xa9cmt"] = [tags.comment]
        mp4.save()
        return
    import mutagen

    audio = mutagen.File(path, easy=True)
    if audio is None:
        raise ValueError(f"mutagen cannot tag {path.name}")
    if audio.tags is None:
        audio.add_tags()
    audio["title"] = tags.title
    audio["artist"] = tags.artist
    if tags.album:
        audio["album"] = tags.album
    if tags.comment:
        try:
            audio["comment"] = tags.comment
        except (KeyError, ValueError):
            logger.debug("container for %s has no easy comment key", path.name)
    audio.save()


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def place_file(src: Path, dest: Path) -> None:
    """Move ``src`` to ``dest`` so ``dest`` never exists half-written.

    Same filesystem: a single ``os.replace``. Across filesystems: copy to a
    ``.part`` sibling (not an audio suffix, so a concurrent library scan
    ignores it), then ``os.replace``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dest)
        return
    except OSError:
        pass
    staging = dest.parent / f".{dest.name}.{uuid.uuid4().hex[:8]}.part"
    try:
        shutil.copyfile(src, staging)
        os.replace(staging, dest)
    finally:
        staging.unlink(missing_ok=True)
    src.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class AcquireResult:
    status: Literal["ok", "fail"]
    message: str
    path: Path | None = None
    chosen: ScoredCandidate | None = None
    measured_s: float | None = None


def provenance_comment(scored: ScoredCandidate, measured_s: float, expected_s: float | None) -> str:
    """Comment-tag text recording where the audio came from."""
    spotify = f", Spotify {expected_s:.1f}s" if expected_s else ""
    return (
        f"dance: downloaded from YouTube {scored.candidate.url} "
        f"(match score {scored.score:.2f}; file {measured_s:.1f}s{spotify})"
    )


def _ok_message(
    dest: Path, scored: ScoredCandidate, measured_s: float, expected_s: float | None
) -> str:
    size_kb = dest.stat().st_size // 1024
    spotify = f" vs {expected_s:.0f}s" if expected_s else ""
    return (
        f"{size_kb} KB · yt:{scored.candidate.video_id} score {scored.score:.2f} "
        f"· {measured_s:.0f}s{spotify}"
    )


def acquire_track(
    target: Target,
    tags: TagValues,
    dest: Path,
    spec: AudioFormatSpec,
    tools: Tools,
    *,
    runner: Runner | None = None,
    timeout_s: int = 180,
    max_attempts: int = MAX_DOWNLOAD_ATTEMPTS,
    work_root: Path | None = None,
) -> AcquireResult:
    """Find, download and verify one recording into ``dest``.

    Never falls back to an unscored or out-of-gate candidate. If no
    candidate qualifies the result is ``fail`` with the best near-miss in
    the message; if every eligible candidate fails download/verification
    the message lists each attempt.
    """
    run: Runner = runner or subprocess.run
    ranked, search_errors = search_candidates(target, tools, run)
    choices = eligible(ranked)
    if not choices:
        msg = no_candidate_message(ranked)
        if search_errors and not ranked:
            msg = f"{msg}; {'; '.join(search_errors)}"
        return AcquireResult(status="fail", message=msg[:300])

    owns_root = work_root is None
    root = Path(tempfile.mkdtemp(prefix="dance-dl-")) if work_root is None else work_root
    failures: list[str] = []
    try:
        for scored in choices[:max_attempts]:
            attempt_dir = Path(tempfile.mkdtemp(prefix=f"{scored.candidate.video_id}-", dir=root))
            try:
                produced = download_candidate(scored, spec, attempt_dir, tools, run, timeout_s)
                probe = verify_audio(produced, target.duration_s, spec, tools, run)
                comment = provenance_comment(scored, probe.duration_s, target.duration_s)
                try:
                    write_tags(produced, TagValues(tags.title, tags.artist, tags.album, comment))
                except Exception as exc:  # noqa: BLE001 — tags are not worth losing audio
                    logger.warning("tag write failed for %s: %s", dest.name, exc)
                place_file(produced, dest)
                logger.info(
                    "acquired %s from %s (score %.2f)",
                    dest.name,
                    scored.candidate.url,
                    scored.score,
                )
                return AcquireResult(
                    status="ok",
                    message=_ok_message(dest, scored, probe.duration_s, target.duration_s),
                    path=dest,
                    chosen=scored,
                    measured_s=probe.duration_s,
                )
            except AttemptError as exc:
                failures.append(f"yt:{scored.candidate.video_id} {exc}")
                logger.info("candidate %s rejected: %s", scored.candidate.video_id, exc)
            finally:
                # Only this attempt's own scratch files are ever deleted.
                shutil.rmtree(attempt_dir, ignore_errors=True)
    finally:
        if owns_root:
            shutil.rmtree(root, ignore_errors=True)

    tried = min(len(choices), max_attempts)
    return AcquireResult(
        status="fail",
        message=f"{tried} eligible candidate(s) failed: {'; '.join(failures)}"[:300],
    )


__all__ = [
    "POLITE_FLAGS",
    "AcquireResult",
    "AttemptError",
    "AudioFormatSpec",
    "TagValues",
    "Tools",
    "acquire_track",
    "audio_format_spec",
    "decode_check",
    "download_candidate",
    "place_file",
    "probe_audio",
    "provenance_comment",
    "search_candidates",
    "target_extension",
    "verify_audio",
    "write_tags",
]
