"""
Command-line interface for Dance.

Commands:
- config: Configure Spotify playlist and settings
- sync: Download new tracks from Spotify
- process: Run the analysis pipeline on pending tracks
- list: List tracks with filters
- run: Full pipeline (sync → process)
- status: Show pipeline state counts
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from dance import __version__
from dance.config import Settings, get_settings
from dance.core.database import (
    Analysis,
    Track,
    TrackState,
    get_session,
    init_db,
)

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Dance — stem-performance brain for Ableton Live."""
    ctx.ensure_object(dict)
    settings = get_settings()
    ctx.obj["settings"] = settings

    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(log_level)

    settings.ensure_directories()
    init_db(settings.db_url)


@main.command()
@click.option("--spotify-playlist", "-s", help="Spotify playlist URL")
@click.option("--library-dir", "-l", type=click.Path(path_type=Path))
@click.option("--show", is_flag=True, help="Show current configuration")
@click.pass_context
def config(
    ctx: click.Context,
    spotify_playlist: Optional[str],
    library_dir: Optional[Path],
    show: bool,
) -> None:
    """Configure Dance settings."""
    settings: Settings = ctx.obj["settings"]

    if show:
        table = Table(title="Dance Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Spotify Playlist", settings.spotify_playlist_url or "[dim]Not set[/dim]")
        table.add_row("Library Directory", str(settings.library_dir))
        table.add_row("Stems Directory", str(settings.stems_dir))
        table.add_row("Data Directory", str(settings.data_dir))
        table.add_row(".als Output Directory", str(settings.als_output_dir))
        table.add_row("Skip Stems", str(settings.skip_stems))
        table.add_row("Skip Embeddings", str(settings.skip_embeddings))
        table.add_row("CLAP Model", settings.clap_model)
        table.add_row("Demucs Model", settings.demucs_model)
        console.print(table)
        return

    changes: list[tuple[str, str]] = []
    if spotify_playlist is not None:
        if "spotify.com/playlist" not in spotify_playlist:
            console.print("[red]Invalid Spotify playlist URL[/red]")
            sys.exit(1)
        changes.append(("DANCE_SPOTIFY_PLAYLIST_URL", spotify_playlist))
    if library_dir is not None:
        changes.append(("DANCE_LIBRARY_DIR", str(library_dir.expanduser().resolve())))

    if not changes:
        console.print(
            "No changes. Use --show to view current config, or pass "
            "--spotify-playlist / --library-dir to update ~/.dance/.env."
        )
        return

    # Persist by merging into ~/.dance/.env (preserve other keys verbatim).
    env_path = settings.data_dir / ".env"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    for k, v in changes:
        existing[k] = v
    env_path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")
    console.print(f"[green]Wrote {len(changes)} setting(s) to {env_path}[/green]")
    for k, v in changes:
        console.print(f"  {k} = {v}")


def _require_ffmpeg_or_exit() -> None:
    """Fail fast with an actionable message if ffmpeg is missing.

    ffmpeg is a hard native dependency for both ``sync`` (spotDL downloads)
    and ``process`` (demucs / librosa decoding). Check it before any work so
    a missing binary doesn't surface as an opaque mid-pipeline error.
    """
    from dance.pipeline.preflight import PreflightError, require_ffmpeg

    try:
        require_ffmpeg()
    except PreflightError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


def _run_sync(settings: Settings, dry_run: bool = False):
    """Core sync logic. Returns the DownloadResult.

    Shared by the ``sync`` and ``ingest`` commands. Assumes a playlist is
    configured and ffmpeg is present (callers check).
    """
    from dance.spotify.downloader import SpotifyDownloader

    console.print(f"[cyan]Syncing:[/cyan] {settings.spotify_playlist_url}")
    downloader = SpotifyDownloader(settings)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=None)
        result = downloader.sync_playlist(dry_run=dry_run)
        progress.update(task, completed=True)

    console.print(f"[green]Downloaded:[/green] {result.downloaded}")
    console.print(f"[yellow]Skipped:[/yellow] {result.skipped}")
    console.print(f"[red]Failed:[/red] {result.failed}")
    return result


@main.command()
@click.option("--dry-run", is_flag=True)
@click.pass_context
def sync(ctx: click.Context, dry_run: bool) -> None:
    """Sync tracks from the configured Spotify playlist."""
    settings: Settings = ctx.obj["settings"]

    if not settings.spotify_playlist_url:
        console.print("[red]Error:[/red] No Spotify playlist configured")
        sys.exit(1)

    _require_ffmpeg_or_exit()

    result = _run_sync(settings, dry_run=dry_run)

    if not dry_run and result.downloaded > 0:
        console.print(
            f"\n[bold]Next:[/bold] {result.downloaded} new track(s) downloaded — "
            "run [cyan]dance process[/cyan] (or [cyan]dance ingest[/cyan]) to make "
            "them playable."
        )


@main.command()
@click.option("--limit", "-n", type=int)
@click.option("--skip-stems", is_flag=True)
@click.option("--skip-embeddings", is_flag=True)
@click.option(
    "--skip-ingest",
    is_flag=True,
    help=(
        "Skip the library-scan ingest pass and process only tracks already "
        "in the DB. The API's /pipeline/process worker uses this when "
        "advancing optimistically-ingested Spotify tracks — those rows "
        "are pre-created, so a re-scan is just confirming what we already "
        "know. Use this when you've added tracks via the API and want a "
        "fast process-pending pass."
    ),
)
@click.option("--track-id", "-t", type=int)
@click.pass_context
def process(
    ctx: click.Context,
    limit: Optional[int],
    skip_stems: bool,
    skip_embeddings: bool,
    skip_ingest: bool,
    track_id: Optional[int],
) -> None:
    """Run the pipeline on pending tracks.

    Uses the parallel dispatcher: each stage gets its own worker pool (size
    from ``stage.concurrency``), stages run concurrently, and GPU-bound
    stages share a semaphore (``DANCE_GPU_CONCURRENCY``, default 1).
    """
    settings: Settings = ctx.obj["settings"]

    _require_ffmpeg_or_exit()

    _run_process(
        settings,
        limit=limit,
        skip_stems=skip_stems,
        skip_embeddings=skip_embeddings,
        skip_ingest=skip_ingest,
        track_id=track_id,
    )

    _print_remaining_work_hint(settings)


def _run_process(
    settings: Settings,
    *,
    limit: int | None = None,
    skip_stems: bool = False,
    skip_embeddings: bool = False,
    skip_ingest: bool = False,
    track_id: int | None = None,
) -> dict[str, dict[str, int]]:
    """Core process logic (scan + dispatch). Returns per-stage counts.

    Shared by the ``process`` and ``ingest`` commands. Callers handle the
    ffmpeg preflight and end-of-run hints.
    """
    from dance.core.database import get_session_factory
    from dance.pipeline.dispatcher import Dispatcher

    session_factory = get_session_factory(settings.db_url)
    dispatcher = Dispatcher(settings, session_factory=session_factory)

    if not skip_ingest:
        # Ingest first — scans library_dir, registers new audio files.
        console.print("[cyan]Scanning for new files...[/cyan]")
        dispatcher.ingest()
    else:
        console.print("[dim]Skipping library scan (--skip-ingest)[/dim]")

    # Run all enabled stages
    skip: set[str] = set()
    if skip_stems:
        skip.add("separate")
    if skip_embeddings:
        skip.add("embed")
    result = dispatcher.run(limit=limit, skip=skip, track_id=track_id)

    for stage_name, counts in result.items():
        console.print(f"[green]{stage_name}:[/green] {counts}")

    return result


def _print_remaining_work_hint(settings: Settings) -> None:
    """After a process pass, nudge the user if tracks are stuck short of
    COMPLETE (errored, or still mid-pipeline)."""
    session = get_session(settings.db_url)
    try:
        errored = (
            session.query(Track).filter(Track.state == TrackState.ERROR.value).count()
        )
        incomplete = (
            session.query(Track)
            .filter(
                Track.state.notin_(
                    [TrackState.COMPLETE.value, TrackState.ERROR.value]
                )
            )
            .count()
        )
    finally:
        session.close()

    if errored:
        console.print(
            f"\n[yellow]{errored} track(s) in ERROR[/yellow] — run "
            "[cyan]dance status[/cyan] for details (repeated native crashes are "
            "flagged there)."
        )
    if incomplete:
        console.print(
            f"[dim]{incomplete} track(s) still mid-pipeline — re-run "
            "[cyan]dance process[/cyan] to advance them.[/dim]"
        )


@main.command("list")
@click.option("--energy", "-e", type=click.IntRange(1, 10))
@click.option("--bpm-range", "-b")
@click.option("--key", "-k")
@click.option("--state", "-s")
@click.option("--limit", "-n", type=int, default=50)
@click.pass_context
def list_tracks(
    ctx: click.Context,
    energy: Optional[int],
    bpm_range: Optional[str],
    key: Optional[str],
    state: Optional[str],
    limit: int,
) -> None:
    """List tracks in the database."""
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)

    try:
        query = session.query(Track).outerjoin(Analysis, Analysis.track_id == Track.id)

        if energy:
            query = query.filter(Analysis.floor_energy == energy)
        if bpm_range:
            try:
                low, high = map(float, bpm_range.split("-"))
                query = query.filter(Analysis.bpm.between(low, high))
            except ValueError:
                console.print("[red]Invalid BPM range format. Use: 125-130[/red]")
                sys.exit(1)
        if key:
            query = query.filter(Analysis.key_camelot == key.upper())
        if state:
            query = query.filter(Track.state == state.lower())

        tracks = query.limit(limit).all()

        if not tracks:
            console.print("[yellow]No tracks found[/yellow]")
            return

        table = Table(title=f"Tracks ({len(tracks)} shown)")
        for col in ("ID", "Title", "Artist", "BPM", "Key", "Energy", "State"):
            table.add_column(col)

        for track in tracks:
            analysis = track.analysis
            bpm_str = f"{analysis.bpm:.1f}" if analysis and analysis.bpm else "-"
            key_str = analysis.key_camelot if analysis and analysis.key_camelot else "-"
            energy_str = f"E{analysis.floor_energy}" if analysis and analysis.floor_energy else "-"
            table.add_row(
                str(track.id),
                (track.title or track.file_name)[:30],
                (track.artist or "-")[:20],
                bpm_str,
                key_str,
                energy_str,
                track.state,
            )
        console.print(table)
    finally:
        session.close()


@main.command()
@click.option("--once", is_flag=True, help="Run once and exit")
@click.option("--skip-sync", is_flag=True)
@click.pass_context
def run(ctx: click.Context, once: bool, skip_sync: bool) -> None:
    """Run full pipeline (sync → process) once or in daemon mode."""
    settings: Settings = ctx.obj["settings"]

    def one_pass():
        if not skip_sync and settings.spotify_playlist_url:
            console.print("\n[bold cyan]Sync[/bold cyan]")
            ctx.invoke(sync)
        console.print("\n[bold cyan]Process[/bold cyan]")
        ctx.invoke(process)

    if once:
        one_pass()
        console.print("\n[green]Done.[/green]")
        return

    console.print(f"[bold]Daemon mode (every {settings.sync_interval_minutes}m). Ctrl+C to stop.[/bold]")
    try:
        while True:
            one_pass()
            time.sleep(settings.sync_interval_minutes * 60)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@main.command("build-graph")
@click.option("--track-id", "-t", type=int, multiple=True, help="Incremental: only rebuild edges for these tracks")
@click.pass_context
def build_graph(ctx: click.Context, track_id: tuple[int, ...]) -> None:
    """(Re)build the recommendation graph (track_edges).

    Run this after processing new tracks. Library-level operation, not part of
    the per-track stage pipeline.
    """
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)
    try:
        from dance.recommender import GraphBuilder

        builder = GraphBuilder(session, settings)
        ids = list(track_id) if track_id else None
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Building graph...", total=None)
            counts = builder.build(track_ids=ids)
            progress.update(task, completed=True)

        for kind, n in counts.items():
            console.print(f"[green]{kind}:[/green] {n} edges")
    finally:
        session.close()


@main.command()
@click.option("--apply", "apply_", is_flag=True, help="Actually write. Without this, dry run only.")
@click.option("--undo", is_flag=True, help="Clear every duplicate_of marker (full revert).")
@click.pass_context
def dedupe(ctx: click.Context, apply_: bool, undo: bool) -> None:
    """Mark redundant copies of the same recording (reversible).

    Three ingest runs pulled overlapping songs under different filenames, so
    file_hash differed and ingest's dedup never fired — 82 groups, 145
    redundant rows of 353. See docs/proposals/library-duplicates.md.

    This MARKS rather than deletes: session_plays and track_edges reference
    these rows and the audio stays on disk. ``--undo`` restores the previous
    state exactly.

    DRY RUN BY DEFAULT. Nothing is written without ``--apply``.
    """
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)
    try:
        from dance.core.database import Track
        from dance.recommender.dedup import find_duplicate_groups

        if undo:
            n = session.query(Track).filter(Track.duplicate_of.isnot(None)).count()
            if not apply_:
                console.print(f"[yellow]DRY RUN[/yellow] would clear {n} duplicate_of markers")
                console.print("Re-run with [bold]--undo --apply[/bold] to write.")
                return
            session.query(Track).filter(Track.duplicate_of.isnot(None)).update(
                {Track.duplicate_of: None}, synchronize_session=False
            )
            session.commit()
            console.print(f"[green]cleared[/green] {n} markers")
            return

        from dance.recommender.dedup_audio import compare_recordings

        tracks = session.query(Track).all()
        groups = find_duplicate_groups(tracks)
        candidates = [(c, d) for c, ds in groups for d in ds]

        # Metadata proposes, audio disposes. Title+duration alone would have
        # hidden 14 real tracks on this library (see dedup_audio). Every pair
        # is confirmed by listening before anything is marked.
        console.print(
            f"[bold]{len(groups)}[/bold] candidate groups, "
            f"[bold]{len(candidates)}[/bold] copies — verifying audio…"
        )
        verified: list[tuple] = []
        rejected: list[tuple] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Comparing…", total=len(candidates))
            for canon, dup in candidates:
                m = compare_recordings(
                    canon.file_path, dup.file_path,
                    float(canon.duration_seconds or 0), float(dup.duration_seconds or 0),
                )
                (verified if m.same_recording else rejected).append((canon, dup, m))
                progress.advance(task)

        redundant = len(verified)

        console.print(
            f"\n[green]{len(verified)}[/green] confirmed same recording, "
            f"[yellow]{len(rejected)}[/yellow] NOT confirmed (left visible), "
            f"of {len(tracks)} tracks"
        )
        for canon, d, m in verified[:10]:
            console.print(
                f"  mark [red]#{d.id}[/red] -> keep [green]#{canon.id}[/green] "
                f"sim={m.similarity:.3f}  {(canon.title or '')[:40]}"
            )
        if len(verified) > 10:
            console.print(f"  [dim]… and {len(verified) - 10} more[/dim]")
        if rejected:
            console.print("\n[yellow]Not confirmed — these stay visible:[/yellow]")
            for canon, d, m in rejected:
                why = m.error or f"sim={m.similarity:.3f}" + (
                    " (alignment hit search limit)" if m.at_search_limit else ""
                )
                console.print(
                    f"  keep both [dim]#{d.id} / #{canon.id}[/dim] {why}  "
                    f"{(canon.title or '')[:36]}"
                )

        if not apply_:
            console.print("\n[yellow]DRY RUN[/yellow] — nothing written.")
            console.print("Re-run with [bold]--apply[/bold] to mark. Reverse with [bold]--undo --apply[/bold].")
            return

        for canon, d, _m in verified:
            d.duplicate_of = canon.id
        session.commit()
        console.print(f"\n[green]marked[/green] {redundant} copies. Reverse: dance dedupe --undo --apply")
    finally:
        session.close()


@main.command()
@click.option("--track-id", "-t", type=int, help="Tag a single track")
@click.option("--limit", "-n", type=int, default=None, help="Max tracks to tag")
@click.option("--retag", is_flag=True, help="Re-tag tracks that already have tags from this mode")
@click.option(
    "--deep",
    is_flag=True,
    help="Use the Qwen2-Audio generative tagger (slow, ~10-30 s/track, ~8 GB weights) "
    "instead of the default CLAP zero-shot tagger.",
)
@click.pass_context
def tag(
    ctx: click.Context,
    track_id: Optional[int],
    limit: Optional[int],
    retag: bool,
    deep: bool,
) -> None:
    """Run the local tagger over COMPLETE tracks.

    Two modes:
      * default — CLAP zero-shot (fast, ~50 ms/track, controlled vocabulary)
      * --deep  — Qwen2-Audio (slow, generative, free-form dj_notes)

    Both run locally — no API keys, no cloud.
    """
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)

    try:
        from dance.core.database import TagSource, TrackTag
        from dance.llm import ClapZeroShotTagger, Qwen2AudioTagger

        if deep:
            if not settings.deep_tagger_enabled:
                console.print(
                    "[yellow]Deep tagger disabled. Set DANCE_DEEP_TAGGER_ENABLED=true.[/yellow]"
                )
                return
            tagger = Qwen2AudioTagger(settings)
            mode_label = f"Qwen2-Audio ({settings.deep_tagger_model})"
            src_value = TagSource.LLM.value
        else:
            if not settings.tagger_enabled:
                console.print("[yellow]Tagger disabled in settings[/yellow]")
                return
            tagger = ClapZeroShotTagger(settings)
            mode_label = f"CLAP zero-shot ({settings.clap_model})"
            src_value = TagSource.INFERRED.value

        q = session.query(Track).filter(Track.state == TrackState.COMPLETE.value)
        if track_id is not None:
            q = q.filter(Track.id == track_id)
        elif not retag:
            # Skip tracks that already have tags FROM THIS SOURCE.
            already = (
                session.query(TrackTag.track_id)
                .filter(TrackTag.source == src_value)
                .distinct()
                .subquery()
            )
            q = q.filter(Track.id.notin_(already))
        if limit is not None:
            q = q.limit(limit)
        targets = q.all()

        if not targets:
            console.print("[yellow]No tracks to tag[/yellow]")
            return

        console.print(f"[cyan]Tagging {len(targets)} track(s) via {mode_label}...[/cyan]")
        ok = 0
        errs = 0
        for t in targets:
            try:
                res = tagger.tag_track(session, t)
                tags = [v for _, v in res.all_tags()]
                console.print(
                    f"  [green]✓[/green] {t.title or t.file_name}: {', '.join(tags[:6])}"
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]✗[/red] {t.title or t.file_name}: {exc}")
                errs += 1
        console.print(f"\n[green]Tagged: {ok}[/green]  [red]Errors: {errs}[/red]")
    finally:
        session.close()


@main.command("export-als")
@click.argument("track_id", type=int, required=False)
@click.option("--all", "all_tracks", is_flag=True, help="Export every COMPLETE track")
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for a single-track export (must be inside als_output_dir).",
)
@click.pass_context
def export_als(
    ctx: click.Context,
    track_id: Optional[int],
    all_tracks: bool,
    out_path: Optional[Path],
) -> None:
    """Generate Ableton Live Set (.als) file(s).

    Either pass a TRACK_ID for a single export, or ``--all`` to export
    every COMPLETE track in the library. All output goes under
    ``settings.als_output_dir`` (default ``~/Music/Dance/Sets``).
    """
    settings: Settings = ctx.obj["settings"]

    if not (all_tracks or track_id is not None):
        console.print("[red]Pass either TRACK_ID or --all[/red]")
        sys.exit(1)
    if all_tracks and out_path is not None:
        console.print("[red]--out is only valid for single-track exports[/red]")
        sys.exit(1)

    from dance.als import AlsGenerator
    from dance.als.generator import AlsExportError

    session = get_session(settings.db_url)
    settings.ensure_directories()
    gen = AlsGenerator(session, settings)

    try:
        if all_tracks:
            tracks = (
                session.query(Track)
                .filter(Track.state == TrackState.COMPLETE.value)
                .all()
            )
            if not tracks:
                console.print("[yellow]No COMPLETE tracks to export[/yellow]")
                return
            console.print(f"[cyan]Exporting {len(tracks)} track(s)...[/cyan]")
            ok = 0
            errs = 0
            for t in tracks:
                try:
                    written = gen.write(t, None)
                    console.print(f"  [green]✓[/green] {written}")
                    ok += 1
                except AlsExportError as exc:
                    console.print(f"  [red]✗[/red] track {t.id}: {exc}")
                    errs += 1
            console.print(
                f"\n[green]Exported: {ok}[/green]  [red]Errors: {errs}[/red]"
            )
        else:
            track = session.get(Track, track_id)
            if track is None:
                console.print(f"[red]Track {track_id} not found[/red]")
                sys.exit(1)
            try:
                written = gen.write(track, out_path)
                console.print(f"[green]✓ Wrote[/green] {written}")
            except AlsExportError as exc:
                console.print(f"[red]✗ {exc}[/red]")
                sys.exit(1)
    finally:
        session.close()


@main.command()
@click.option("--skip-sync", is_flag=True, help="Skip the Spotify sync phase")
@click.pass_context
def ingest(ctx: click.Context, skip_sync: bool) -> None:
    """Full new-track lifecycle in one command.

    Runs, in order: sync → process → build-graph → tag → export-als --all.
    This is the one-shot equivalent of the five-command runbook chain. Use
    ``--skip-sync`` to re-ingest the existing library without hitting Spotify.
    """
    settings: Settings = ctx.obj["settings"]

    # Preflight once, up front — both sync and process need ffmpeg.
    _require_ffmpeg_or_exit()

    # 1) Sync ----------------------------------------------------------------
    downloaded = 0
    if skip_sync:
        console.print("[dim]Skipping Spotify sync (--skip-sync)[/dim]")
    elif not settings.spotify_playlist_url:
        console.print(
            "[yellow]No Spotify playlist configured — skipping sync.[/yellow] "
            "Set one with [cyan]dance config --spotify-playlist <url>[/cyan]."
        )
    else:
        console.print("\n[bold cyan]1/5 Sync[/bold cyan]")
        result = _run_sync(settings)
        downloaded = result.downloaded

    # 2) Process -------------------------------------------------------------
    console.print("\n[bold cyan]2/5 Process[/bold cyan]")
    _run_process(settings)

    # 3) Build graph ---------------------------------------------------------
    console.print("\n[bold cyan]3/5 Build graph[/bold cyan]")
    ctx.invoke(build_graph)

    # 4) Tag -----------------------------------------------------------------
    console.print("\n[bold cyan]4/5 Tag[/bold cyan]")
    ctx.invoke(tag)

    # 5) Export .als ---------------------------------------------------------
    console.print("\n[bold cyan]5/5 Export .als[/bold cyan]")
    ctx.invoke(export_als, all_tracks=True)

    # Final summary ----------------------------------------------------------
    session = get_session(settings.db_url)
    try:
        playable = (
            session.query(Track).filter(Track.state == TrackState.COMPLETE.value).count()
        )
        stuck = (
            session.query(Track)
            .filter(
                Track.state.notin_(
                    [TrackState.COMPLETE.value, TrackState.ERROR.value]
                )
            )
            .count()
        )
        errored = (
            session.query(Track).filter(Track.state == TrackState.ERROR.value).count()
        )
    finally:
        session.close()

    als_files = 0
    try:
        als_files = len(list(settings.als_output_dir.glob("*.als")))
    except OSError:
        pass

    console.print("\n[bold green]Ingest complete.[/bold green]")
    parts = [
        f"{playable} track(s) now playable",
        f"{als_files} .als file(s) in {settings.als_output_dir}",
    ]
    if downloaded:
        parts.insert(0, f"{downloaded} newly downloaded")
    console.print("  " + "; ".join(parts) + ".")
    if stuck or errored:
        blocked = stuck + errored
        console.print(
            f"  [yellow]{blocked} still stuck[/yellow] "
            f"({errored} errored, {stuck} mid-pipeline) — run "
            "[cyan]dance status[/cyan]."
        )


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pipeline state counts."""
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)

    try:
        table = Table(title="Pipeline Status")
        table.add_column("State", style="cyan")
        table.add_column("Count", justify="right")

        for state in TrackState:
            count = session.query(Track).filter(Track.state == state.value).count()
            table.add_row(state.value, str(count))
        table.add_row("[bold]total[/bold]", f"[bold]{session.query(Track).count()}[/bold]")
        console.print(table)

        # Surface repeatedly-crashing tracks so the heal loop is never silent.
        _print_stuck_track_detail(settings, session)
    finally:
        session.close()


def _print_stuck_track_detail(settings: Settings, session) -> None:
    """Show ERROR tracks and any tracks the heal ledger is still tracking
    (i.e. healed-from-inflight but not yet over the ERROR threshold), so a
    repeated native crash is visible instead of silent."""
    errored = (
        session.query(Track)
        .filter(Track.state == TrackState.ERROR.value)
        .order_by(Track.id)
        .all()
    )
    if errored:
        etable = Table(title="Tracks in ERROR")
        etable.add_column("ID", justify="right")
        etable.add_column("Title")
        etable.add_column("Error", overflow="fold")
        for t in errored:
            etable.add_row(
                str(t.id),
                (t.title or t.file_name or "-")[:30],
                (t.error_message or "-"),
            )
        console.print(etable)

    from dance.pipeline.heal_ledger import HealLedger

    tracked = HealLedger(settings.data_dir).stuck_tracks()
    if tracked:
        console.print(
            f"[yellow]{len(tracked)} track(s) have been healed from a crash and "
            "re-queued[/yellow] (will be flagged ERROR if they keep crashing):"
        )
        for tid, info in sorted(tracked.items()):
            console.print(
                f"  track {tid}: healed {info.get('count')}× from "
                f"{info.get('state')}"
            )


if __name__ == "__main__":
    main()
