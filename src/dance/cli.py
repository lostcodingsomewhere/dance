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
        table.add_row("Skip Stems", str(settings.skip_stems))
        table.add_row("Skip Embeddings", str(settings.skip_embeddings))
        table.add_row("CLAP Model", settings.clap_model)
        table.add_row("Demucs Model", settings.demucs_model)
        console.print(table)
        return

    console.print("[yellow]Settings are configured via environment variables or ~/.dance/.env[/yellow]")
    console.print("Use --show to view current config.")


@main.command()
@click.option("--dry-run", is_flag=True)
@click.pass_context
def sync(ctx: click.Context, dry_run: bool) -> None:
    """Sync tracks from the configured Spotify playlist."""
    settings: Settings = ctx.obj["settings"]

    if not settings.spotify_playlist_url:
        console.print("[red]Error:[/red] No Spotify playlist configured")
        sys.exit(1)

    from dance.spotify.downloader import SpotifyDownloader

    console.print(f"[cyan]Syncing:[/cyan] {settings.spotify_playlist_url}")
    downloader = SpotifyDownloader(settings)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Downloading...", total=None)
        result = downloader.sync_playlist(dry_run=dry_run)
        progress.update(task, completed=True)

    console.print(f"[green]Downloaded:[/green] {result.downloaded}")
    console.print(f"[yellow]Skipped:[/yellow] {result.skipped}")
    console.print(f"[red]Failed:[/red] {result.failed}")


@main.command()
@click.option("--limit", "-n", type=int)
@click.option("--skip-stems", is_flag=True)
@click.option("--skip-embeddings", is_flag=True)
@click.option("--track-id", "-t", type=int)
@click.pass_context
def process(
    ctx: click.Context,
    limit: Optional[int],
    skip_stems: bool,
    skip_embeddings: bool,
    track_id: Optional[int],
) -> None:
    """Run the pipeline on pending tracks."""
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)

    try:
        from dance.pipeline.dispatcher import Dispatcher

        dispatcher = Dispatcher(settings, session)

        # Ingest first
        console.print("[cyan]Scanning for new files...[/cyan]")
        dispatcher.ingest()

        # Run all enabled stages
        skip: set[str] = set()
        if skip_stems:
            skip.add("separate")
        if skip_embeddings:
            skip.add("embed")
        result = dispatcher.run(limit=limit, skip=skip, track_id=track_id)

        for stage_name, counts in result.items():
            console.print(f"[green]{stage_name}:[/green] {counts}")
    finally:
        session.close()


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
    finally:
        session.close()


if __name__ == "__main__":
    main()
