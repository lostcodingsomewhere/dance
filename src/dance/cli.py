"""
Command-line interface for Dance DJ Pipeline.

Commands:
- config: Configure Spotify playlist and settings
- sync: Download new tracks from Spotify
- process: Analyze downloaded tracks
- export: Export to Traktor
- list: List tracks with filters
- run: Full pipeline (sync → process → export)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from dance import __version__
from dance.config import Settings, get_settings, reload_settings
from dance.core.database import (
    Analysis,
    Track,
    TrackState,
    init_db,
    get_session,
)

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with Rich handler."""
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
    """Dance - DJ Track Analysis Pipeline for House/Techno

    Automates cue points, energy tagging, and Traktor integration.
    """
    ctx.ensure_object(dict)
    settings = get_settings()
    ctx.obj["settings"] = settings

    # Setup logging
    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(log_level)

    # Ensure directories exist
    settings.ensure_directories()

    # Initialize database
    init_db(settings.db_url)


@main.command()
@click.option(
    "--spotify-playlist", "-s",
    help="Spotify playlist URL to sync from",
)
@click.option(
    "--library-dir", "-l",
    type=click.Path(path_type=Path),
    help="Directory for downloaded tracks",
)
@click.option(
    "--show", is_flag=True,
    help="Show current configuration",
)
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
        table.add_row("Traktor Collection", str(settings.traktor_collection_path) if settings.traktor_collection_path else "[dim]Not found[/dim]")
        table.add_row("Skip Stems", str(settings.skip_stems))
        table.add_row("Skip LLM", str(settings.skip_llm))
        table.add_row("LLM Model", settings.llm_model)
        table.add_row("LLM Device", settings.llm_device)
        table.add_row("LLM Quantization", settings.llm_quantize or "none")
        table.add_row("Audio Format", settings.audio_format)

        console.print(table)
        return

    # Update settings
    changed = False

    if spotify_playlist:
        # Validate URL format
        if "spotify.com/playlist" not in spotify_playlist:
            console.print("[red]Error:[/red] Invalid Spotify playlist URL")
            console.print("Expected format: https://open.spotify.com/playlist/xxxxx")
            sys.exit(1)

        settings.spotify_playlist_url = spotify_playlist
        changed = True
        console.print(f"[green]✓[/green] Spotify playlist set")

    if library_dir:
        settings.library_dir = library_dir.expanduser().resolve()
        settings.library_dir.mkdir(parents=True, exist_ok=True)
        changed = True
        console.print(f"[green]✓[/green] Library directory set to: {settings.library_dir}")

    if changed:
        # Save to .env file
        settings.save_to_env_file()
        console.print(f"\nConfiguration saved to: {settings.data_dir / '.env'}")
    else:
        console.print("No changes made. Use --show to view current config.")


@main.command()
@click.option("--dry-run", is_flag=True, help="Show what would be downloaded")
@click.pass_context
def sync(ctx: click.Context, dry_run: bool) -> None:
    """Sync tracks from Spotify playlist."""
    settings: Settings = ctx.obj["settings"]

    if not settings.spotify_playlist_url:
        console.print("[red]Error:[/red] No Spotify playlist configured")
        console.print("Run: dance config --spotify-playlist <url>")
        sys.exit(1)

    from dance.spotify.downloader import SpotifyDownloader

    console.print(f"[cyan]Syncing from:[/cyan] {settings.spotify_playlist_url}")

    downloader = SpotifyDownloader(settings)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading tracks...", total=None)

        result = downloader.sync_playlist(dry_run=dry_run)

        progress.update(task, completed=True)

    console.print()
    console.print(f"[green]Downloaded:[/green] {result.downloaded}")
    console.print(f"[yellow]Skipped:[/yellow] {result.skipped}")
    console.print(f"[red]Failed:[/red] {result.failed}")

    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors[:5]:  # Show first 5 errors
            console.print(f"  • {error}")


@main.command()
@click.option("--limit", "-n", type=int, help="Max tracks to process")
@click.option("--skip-stems", is_flag=True, help="Skip stem separation")
@click.option("--skip-llm", is_flag=True, help="Skip LLM augmentation")
@click.option("--track-id", "-t", type=int, help="Process specific track by ID")
@click.pass_context
def process(
    ctx: click.Context,
    limit: Optional[int],
    skip_stems: bool,
    skip_llm: bool,
    track_id: Optional[int],
) -> None:
    """Process downloaded tracks (analyze, separate stems, LLM augment, detect cues)."""
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)

    try:
        from dance.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(settings, session)

        # First, ingest any new files
        console.print("[cyan]Step 1: Scanning for new files...[/cyan]")
        ingest_result = orchestrator.ingest_new_files()

        if ingest_result["new"] > 0:
            console.print(f"[green]Found {ingest_result['new']} new tracks[/green]")

        # Process specific track or all pending
        if track_id:
            track = session.query(Track).filter_by(id=track_id).first()
            if not track:
                console.print(f"[red]Track not found: {track_id}[/red]")
                sys.exit(1)

            console.print(f"[cyan]Processing: {track.title or track.file_name}[/cyan]")
            if orchestrator.process_track(
                track,
                skip_stems=skip_stems or settings.skip_stems,
                skip_llm=skip_llm or settings.skip_llm,
            ):
                console.print("[green]Processing complete![/green]")
            else:
                console.print("[red]Processing failed[/red]")
            return

        # Get pending tracks count
        pending_count = session.query(Track).filter(
            Track.state == TrackState.PENDING.value
        ).count()

        if pending_count == 0:
            console.print("[yellow]No pending tracks to process[/yellow]")
            return

        # Process all pending tracks
        console.print(f"[cyan]Step 2: Processing {pending_count} tracks...[/cyan]")

        with Progress(console=console) as progress:
            task = progress.add_task("Processing...", total=None)

            result = orchestrator.process_pending(
                limit=limit,
                skip_stems=skip_stems or settings.skip_stems,
                skip_llm=skip_llm or settings.skip_llm,
            )

            progress.update(task, completed=True)

        console.print()
        console.print(f"[green]Analyzed:[/green] {result['analyzed']}")
        if not (skip_stems or settings.skip_stems):
            console.print(f"[green]Stems separated:[/green] {result['separated']}")
        if not (skip_llm or settings.skip_llm):
            console.print(f"[green]LLM augmented:[/green] {result['llm_augmented']}")
        console.print(f"[green]Cue points detected:[/green] {result['cues_detected']}")
        console.print(f"[red]Errors:[/red] {result['errors']}")

    finally:
        session.close()


@main.command()
@click.pass_context
def export(ctx: click.Context) -> None:
    """Export analyzed tracks to Traktor."""
    settings: Settings = ctx.obj["settings"]

    if not settings.traktor_collection_path:
        console.print("[red]Error:[/red] Traktor collection not found")
        console.print("Make sure Traktor is installed, or set DANCE_TRAKTOR_COLLECTION_PATH")
        sys.exit(1)

    from dance.export.traktor import TraktorExporter

    session = get_session(settings.db_url)

    try:
        exporter = TraktorExporter(settings)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting to Traktor...", total=None)
            result = exporter.export_all(session)
            progress.update(task, completed=True)

        console.print()
        console.print(f"[green]Exported:[/green] {result['exported']}")
        console.print(f"[red]Failed:[/red] {result['failed']}")
        console.print(f"\n[dim]Traktor collection: {settings.traktor_collection_path}[/dim]")

    finally:
        session.close()


@main.command("list")
@click.option("--energy", "-e", type=click.IntRange(1, 10), help="Filter by energy level")
@click.option("--bpm-range", "-b", help="BPM range (e.g., '125-130')")
@click.option("--key", "-k", help="Camelot key (e.g., '8A')")
@click.option("--state", "-s", help="Filter by state (pending, analyzed, complete, error)")
@click.option("--limit", "-n", type=int, default=50, help="Max tracks to show")
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
        query = session.query(Track).outerjoin(Analysis)

        # Apply filters
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

        query = query.limit(limit)
        tracks = query.all()

        if not tracks:
            console.print("[yellow]No tracks found[/yellow]")
            return

        table = Table(title=f"Tracks ({len(tracks)} shown)")
        table.add_column("ID", style="dim")
        table.add_column("Title", style="cyan", max_width=30)
        table.add_column("Artist", max_width=20)
        table.add_column("BPM", justify="right")
        table.add_column("Key")
        table.add_column("Energy", justify="center")
        table.add_column("State")

        for track in tracks:
            analysis = track.analysis

            bpm_str = f"{analysis.bpm:.1f}" if analysis and analysis.bpm else "-"
            key_str = analysis.key_camelot if analysis and analysis.key_camelot else "-"
            energy_str = f"E{analysis.floor_energy}" if analysis and analysis.floor_energy else "-"

            state_colors = {
                "pending": "yellow",
                "analyzing": "blue",
                "analyzed": "green",
                "complete": "green",
                "error": "red",
            }
            state_color = state_colors.get(track.state, "white")

            table.add_row(
                str(track.id),
                track.title or track.file_name[:30],
                track.artist or "-",
                bpm_str,
                key_str,
                energy_str,
                f"[{state_color}]{track.state}[/{state_color}]",
            )

        console.print(table)

    finally:
        session.close()


@main.command()
@click.option("--once", is_flag=True, help="Run once and exit (vs daemon mode)")
@click.option("--skip-sync", is_flag=True, help="Skip Spotify sync")
@click.pass_context
def run(ctx: click.Context, once: bool, skip_sync: bool) -> None:
    """Run full pipeline: sync → process → export.

    By default, runs continuously (daemon mode).
    Use --once for a single run.
    """
    import time

    settings: Settings = ctx.obj["settings"]

    def run_pipeline():
        """Execute one pipeline run."""
        # Sync from Spotify
        if not skip_sync and settings.spotify_playlist_url:
            console.print("\n[bold cyan]Step 1: Syncing from Spotify[/bold cyan]")
            ctx.invoke(sync)

        # Process tracks
        console.print("\n[bold cyan]Step 2: Processing tracks[/bold cyan]")
        ctx.invoke(process)

        # Export to Traktor
        if settings.traktor_collection_path:
            console.print("\n[bold cyan]Step 3: Exporting to Traktor[/bold cyan]")
            ctx.invoke(export)

    if once:
        run_pipeline()
        console.print("\n[green]Pipeline complete![/green]")
    else:
        console.print("[bold]Starting Dance daemon...[/bold]")
        console.print(f"Sync interval: {settings.sync_interval_minutes} minutes")
        console.print("Press Ctrl+C to stop\n")

        try:
            while True:
                run_pipeline()
                console.print(
                    f"\n[dim]Next sync in {settings.sync_interval_minutes} minutes...[/dim]"
                )
                time.sleep(settings.sync_interval_minutes * 60)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping daemon...[/yellow]")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pipeline status and statistics."""
    settings: Settings = ctx.obj["settings"]
    session = get_session(settings.db_url)

    try:
        # Count tracks by state
        total = session.query(Track).count()
        pending = session.query(Track).filter_by(state=TrackState.PENDING.value).count()
        analyzed = session.query(Track).filter_by(state=TrackState.ANALYZED.value).count()
        complete = session.query(Track).filter_by(state=TrackState.COMPLETE.value).count()
        errors = session.query(Track).filter_by(state=TrackState.ERROR.value).count()

        table = Table(title="Pipeline Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        table.add_row("Total tracks", str(total))
        table.add_row("Pending", f"[yellow]{pending}[/yellow]")
        table.add_row("Analyzed", f"[green]{analyzed}[/green]")
        table.add_row("Complete", f"[green]{complete}[/green]")
        table.add_row("Errors", f"[red]{errors}[/red]")

        console.print(table)

        # Energy distribution
        if analyzed + complete > 0:
            console.print("\n[bold]Energy Distribution:[/bold]")
            for e in range(1, 11):
                count = session.query(Analysis).filter_by(floor_energy=e).count()
                if count > 0:
                    bar = "█" * min(count, 20)
                    console.print(f"  E{e:2d}: {bar} ({count})")

    finally:
        session.close()


@main.command("llm-status")
@click.pass_context
def llm_status(ctx: click.Context) -> None:
    """Show LLM model status and GPU info."""
    settings: Settings = ctx.obj["settings"]

    table = Table(title="LLM Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Skip LLM", str(settings.skip_llm))
    table.add_row("Model", settings.llm_model)
    table.add_row("Device", settings.llm_device)
    table.add_row("Quantization", settings.llm_quantize or "none")
    table.add_row("Cache Responses", str(settings.llm_cache_responses))

    console.print(table)

    # Check if LLM is available
    console.print("\n[bold]LLM Availability:[/bold]")

    try:
        from dance.pipeline.stages.llm_augment import get_llm_status
        status = get_llm_status()

        if status.get("error"):
            console.print(f"[red]Error:[/red] {status['error']}")
        else:
            for key, value in status.items():
                if key not in ["error"]:
                    console.print(f"  {key}: {value}")

    except ImportError as e:
        console.print(f"[yellow]LLM dependencies not installed:[/yellow] {e}")
        console.print("Install with: pip install dance[llm]")
    except Exception as e:
        console.print(f"[red]Error checking LLM status:[/red] {e}")


@main.command("llm-analyze")
@click.option("--track-id", "-t", type=int, help="Analyze specific track by ID")
@click.option("--limit", "-n", type=int, help="Max tracks to analyze")
@click.option("--reanalyze", is_flag=True, help="Re-analyze already processed tracks")
@click.pass_context
def llm_analyze(
    ctx: click.Context,
    track_id: Optional[int],
    limit: Optional[int],
    reanalyze: bool,
) -> None:
    """Run LLM augmentation on tracks.

    By default, processes SEPARATED tracks that haven't been LLM-analyzed.
    Use --reanalyze to re-process already completed tracks.
    """
    settings: Settings = ctx.obj["settings"]

    if settings.skip_llm:
        console.print("[yellow]LLM is disabled in settings (DANCE_SKIP_LLM=true)[/yellow]")
        console.print("Set DANCE_SKIP_LLM=false to enable LLM analysis")
        return

    session = get_session(settings.db_url)

    try:
        from dance.pipeline.stages.llm_augment import (
            LLMAugmentStage,
            reanalyze_with_llm,
            is_llm_available,
        )

        if not is_llm_available():
            console.print("[red]LLM not available[/red]")
            console.print("Install dependencies: pip install dance[llm]")
            return

        if track_id:
            # Analyze specific track
            track = session.query(Track).filter_by(id=track_id).first()
            if not track:
                console.print(f"[red]Track not found: {track_id}[/red]")
                return

            console.print(f"[cyan]LLM analyzing: {track.title or track.file_name}[/cyan]")

            stage = LLMAugmentStage()
            if stage.augment_track(session, track):
                console.print("[green]LLM analysis complete![/green]")

                # Show results
                analysis = session.query(Analysis).filter_by(track_id=track.id).first()
                if analysis and analysis.llm_subgenre:
                    console.print(f"\n[bold]Results:[/bold]")
                    console.print(f"  Subgenre: {analysis.llm_subgenre}")
                    if analysis.llm_mood_tags:
                        import json
                        tags = json.loads(analysis.llm_mood_tags)
                        console.print(f"  Mood: {', '.join(tags)}")
                    if analysis.llm_dj_notes:
                        console.print(f"  DJ Notes: {analysis.llm_dj_notes}")
            else:
                console.print("[red]LLM analysis failed[/red]")

        elif reanalyze:
            # Re-analyze completed tracks
            console.print("[cyan]Re-analyzing completed tracks with LLM...[/cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing...", total=None)
                success, errors = reanalyze_with_llm(session, limit=limit)
                progress.update(task, completed=True)

            console.print(f"\n[green]Analyzed:[/green] {success}")
            console.print(f"[red]Errors:[/red] {errors}")

        else:
            # Analyze separated tracks
            separated_count = session.query(Track).filter(
                Track.state == TrackState.SEPARATED.value
            ).count()

            if separated_count == 0:
                console.print("[yellow]No tracks ready for LLM analysis[/yellow]")
                console.print("Run 'dance process' first to analyze and separate tracks")
                return

            console.print(f"[cyan]LLM analyzing {separated_count} tracks...[/cyan]")

            from dance.pipeline.stages.llm_augment import augment_separated_tracks

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing...", total=None)
                success, skipped, errors = augment_separated_tracks(session, limit=limit)
                progress.update(task, completed=True)

            console.print(f"\n[green]Analyzed:[/green] {success}")
            console.print(f"[yellow]Skipped:[/yellow] {skipped}")
            console.print(f"[red]Errors:[/red] {errors}")

    except ImportError as e:
        console.print(f"[red]LLM dependencies not installed:[/red] {e}")
        console.print("Install with: pip install dance[llm]")

    finally:
        session.close()


if __name__ == "__main__":
    main()
