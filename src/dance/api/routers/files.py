"""Local filesystem operations — reveal a track/stem in the OS file browser.

This is a single-user, single-machine app, so we can safely shell out to the
native file manager. Path validation: only paths inside ``library_dir`` or
``stems_dir`` are allowed, to prevent the endpoint from being used to open
arbitrary files via a stray HTTP request.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from dance.api.deps import get_settings
from dance.config import Settings
from dance.doctor import live_install_for_sets

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


def _is_allowed(path: Path, settings: Settings) -> bool:
    """True if ``path`` lives inside library_dir, stems_dir, or als_output_dir.

    Wave 4 added ``als_output_dir`` to the allowlist so the UI can reveal
    a freshly-exported Live Set after ``POST /tracks/{id}/als``.
    """
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    for root in (settings.library_dir, settings.stems_dir, settings.als_output_dir):
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _reveal_command(target: Path) -> list[str]:
    """Native command to show a file/folder in the OS file browser."""
    system = platform.system()
    if system == "Darwin":
        # `open -R` reveals the file in Finder, selecting it.
        return ["open", "-R", str(target)] if target.is_file() else ["open", str(target)]
    if system == "Windows":
        # `explorer /select,FILE` highlights the file.
        return ["explorer", f"/select,{target}"] if target.is_file() else ["explorer", str(target)]
    # Linux / *nix
    return ["xdg-open", str(target.parent if target.is_file() else target)]


@router.post("/reveal")
def reveal(
    body: dict,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Open the OS file manager at the given path.

    Body: ``{"path": "/absolute/path/to/file_or_dir"}``.
    Only paths inside ``library_dir``, ``stems_dir``, or ``als_output_dir``
    are accepted.
    """
    raw = body.get("path")
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="path required")

    target = Path(raw)
    if not target.exists():
        raise HTTPException(status_code=404, detail="path does not exist")
    if not _is_allowed(target, settings):
        raise HTTPException(
            status_code=403,
            detail="path must be inside library_dir, stems_dir, or als_output_dir",
        )

    cmd = _reveal_command(target)
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail=f"reveal command not available: {cmd[0]}"
        ) from exc
    logger.info("Revealed %s via %s", target, cmd[0])
    return {"ok": True, "command": cmd[0]}


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


@router.post("/open-in-live")
def open_in_live(
    body: dict,
    settings: Settings = Depends(get_settings),
) -> dict:
    raw = body.get("path")
    if not raw or not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="path required")
    target = Path(raw)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="path does not exist")
    if target.suffix != ".als" or not _is_inside(target, settings.als_output_dir):
        raise HTTPException(status_code=403, detail="only .als files inside als_output_dir open")
    install = live_install_for_sets() if platform.system() == "Darwin" else None
    if install is None:
        raise HTTPException(
            status_code=409,
            detail="No Ableton Live Standard or Suite that can open this Set was found. "
            "Run `dance doctor` to see what is installed.",
        )
    # Live Lite shares Standard's bundle id, so the app is named by its path.
    subprocess.Popen(
        ["open", "-a", str(install.app), str(target.resolve())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Opened %s in %s", target, install.app.name)
    return {"ok": True, "app": install.app.name.removesuffix(".app")}

