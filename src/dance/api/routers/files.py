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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


def _is_allowed(path: Path, settings: Settings) -> bool:
    """True if ``path`` lives inside library_dir or stems_dir."""
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    for root in (settings.library_dir, settings.stems_dir):
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
    Only paths inside ``library_dir`` or ``stems_dir`` are accepted.
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
            detail="path must be inside library_dir or stems_dir",
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
