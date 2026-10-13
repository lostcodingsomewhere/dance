"""Apply dance's AbletonOSC patch series to an installed AbletonOSC clone.

The patches live in ``vendor/abletonosc/patches`` (pinned to the upstream
commit in ``vendor/abletonosc/UPSTREAM``). Each one is classified before
anything is touched:

- ``applied``  — ``git apply --reverse --check`` succeeds,
- ``pending``  — ``git apply --check`` succeeds,
- ``conflict`` — neither; the install has diverged from what the patch expects.

``apply_patches`` refuses to change anything if any patch conflicts, backs the
script's ``abletonosc/`` package up, then applies the pending patches in order
as working-tree changes (never commits in the user's clone).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = REPO_ROOT / "vendor" / "abletonosc"
DEFAULT_TARGET = (
    Path.home() / "Music" / "Ableton" / "User Library" / "Remote Scripts" / "AbletonOSC"
)

APPLIED = "applied"
PENDING = "pending"
CONFLICT = "conflict"


@dataclass(frozen=True)
class PatchStatus:
    patch: Path
    status: str  # APPLIED | PENDING | CONFLICT
    detail: str = ""


def read_upstream(vendor_dir: Path = VENDOR_DIR) -> dict[str, str]:
    """``UPSTREAM`` as a dict (``url``, ``commit``)."""
    out: dict[str, str] = {}
    for line in (vendor_dir / "UPSTREAM").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def patch_files(vendor_dir: Path = VENDOR_DIR) -> list[Path]:
    return sorted((vendor_dir / "patches").glob("*.patch"))


def _git(target: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(target), *args], capture_output=True, text=True, check=False
    )


def head_commit(target: Path) -> str | None:
    proc = _git(target, "rev-parse", "HEAD")
    return proc.stdout.strip() if proc.returncode == 0 else None


def classify(target: Path, patches: list[Path]) -> list[PatchStatus]:
    """Status of each patch against ``target`` as it is now.

    Patches are checked in order against the current tree, so a pending patch
    that a later one depends on makes the later one look like a conflict; the
    series here touches disjoint hunks, which keeps the check independent.
    """
    out: list[PatchStatus] = []
    for p in patches:
        if _git(target, "apply", "--reverse", "--check", str(p)).returncode == 0:
            out.append(PatchStatus(p, APPLIED))
            continue
        fwd = _git(target, "apply", "--check", str(p))
        if fwd.returncode == 0:
            out.append(PatchStatus(p, PENDING))
        else:
            out.append(PatchStatus(p, CONFLICT, (fwd.stderr or fwd.stdout).strip()))
    return out


class PatchConflictError(RuntimeError):
    pass


def apply_patches(
    target: Path, patches: list[Path], backup_root: Path, *, now: datetime | None = None
) -> tuple[list[PatchStatus], Path | None]:
    """Apply every pending patch; return (statuses before applying, backup dir).

    Raises ``PatchConflictError`` (changing nothing) if any patch conflicts.
    Returns ``(statuses, None)`` when nothing was pending.
    """
    statuses = classify(target, patches)
    conflicts = [s for s in statuses if s.status == CONFLICT]
    if conflicts:
        raise PatchConflictError(
            "; ".join(f"{s.patch.name}: {s.detail or 'does not apply'}" for s in conflicts)
        )
    pending = [s for s in statuses if s.status == PENDING]
    if not pending:
        return statuses, None

    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    backup = backup_root / f"abletonosc-{stamp}"
    shutil.copytree(target / "abletonosc", backup / "abletonosc")

    for s in pending:
        proc = _git(target, "apply", str(s.patch))
        if proc.returncode != 0:  # pragma: no cover - --check passed moments ago
            raise PatchConflictError(f"{s.patch.name}: {proc.stderr.strip()}")
    return statuses, backup
