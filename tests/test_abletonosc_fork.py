"""Tests for the vendored AbletonOSC patch series and the patch engine."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from dance import doctor
from dance.osc import fork


def _added_lines(patch: Path) -> str:
    return "\n".join(
        line[1:]
        for line in patch.read_text(encoding="utf-8").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_upstream_is_pinned_to_a_full_commit():
    up = fork.read_upstream()
    assert up["url"].endswith("AbletonOSC.git")
    assert re.fullmatch(r"[0-9a-f]{40}", up["commit"])


def test_series_registers_every_address_doctor_expects():
    added = "\n".join(_added_lines(p) for p in fork.patch_files())
    literal = set(re.findall(r'add_handler\(\s*"([^"]+)"', added))
    # create_audio_clip is registered via upstream's method list, not a literal.
    assert '"create_audio_clip"' in added
    for p in doctor.KNOWN_PATCHES:
        if p.key == "create_audio_clip":
            continue
        assert p.addresses <= literal, (p.key, sorted(p.addresses - literal))


def test_log_fix_patch_replaces_the_buggy_call():
    patch = next(p for p in fork.patch_files() if "logging" in p.name)
    text = patch.read_text(encoding="utf-8")
    removed = [ln for ln in text.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    assert any(doctor._LOG_BUG_RE.search(ln) for ln in removed)
    assert not doctor._LOG_BUG_RE.search(_added_lines(patch))


def test_doctor_points_at_files_that_exist():
    root = fork.REPO_ROOT
    for p in doctor.KNOWN_PATCHES:
        assert (root / p.where).is_file(), p.where
    assert (root / doctor._LOG_FIX_DOC).is_file()


# ---------------------------------------------------------------------------
# Patch engine on a throwaway git repo
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def install(tmp_path) -> Path:
    root = tmp_path / "AbletonOSC"
    (root / "abletonosc").mkdir(parents=True)
    (root / "abletonosc" / "clip.py").write_text("a = 1\nb = 2\nc = 3\n")
    _git(root, "init", "-q")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    return root


@pytest.fixture
def patch(tmp_path, install) -> Path:
    clip = install / "abletonosc" / "clip.py"
    clip.write_text("a = 1\nb = 2\nadded = True\nc = 3\n")
    diff = subprocess.run(
        ["git", "diff"], cwd=install, check=True, capture_output=True, text=True
    ).stdout
    _git(install, "checkout", "--", ".")
    p = tmp_path / "0001-add.patch"
    p.write_text(diff)
    return p


def test_pending_then_applied_then_idempotent(tmp_path, install, patch):
    backups = tmp_path / "backups"
    assert [s.status for s in fork.classify(install, [patch])] == [fork.PENDING]

    statuses, backup = fork.apply_patches(
        install, [patch], backups, now=datetime(2026, 10, 13, 9, 0, 0)
    )

    assert [s.status for s in statuses] == [fork.PENDING]
    assert backup == backups / "abletonosc-20261013-090000"
    assert "added" not in (backup / "abletonosc" / "clip.py").read_text()  # pre-patch copy
    assert "added = True" in (install / "abletonosc" / "clip.py").read_text()
    assert [s.status for s in fork.classify(install, [patch])] == [fork.APPLIED]

    statuses, backup = fork.apply_patches(install, [patch], backups)
    assert backup is None
    assert [s.status for s in statuses] == [fork.APPLIED]


def test_conflict_changes_nothing(tmp_path, install, patch):
    clip = install / "abletonosc" / "clip.py"
    clip.write_text("something else entirely\n")

    with pytest.raises(fork.PatchConflictError):
        fork.apply_patches(install, [patch], tmp_path / "backups")

    assert clip.read_text() == "something else entirely\n"
    assert not (tmp_path / "backups").exists()
