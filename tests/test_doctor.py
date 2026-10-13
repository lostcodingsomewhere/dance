"""Tests for ``dance doctor`` (src/dance/doctor.py).

Everything here runs against temp fixtures — fake ``Ableton Live*.app``
bundles, a fake AbletonOSC tree, temp SQLite DBs at chosen Alembic
revisions, fake model caches, a throwaway HTTP server. Nothing reads the
real ``/Applications``, ``~/Music`` or ``~/.dance``: every root is passed in
explicitly or monkeypatched (and conftest's autouse fixture keeps Settings
in a temp home).
"""

from __future__ import annotations

import json
import plistlib
import shutil
import socket
import sqlite3
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from click.testing import CliRunner
from sqlalchemy import create_engine

from dance import doctor
from dance.doctor import FAIL, INFO, OK, WARN, TemplateInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
SCRIPT_LOCATION = REPO_ROOT / "src" / "dance" / "alembic"


# ---------------------------------------------------------------------------
# DB migration state
# ---------------------------------------------------------------------------


def _upgrade(db_path: Path, rev: str = "head") -> str:
    url = f"sqlite:///{db_path}"
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(SCRIPT_LOCATION))
    cfg.cmd_opts = type("X", (), {"x": [f"url={url}"]})()  # type: ignore[attr-defined]
    command.upgrade(cfg, rev)
    return url


def _version(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as c:
        return [r[0] for r in c.execute("SELECT version_num FROM alembic_version")]


def _head() -> str:
    return ScriptDirectory(str(SCRIPT_LOCATION)).get_current_head()


def test_db_at_head_is_ok_and_untouched(tmp_path: Path):
    db = tmp_path / "dance.db"
    url = _upgrade(db)
    before = db.read_bytes()
    mtime = db.stat().st_mtime_ns

    res = doctor.check_db_migrations(url)

    assert res.status == OK, res
    assert _head() in res.summary
    # Strictly read-only: same bytes, same mtime, no journal side files.
    assert db.read_bytes() == before
    assert db.stat().st_mtime_ns == mtime
    assert not Path(str(db) + "-wal").exists()
    assert not Path(str(db) + "-shm").exists()
    assert not Path(str(db) + "-journal").exists()


def test_db_behind_head_fails_lists_pending_and_never_upgrades(tmp_path: Path):
    db = tmp_path / "dance.db"
    old = "c6e3ba31c1f0"  # add sets and set_tracks tables
    url = _upgrade(db, old)
    script = ScriptDirectory(str(SCRIPT_LOCATION))
    expected = [r.revision for r in script.iterate_revisions(_head(), old)]

    res = doctor.check_db_migrations(url)

    assert res.status == FAIL
    assert f"{len(expected)} migration(s) behind" in res.summary
    assert old in res.summary
    pending = [d for d in res.details if d.startswith("pending: ")]
    assert [p.split()[1] for p in pending] == list(reversed(expected))  # oldest first
    assert "alembic upgrade head" in (res.fix or "")
    assert _version(db) == [old]  # the doctor did not migrate anything


def test_db_unknown_revision_fails(tmp_path: Path):
    db = tmp_path / "dance.db"
    url = _upgrade(db)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE alembic_version SET version_num = 'deadbeef0000'")

    res = doctor.check_db_migrations(url)

    assert res.status == FAIL
    assert "deadbeef0000" in res.summary
    assert "doesn't know" in res.summary


def test_db_unversioned_but_matching_schema_suggests_stamp(tmp_path: Path):
    from dance.core import database as dbmod

    db = tmp_path / "dance.db"
    eng = create_engine(f"sqlite:///{db}")
    dbmod.Base.metadata.create_all(eng)
    dbmod._create_partial_unique_indexes(eng)
    eng.dispose()

    res = doctor.check_db_migrations(f"sqlite:///{db}")

    assert res.status == WARN
    assert "unversioned" in res.summary
    assert "schema matches" in res.summary
    assert "alembic stamp" in (res.fix or "")


def test_db_unversioned_with_drift_fails(tmp_path: Path):
    db = tmp_path / "dance.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE tracks (id INTEGER PRIMARY KEY)")

    res = doctor.check_db_migrations(f"sqlite:///{db}")

    assert res.status == FAIL
    assert "differs from the models" in res.summary
    assert any(d.startswith("drift: ") for d in res.details)


def test_db_missing_file_warns_and_is_not_created(tmp_path: Path):
    db = tmp_path / "nope" / "dance.db"

    res = doctor.check_db_migrations(f"sqlite:///{db}")

    assert res.status == WARN
    assert "no database yet" in res.summary
    assert not db.exists()
    assert not db.parent.exists()


def test_db_at_head_missing_partial_index_warns(tmp_path: Path):
    db = tmp_path / "dance.db"
    url = _upgrade(db)
    with sqlite3.connect(db) as c:
        c.execute("DROP INDEX uq_sets_one_active")

    res = doctor.check_db_migrations(url)

    assert res.status == WARN
    assert any("uq_sets_one_active" in d for d in res.details)


def test_db_multiple_script_heads_fail(tmp_path: Path):
    scripts = tmp_path / "alembic"
    shutil.copytree(SCRIPT_LOCATION, scripts, ignore=shutil.ignore_patterns("__pycache__"))
    (scripts / "versions" / "ffff00000000_branch.py").write_text(
        '"""a parallel branch"""\n'
        "revision = 'ffff00000000'\n"
        "down_revision = 'c6e3ba31c1f0'\n"
        "branch_labels = None\ndepends_on = None\n"
        "def upgrade():\n    pass\n"
        "def downgrade():\n    pass\n"
    )
    url = _upgrade(tmp_path / "dance.db")

    res = doctor.check_db_migrations(url, script_location=scripts)

    assert res.status == FAIL
    assert "2 heads" in res.summary
    assert "alembic merge" in (res.fix or "")


# ---------------------------------------------------------------------------
# Ableton Live installs
# ---------------------------------------------------------------------------

TPL = TemplateInfo(
    creator="Ableton Live 12.4",
    version="12.4",
    revision="d85a94ab5eb1299391a7ff9400bdb8c0dcd7cc75",
    minor_version="12.0_12402",
    sha256="0" * 64,
)


def _fake_live(apps: Path, name: str, version: str, bundle_id: str = "com.ableton.live") -> Path:
    app = apps / f"{name}.app"
    (app / "Contents").mkdir(parents=True)
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump(
            {
                "CFBundleShortVersionString": version,
                "CFBundleVersion": version,
                "CFBundleIdentifier": bundle_id,
                "CFBundleName": "Live",
            },
            fh,
        )
    return app


def test_live_single_matching_standard_is_ok(tmp_path: Path):
    _fake_live(tmp_path, "Ableton Live 12 Standard", "12.4 (2026-04-24_d85a94ab5e)")

    res = doctor.check_live_installs([tmp_path], template=TPL)

    assert res.status == OK, res
    assert "matches" in res.summary
    assert any("exact build the template was saved from" in d for d in res.details)


def test_live_lite_plus_standard_warns_and_recommends_open_a(tmp_path: Path):
    _fake_live(tmp_path, "Ableton Live 12 Lite", "12.0.2 (2024-04-09_491da1ed88)")
    _fake_live(tmp_path, "Ableton Live 12 Standard", "12.4 (2026-04-24_d85a94ab5e)")

    res = doctor.check_live_installs([tmp_path], template=TPL)

    assert res.status == WARN
    assert "share com.ableton.live" in res.summary
    assert "Lite 12.0.2" in res.summary
    assert res.fix == 'open -a "Ableton Live 12 Standard" <file>.als'
    lite = next(d for d in res.details if d.startswith("Ableton Live 12 Lite.app"))
    assert "track-capped" in lite and "older than the 12.4 template" in lite


def test_live_only_lite_fails(tmp_path: Path):
    _fake_live(tmp_path, "Ableton Live 12 Lite", "12.0.2 (2024-04-09_491da1ed88)")

    res = doctor.check_live_installs([tmp_path], template=TPL)

    assert res.status == FAIL
    assert "Standard or Suite" in (res.fix or "")


def test_live_only_older_standard_fails(tmp_path: Path):
    _fake_live(tmp_path, "Ableton Live 12 Standard", "12.1.5 (2024-12-01_abcdef1234)")

    res = doctor.check_live_installs([tmp_path], template=TPL)

    assert res.status == FAIL


def test_live_newer_suite_warns_about_template(tmp_path: Path):
    _fake_live(tmp_path, "Ableton Live 12 Suite", "12.5 (2026-09-01_0123456789)")

    res = doctor.check_live_installs([tmp_path], template=TPL)

    assert res.status == WARN
    assert "newer than the 12.4 template" in res.summary
    assert "live-compat.md" in (res.fix or "")


def test_live_none_found(tmp_path: Path):
    res = doctor.check_live_installs([tmp_path], template=TPL)
    assert res.status == FAIL

    res = doctor.check_live_installs([tmp_path / "does-not-exist"], template=TPL)
    assert res.status == INFO


def test_live_unreadable_plist_does_not_crash(tmp_path: Path):
    app = tmp_path / "Ableton Live 12 Suite.app" / "Contents"
    app.mkdir(parents=True)
    (app / "Info.plist").write_bytes(b"not a plist")
    _fake_live(tmp_path, "Ableton Live 12 Standard", "12.4 (2026-04-24_d85a94ab5e)")

    res = doctor.check_live_installs([tmp_path], template=TPL)

    assert res.status == OK
    assert any("unreadable Info.plist" in d for d in res.details)


def test_template_info_matches_writer_constants():
    from importlib import resources

    from dance.als.writer import ABLETON_ROOT_ATTRS

    info = doctor.read_template_info()
    data = (resources.files("dance.als") / "templates" / "blank_live12.xml").read_bytes()
    import hashlib

    assert info.creator == ABLETON_ROOT_ATTRS["Creator"]
    assert info.revision == ABLETON_ROOT_ATTRS["Revision"]
    assert info.minor_version == ABLETON_ROOT_ATTRS["MinorVersion"]
    assert info.version == info.creator.rsplit(" ", 1)[-1]
    assert info.sha256 == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# AbletonOSC handler scan
# ---------------------------------------------------------------------------


def _all_sent() -> set[str]:
    return {a for addrs in doctor.dance_sent_addresses().values() for a in addrs}


def _patch_addrs(key: str) -> set[str]:
    return set(next(p for p in doctor.KNOWN_PATCHES if p.key == key).addresses)


def _fake_abletonosc(
    root: Path,
    *,
    drop: set[str] = frozenset(),  # type: ignore[assignment]
    log_bug: bool = False,
    git: str | None = "loose",
) -> Path:
    """A fake AbletonOSC checkout that registers every address dance sends
    except ``drop`` — clip_slot methods via the upstream ``methods`` loop,
    everything else as literal ``add_handler`` calls, show_message in
    manager.py like upstream."""
    pkg = root / "abletonosc"
    pkg.mkdir(parents=True)
    want = _all_sent() - set(drop)
    slot_methods = sorted(
        a.rsplit("/", 1)[1] for a in want if a.startswith("/live/clip_slot/") and a.count("/") == 3
    )
    log_line = (
        "self.logger.info(track_index, clip_index, rv)"
        if log_bug
        else 'self.logger.info("clip_slot %s %s -> %s", track_index, clip_index, rv)'
    )
    (pkg / "clip_slot.py").write_text(
        "class ClipSlotHandler:\n"
        "    def init_api(self):\n"
        "        def cb(params):\n"
        "            track_index, clip_index, rv = 0, 0, None\n"
        f"            {log_line}\n"
        f"        methods = {slot_methods!r}\n"
        "        for method in methods:\n"
        '            self.osc_server.add_handler("/live/clip_slot/%s" % method, cb)\n'
    )
    rest = sorted(
        a
        for a in want
        if not (a.startswith("/live/clip_slot/") and a.count("/") == 3)
        and a != "/live/api/show_message"
    )
    (pkg / "misc.py").write_text(
        "def init_api(self):\n"
        + "".join(f"    self.osc_server.add_handler({a!r}, None)\n" for a in rest)
    )
    if "/live/api/show_message" in want:
        (root / "manager.py").write_text(
            'self.osc_server.add_handler("/live/api/show_message", None)\n'
        )
    if git:
        g = root / ".git"
        g.mkdir()
        (g / "config").write_text(
            '[remote "origin"]\n\turl = https://github.com/ideoforms/AbletonOSC.git\n'
        )
        sha = "0ca68214bd62c9b5cb641ca34006cfd70ba94430"
        if git == "detached":
            (g / "HEAD").write_text(sha + "\n")
        else:
            (g / "HEAD").write_text("ref: refs/heads/master\n")
            if git == "packed":
                (g / "packed-refs").write_text(f"# pack-refs\n{sha} refs/heads/master\n")
            else:
                (g / "refs" / "heads").mkdir(parents=True)
                (g / "refs" / "heads" / "master").write_text(sha + "\n")
    return root


def test_abletonosc_fully_patched_is_ok(tmp_path: Path):
    root = _fake_abletonosc(tmp_path / "AbletonOSC")

    res = doctor.check_abletonosc(root)

    assert res.status == OK, res.details
    assert f"all {len(_all_sent())} addresses" in res.summary
    assert "upstream 0ca68214bd" in res.summary
    assert any("ideoforms/AbletonOSC" in d and "refs/heads/master" in d for d in res.details)
    present = next(d for d in res.details if d.startswith("patches present: "))
    for p in doctor.KNOWN_PATCHES:
        assert p.key in present
    assert "clip_slot_log_fix" in present


def test_abletonosc_missing_crossfade_assign_fails_with_doc_pointer(tmp_path: Path):
    root = _fake_abletonosc(tmp_path / "AbletonOSC", drop=_patch_addrs("crossfade_assign"))

    res = doctor.check_abletonosc(root)

    assert res.status == FAIL
    assert "2 of" in res.summary and "no handler" in res.summary
    line = next(d for d in res.details if d.startswith("MISSING crossfade_assign"))
    assert "/live/track/set/crossfade_assign" in line
    assert "osc/bridge.py" in line  # the live path sends it, hence FAIL
    assert "0003-track-get-set-crossfade_assign" in (res.fix or "")
    assert "fully quit and reopen Live" in (res.fix or "")


def test_abletonosc_missing_probe_only_handlers_warns(tmp_path: Path):
    root = _fake_abletonosc(tmp_path / "AbletonOSC", drop=_patch_addrs("warp_markers"))

    res = doctor.check_abletonosc(root)

    assert res.status == WARN
    line = next(d for d in res.details if d.startswith("MISSING warp_markers"))
    assert "only standalone probes" in line
    assert "0005-clip-read-and-edit-warp-markers" in (res.fix or "")


def test_abletonosc_logging_bug_warns(tmp_path: Path):
    root = _fake_abletonosc(tmp_path / "AbletonOSC", log_bug=True)

    res = doctor.check_abletonosc(root)

    assert res.status == WARN
    assert any("logging bug" in d for d in res.details)
    assert "0002-clip_slot-fix-the-logging-call" in (res.fix or "")
    assert "scripts/abletonosc_patch.py" in (res.fix or "")


def test_abletonosc_undocumented_gap_on_live_path_fails(tmp_path: Path):
    root = _fake_abletonosc(tmp_path / "AbletonOSC", drop={"/live/song/set/tempo"})

    res = doctor.check_abletonosc(root)

    assert res.status == FAIL
    line = next(d for d in res.details if "/live/song/set/tempo" in d)
    assert "no documented patch" in line


def test_abletonosc_not_installed_fails(tmp_path: Path):
    res = doctor.check_abletonosc(tmp_path / "AbletonOSC")
    assert res.status == FAIL
    assert "Install steps" in (res.fix or "")


@pytest.mark.parametrize("layout", ["packed", "detached"])
def test_abletonosc_reads_git_head_without_git(tmp_path: Path, layout: str):
    root = _fake_abletonosc(tmp_path / "AbletonOSC", git=layout)

    res = doctor.check_abletonosc(root)

    assert "upstream 0ca68214bd" in res.summary
    if layout == "detached":
        assert any("(detached)" in d for d in res.details)


def test_abletonosc_without_git_reports_unknown_commit(tmp_path: Path):
    root = _fake_abletonosc(tmp_path / "AbletonOSC", git=None)

    res = doctor.check_abletonosc(root)

    assert res.status == OK
    assert any("upstream commit unknown" in d for d in res.details)


def test_handler_scanner_understands_upstream_patterns(tmp_path: Path):
    pkg = tmp_path / "abletonosc"
    pkg.mkdir()
    (pkg / "track.py").write_text(
        "class TrackHandler:\n"
        "    def init_api(self):\n"
        "        methods = ['stop_all_clips']\n"
        "        properties_r = ['playing_slot_index']\n"
        "        properties_rw = ['solo']\n"
        "        properties_rw += ['name']\n"
        "        for method in methods:\n"
        "            self.osc_server.add_handler('/live/track/%s' % method, cb)\n"
        "        for prop in properties_r + properties_rw:\n"
        "            self.osc_server.add_handler('/live/track/get/%s' % prop, cb)\n"
        "            self.osc_server.add_handler('/live/track/start_listen/%s' % prop, cb)\n"
        "        for prop in properties_rw:\n"
        "            self.osc_server.add_handler(f'/live/track/set/{prop}', cb)\n"
        "        mixer_properties_rw = ['volume']\n"
        "        for prop in mixer_properties_rw:\n"
        "            self.osc_server.add_handler('/live/track/get/' + prop, cb)\n"
        "        self.osc_server.add_handler('/live/track/get/crossfade_assign', cb)\n"
        "        self.osc_server.add_handler(make_address(), cb)\n"
    )
    (pkg / "song.py").write_text(
        "def init_api(self):\n"
        "    for method in ['start_playing', 'stop_playing']:\n"
        "        self.osc_server.add_handler('/live/song/%s' % method, cb)\n"
    )
    (pkg / "broken.py").write_text("def nope(:\n")

    addrs, unresolved, errors = doctor.scan_abletonosc_handlers(tmp_path)

    assert addrs == {
        "/live/track/stop_all_clips",
        "/live/track/get/playing_slot_index",
        "/live/track/start_listen/playing_slot_index",
        "/live/track/get/solo",
        "/live/track/start_listen/solo",
        "/live/track/get/name",
        "/live/track/start_listen/name",
        "/live/track/set/solo",
        "/live/track/set/name",
        "/live/track/get/volume",
        "/live/track/get/crossfade_assign",
        "/live/song/start_playing",
        "/live/song/stop_playing",
    }
    assert unresolved == 1
    assert len(errors) == 1 and errors[0].startswith("broken.py")


def test_known_patches_are_really_sent_by_dance():
    """Cross-check the patch table against client.py so it can't rot."""
    sent = _all_sent()
    for p in doctor.KNOWN_PATCHES:
        assert p.addresses <= sent, f"{p.key}: {sorted(p.addresses - sent)} not sent by client.py"
    # And the live path really depends on the crossfade-assign handler.
    usage = doctor.dance_client_usage()
    assert "osc/bridge.py" in usage["set_track_crossfade_assign"]
    assert "/live/clip_slot/create_audio_clip" in doctor.dance_sent_addresses()["create_audio_clip"]


# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------


def _fake_hf_snapshot(cache: Path, model: str, weights: str | None = "pytorch_model.bin") -> Path:
    repo = cache / ("models--" + model.replace("/", "--"))
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snap / "config.json").write_text("{}")
    if weights:
        (snap / weights).write_bytes(b"\0" * 1024)
    return snap


def test_clap_weights_present_missing_incomplete(tmp_path: Path):
    model = "laion/clap-htsat-unfused"
    assert doctor.check_clap_weights(model, tmp_path / "empty").status == WARN

    _fake_hf_snapshot(tmp_path / "partial", model, weights=None)
    res = doctor.check_clap_weights(model, tmp_path / "partial")
    assert res.status == WARN and "no complete snapshot" in res.summary

    _fake_hf_snapshot(tmp_path / "full", model)
    res = doctor.check_clap_weights(model, tmp_path / "full")
    assert res.status == OK and "pytorch_model.bin" in res.summary


def _fake_demucs_remote(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "htdemucs_ft.yaml").write_text("models: ['f7e0c4bc', 'd12395a8']\nweights: []\n")
    (d / "files.txt").write_text(
        "# Demucs\nroot: hybrid_transformer/\nf7e0c4bc-ba3fe64a.th\nd12395a8-e57c48e6.th\n"
    )
    return d


def test_demucs_weights_present_and_missing(tmp_path: Path):
    remote = _fake_demucs_remote(tmp_path / "remote")
    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / "f7e0c4bc-ba3fe64a.th").write_bytes(b"\0" * 10)

    res = doctor.check_demucs_weights("htdemucs_ft", hub, remote)
    assert res.status == WARN
    assert "1 of 2" in res.summary
    assert "d12395a8-e57c48e6.th" in res.details[0]

    (hub / "checkpoints" / "d12395a8-e57c48e6.th").write_bytes(b"\0" * 10)
    assert doctor.check_demucs_weights("htdemucs_ft", hub, remote).status == OK
    assert doctor.check_demucs_weights("my_local_model", hub, remote).status == INFO


def test_demucs_real_package_metadata_lists_four_ft_specialists():
    remote = doctor._demucs_remote_dir()
    if remote is None:
        pytest.skip("demucs not installed")
    files = doctor.demucs_checkpoint_files("htdemucs_ft", remote)
    assert files is not None and len(files) == 4
    assert all(f.endswith(".th") and "*" not in f for f in files)


def test_cache_dir_resolution_follows_env():
    assert doctor.hf_hub_cache({"HF_HUB_CACHE": "/x/hub"}) == Path("/x/hub")
    assert doctor.hf_hub_cache({"HF_HOME": "/h"}) == Path("/h/hub")
    assert doctor.hf_hub_cache({"XDG_CACHE_HOME": "/c"}) == Path("/c/huggingface/hub")
    assert doctor.torch_hub_dir({"TORCH_HOME": "/t"}) == Path("/t/hub")
    assert doctor.torch_hub_dir({"XDG_CACHE_HOME": "/c"}) == Path("/c/torch/hub")


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe / ingest deps
# ---------------------------------------------------------------------------


def _run_ok(cmd, **_kw):
    return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version 7.1 Copyright\n", stderr="")


def _run_broken(cmd, **_kw):
    return subprocess.CompletedProcess(cmd, 134, stdout="", stderr="dyld: Library not loaded\n")


def _by_key(results):
    return {r.key: r for r in results}


def test_ffmpeg_ok_on_path():
    tools = {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}
    res = _by_key(doctor.check_ffmpeg(tools.get, _run_ok, lambda p: False))
    assert res["ffmpeg"].status == OK and "7.1" in res["ffmpeg"].summary
    assert res["ffprobe"].status == OK


def test_ffmpeg_missing_fails():
    res = _by_key(doctor.check_ffmpeg(lambda n: None, _run_ok, lambda p: False))
    assert res["ffmpeg"].status == FAIL
    assert res["ffprobe"].status == WARN


def test_ffmpeg_off_path_homebrew_warns_and_finds_sibling_ffprobe():
    present = {"/opt/homebrew/bin/ffmpeg", "/opt/homebrew/bin/ffprobe"}
    res = _by_key(doctor.check_ffmpeg(lambda n: None, _run_ok, present.__contains__))
    assert res["ffmpeg"].status == WARN and "not on PATH" in res["ffmpeg"].summary
    assert res["ffprobe"].status == OK and res["ffprobe"].summary == "/opt/homebrew/bin/ffprobe"


def test_ffmpeg_that_wont_run_fails():
    tools = {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}
    res = _by_key(doctor.check_ffmpeg(tools.get, _run_broken, lambda p: False))
    assert res["ffmpeg"].status == FAIL
    assert "exit 134" in res["ffmpeg"].summary


def test_ingest_deps_reuse_health_deps(monkeypatch: pytest.MonkeyPatch):
    from dance.config import get_settings
    from dance.spotify import csv_importer

    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: None)
    res = _by_key(doctor.check_ingest_deps(get_settings()))
    assert "ffmpeg" not in res  # check_ffmpeg owns that
    assert res["yt_dlp"].status == FAIL
    assert res["spotify_creds"].status == INFO

    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: "/venv/bin/yt-dlp")
    assert _by_key(doctor.check_ingest_deps(get_settings()))["yt_dlp"].status == OK


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _closed_port_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/health"


def test_backend_up_and_down():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/health"
        assert doctor.check_backend(url, timeout=2).status == OK
    finally:
        srv.shutdown()
        srv.server_close()

    res = doctor.check_backend(_closed_port_url(), timeout=0.5)
    assert res.status == INFO
    assert "uvicorn" in (res.fix or "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def healthy_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every doctor default at temp fixtures that should all pass."""
    from dance.config import get_settings
    from dance.spotify import csv_importer

    settings = get_settings()
    db_path = Path(settings.db_url.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _upgrade(db_path)

    tpl = doctor.read_template_info()
    apps = tmp_path / "Applications"
    apps.mkdir()
    _fake_live(apps, "Ableton Live 12 Standard", f"{tpl.version} (2026-04-24_{tpl.revision[:10]})")
    monkeypatch.setattr(doctor, "DEFAULT_APPS_DIRS", (apps,))
    monkeypatch.setattr(doctor, "DEFAULT_ABLETONOSC_DIR", _fake_abletonosc(tmp_path / "AbletonOSC"))
    monkeypatch.setattr(doctor, "BACKEND_HEALTH_URL", _closed_port_url())
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setattr(csv_importer, "_resolve_yt_dlp", lambda: "/venv/bin/yt-dlp")
    tools = {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}
    monkeypatch.setattr(doctor.shutil, "which", tools.get)
    monkeypatch.setattr(doctor.subprocess, "run", _run_ok)
    return settings, db_path


def test_cli_doctor_json_reports_every_check_and_exits_zero(healthy_env):
    from dance.cli import main

    settings, db_path = healthy_env
    before = db_path.read_bytes()

    result = CliRunner().invoke(main, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)  # stdout is pure JSON — no log noise
    keys = [c["key"] for c in report["checks"]]
    for k in (
        "db_migrations",
        "ffmpeg",
        "ffprobe",
        "yt_dlp",
        "live",
        "abletonosc",
        "clap_weights",
        "demucs_weights",
        "backend",
    ):
        assert k in keys
    by = {c["key"]: c for c in report["checks"]}
    assert by["db_migrations"]["status"] == OK
    assert by["live"]["status"] == OK
    assert by["abletonosc"]["status"] == OK
    assert by["clap_weights"]["status"] == WARN  # empty temp HF cache
    assert by["backend"]["status"] == INFO
    assert report["ok"] is True
    # The doctor skipped init_db/ensure_directories: nothing created or touched.
    assert db_path.read_bytes() == before
    assert not settings.library_dir.exists()
    assert not settings.stems_dir.exists()


def test_cli_doctor_json_survives_chatty_checks(healthy_env, monkeypatch: pytest.MonkeyPatch):
    """Log lines go to stdout through RichHandler; --json must stay parseable
    even when a dependency logs mid-check (alembic's autogenerate import does)."""
    import logging

    from dance.cli import main

    real = doctor.check_backend

    def chatty(url=None, timeout=0.5):
        logging.getLogger("alembic.runtime.plugins").warning("setup plugin noise")
        logging.getLogger("dance").error("something loud")
        return real(url, timeout)

    monkeypatch.setattr(doctor, "check_backend", chatty)

    # Under pytest, basicConfig() is a no-op (pytest owns the root handlers),
    # so emulate the real CLI: a root handler that writes to whatever
    # sys.stdout is at emit time, exactly like RichHandler(console=Console()).
    class StdoutHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            import sys

            sys.stdout.write(self.format(record) + "\n")

    handler = StdoutHandler()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        result = CliRunner().invoke(main, ["doctor", "--json"])
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ok"] is True
    assert logging.root.manager.disable == logging.NOTSET  # restored afterwards


def test_cli_doctor_exits_nonzero_on_fail_and_prints_fix(healthy_env):
    from dance.cli import main

    _settings, db_path = healthy_env
    with sqlite3.connect(db_path) as c:
        c.execute("UPDATE alembic_version SET version_num = 'c6e3ba31c1f0'")

    result = CliRunner().invoke(main, ["doctor"])

    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
    assert "migration(s) behind" in result.output
    assert "alembic upgrade head" in result.output
    assert _version(db_path) == ["c6e3ba31c1f0"]


def test_crashing_check_becomes_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from dance.config import get_settings

    def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(doctor, "check_live_installs", boom)
    monkeypatch.setattr(doctor, "BACKEND_HEALTH_URL", _closed_port_url())
    report = doctor.run_doctor(
        get_settings(),
        abletonosc_dir=tmp_path / "none",
        hf_cache=tmp_path / "hf",
        torch_hub=tmp_path / "torch",
    )
    live = next(c for c in report.checks if c.key == "live")
    assert live.status == WARN and "kaboom" in live.summary
