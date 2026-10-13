"""``dance doctor`` — read-only diagnostics for everything dance depends on
outside its own code.

Each check returns a :class:`CheckResult` with a status:

- ``ok``   — nothing to do.
- ``info`` — worth knowing, not a problem (e.g. the backend isn't running).
- ``warn`` — something degrades or will bite later; fix when convenient.
- ``fail`` — something dance relies on is broken right now. ``dance doctor``
  exits non-zero when any check fails.

Every check is strictly READ-ONLY. In particular the database is opened with
SQLite's ``immutable=1`` URI flag (no locks, no journal, no ``-wal``/``-shm``
side files) and the migration check never upgrades or stamps anything — it
only tells you what to run. The CLI skips its usual ``init_db()`` /
``ensure_directories()`` for this command so that running the doctor can't
change the state it is reporting on.

Why these checks exist (each one cost an evening at least once):

- **DB migrations** — the user's DB sat two Alembic revisions behind the code
  with nothing warning about it.
- **Ableton Live installs** — Live 12 Lite 12.0.2 and Live 12 Standard 12.4
  both register ``com.ableton.live``, so ``open x.als`` may launch the older,
  track-capped Lite. The ``.als`` template is captured from 12.4.
- **AbletonOSC** — dance drives Live through a hand-patched AbletonOSC clone.
  When a handler dance sends to is missing, Live drops the message and the
  bridge silently no-ops. The check derives the addresses dance sends from
  ``dance/osc/client.py`` and statically scans the installed Remote Script for
  the handlers it registers.
- **ffmpeg / ffprobe, model weights, backend** — the usual first-run friction.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import plistlib
import re
import shutil
import sqlite3
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OK = "ok"
INFO = "info"
WARN = "warn"
FAIL = "fail"

_SEVERITY = {OK: 0, INFO: 0, WARN: 1, FAIL: 2}

# Defaults. Module-level so tests (and unusual installs) can point them
# elsewhere; ``run_doctor`` also accepts explicit overrides.
DEFAULT_APPS_DIRS: tuple[Path, ...] = (
    Path("/Applications"),
    Path.home() / "Applications",
)
DEFAULT_ABLETONOSC_DIR = (
    Path.home() / "Music" / "Ableton" / "User Library" / "Remote Scripts" / "AbletonOSC"
)
BACKEND_HEALTH_URL = "http://127.0.0.1:8000/health"

_DANCE_PKG_DIR = Path(__file__).resolve().parent
_SETUP_DOC = "docs/abletonosc_setup.md"


@dataclass
class CheckResult:
    key: str
    label: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)
    fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    def counts(self) -> dict[str, int]:
        out = {OK: 0, INFO: 0, WARN: 0, FAIL: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": self.counts(),
            "checks": [c.to_dict() for c in self.checks],
        }


def _worst(statuses: Iterable[str]) -> str:
    worst = OK
    for s in statuses:
        if _SEVERITY[s] > _SEVERITY[worst]:
            worst = s
    return worst


# ---------------------------------------------------------------------------
# 1. Database migration state
# ---------------------------------------------------------------------------

# Partial unique indexes created by raw DDL (``init_db`` and the migrations).
# Autogenerate can't model them, so the schema comparison reports them as
# "extra" — expected, not drift.
_RAW_DDL_INDEXES = frozenset(
    {"uq_audio_analysis_track_fullmix", "uq_audio_analysis_stem", "uq_sets_one_active"}
)


def _sqlite_path(db_url: str) -> Path | None | str:
    """Return the file path of a ``sqlite:///`` URL, ``None`` for an
    in-memory DB, or the URL unchanged when it isn't SQLite."""
    from sqlalchemy.engine import make_url

    url = make_url(db_url)
    if url.get_backend_name() != "sqlite":
        return db_url
    if not url.database or url.database == ":memory:":
        return None
    return Path(url.database).expanduser()


def _readonly_engine(db_url: str) -> Any:
    """An engine that cannot write: SQLite files are opened ``immutable=1``."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    target = _sqlite_path(db_url)
    if isinstance(target, Path):
        uri = f"file:{urllib.parse.quote(str(target.resolve()))}?immutable=1"
        return create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(uri, uri=True),
            poolclass=NullPool,
        )
    return create_engine(db_url, poolclass=NullPool)


def _script_directory(script_location: Path | None) -> Any:
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(script_location or (_DANCE_PKG_DIR / "alembic")))


def _schema_drift(conn: Any) -> tuple[list[str], list[str]]:
    """Compare the live schema with the SQLAlchemy models (read-only).

    Returns ``(diffs, missing_raw_indexes)`` with the raw-DDL partial
    indexes filtered out of ``diffs``.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from dance.core.database import Base

    diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)

    seen_raw: set[str] = set()
    out: list[str] = []
    for d in diffs:
        if isinstance(d, tuple) and d and d[0] == "remove_index":
            name = getattr(d[1], "name", None)
            if name in _RAW_DDL_INDEXES:
                seen_raw.add(str(name))
                continue
        out.append(_describe_diff(d))
    missing = sorted(_RAW_DDL_INDEXES - seen_raw)
    return out, missing


def _describe_diff(d: Any) -> str:
    if isinstance(d, list):  # modify_* diffs come as a list of tuples
        return "; ".join(_describe_diff(x) for x in d)
    if not isinstance(d, tuple) or not d:
        return repr(d)
    op = str(d[0])
    obj = d[1] if len(d) > 1 else None
    if op in ("add_table", "remove_table"):
        return f"{op} {getattr(obj, 'name', obj)}"
    if op in ("add_column", "remove_column"):
        col = d[3] if len(d) > 3 else None
        return f"{op} {d[2]}.{getattr(col, 'name', col)}"
    if op in ("add_index", "remove_index"):
        return f"{op} {getattr(obj, 'name', obj)}"
    if op.startswith("modify_"):
        return f"{op} {d[2]}.{d[3]}" if len(d) > 3 else op
    return op


def check_db_migrations(db_url: str, script_location: Path | None = None) -> CheckResult:
    """Current Alembic revision of ``db_url`` vs the migration scripts' head.

    Never upgrades or stamps — the fix hint says what to run.
    """
    # The first ``import alembic`` logs "setup plugin …" at INFO, and so does
    # MigrationContext.configure(); keep that out of the report (and --json).
    alembic_log = logging.getLogger("alembic")
    prev = alembic_log.level
    alembic_log.setLevel(logging.WARNING)
    try:
        return _check_db_migrations(db_url, script_location)
    finally:
        alembic_log.setLevel(prev)


def _check_db_migrations(db_url: str, script_location: Path | None) -> CheckResult:
    from sqlalchemy import inspect, text

    key, label = "db_migrations", "Database migrations"
    script = _script_directory(script_location)
    heads = list(script.get_heads())
    target = _sqlite_path(db_url)
    where = str(target) if isinstance(target, Path) else db_url

    if len(heads) > 1:
        return CheckResult(
            key,
            label,
            FAIL,
            f"migration scripts have {len(heads)} heads: {', '.join(heads)}",
            fix="Two branches each added a migration. Merge them: "
            "`alembic merge -m 'merge heads' " + " ".join(heads) + "`.",
        )
    head = heads[0] if heads else None

    if target is None:
        return CheckResult(key, label, INFO, "in-memory database — nothing to check")
    if isinstance(target, Path) and not target.exists():
        return CheckResult(
            key,
            label,
            WARN,
            f"no database yet at {target}",
            fix="From the repo root: `alembic upgrade head` (bin/setup.sh does this) — "
            "creating it with alembic records the revision so later upgrades work.",
        )

    details: list[str] = []
    if isinstance(target, Path):
        wal = Path(str(target) + "-wal")
        if wal.exists() and wal.stat().st_size > 0:
            details.append(
                "A -wal file is present (backend running?). The doctor reads the DB "
                "immutably, so writes not yet checkpointed into the main file are not seen."
            )

    engine = _readonly_engine(db_url)
    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            tables = set(insp.get_table_names())
            if "alembic_version" in tables:
                current = sorted(
                    r[0] for r in conn.execute(text("SELECT version_num FROM alembic_version"))
                )
            else:
                current = []
            drift: list[str] = []
            missing_idx: list[str] = []
            drift_error: str | None = None
            if tables:
                try:
                    drift, missing_idx = _schema_drift(conn)
                except Exception as exc:  # noqa: BLE001 — diagnostics must not crash
                    drift_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            key,
            label,
            FAIL,
            f"could not read {where}: {type(exc).__name__}: {exc}",
            details=details,
        )
    finally:
        engine.dispose()

    if drift_error:
        details.append(f"schema comparison skipped ({drift_error})")
    if missing_idx:
        details.append("missing partial unique index(es): " + ", ".join(missing_idx))

    tables.discard("alembic_version")
    if not current:
        if not tables:
            return CheckResult(
                key,
                label,
                WARN,
                f"database at {where} is empty",
                details=details,
                fix="From the repo root: `alembic upgrade head`.",
            )
        if drift:
            details.extend(f"drift: {d}" for d in drift[:12])
            return CheckResult(
                key,
                label,
                FAIL,
                f"unversioned database ({len(tables)} tables, no alembic_version) "
                f"that differs from the models in {len(drift)} place(s)",
                details=details,
                fix="Created by init_db() rather than Alembic. Back it up, then work out "
                "which revision it matches before `alembic stamp <rev>` + `alembic upgrade head`.",
            )
        return CheckResult(
            key,
            label,
            WARN,
            f"unversioned database ({len(tables)} tables, no alembic_version); "
            "schema matches the models",
            details=details,
            fix="Record the revision so future upgrades apply: back up the DB, then "
            f"`alembic stamp {head or 'head'}` from the repo root.",
        )

    unknown = [r for r in current if _get_revision(script, r) is None]
    if unknown:
        return CheckResult(
            key,
            label,
            FAIL,
            f"DB is at revision {', '.join(unknown)}, which this checkout doesn't know",
            details=details + [f"code head: {head}"],
            fix="The DB was migrated by a newer or different checkout. Check out the code "
            "that has that revision (or merge it) before running dance against this DB.",
        )

    if head is not None and current == [head]:
        status = OK
        rev = _get_revision(script, head)
        summary = f"at head {head}" + (f" — {_first_line(rev.doc)}" if rev is not None else "")
        if drift:
            status = WARN
            summary += f", but schema differs from the models in {len(drift)} place(s)"
            details.extend(f"drift: {d}" for d in drift[:12])
        if missing_idx:
            status = WARN
        fix = None
        if missing_idx:
            fix = "Run any regular dance command once (e.g. `dance status`) — init_db() "
            fix += "creates the partial indexes."
        elif drift:
            fix = "A migration is missing for a model change, or the DB was edited by hand."
        return CheckResult(key, label, status, summary, details=details, fix=fix)

    # Behind: list the revisions between current and head, oldest first.
    pending: list[Any] = []
    try:
        pending = list(script.iterate_revisions(head, current[0] if len(current) == 1 else current))
    except Exception as exc:  # noqa: BLE001
        details.append(f"could not walk revisions: {exc}")
    pending.reverse()
    details.extend(f"pending: {r.revision} — {_first_line(r.doc)}" for r in pending)
    backup = f"cp '{target}' '{target}.bak-$(date +%Y%m%d)' && " if isinstance(target, Path) else ""
    return CheckResult(
        key,
        label,
        FAIL,
        f"{len(pending) or 'some'} migration(s) behind: DB at {', '.join(current)}, code at {head}",
        details=details,
        fix=f"Stop the backend, then from the repo root: {backup}alembic upgrade head",
    )


def _get_revision(script: Any, rev: str) -> Any:
    try:
        return script.get_revision(rev)
    except Exception:  # noqa: BLE001 — alembic raises several types for unknown revs
        return None


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().splitlines()[0] if (doc or "").strip() else ""


# ---------------------------------------------------------------------------
# 2. ffmpeg / ffprobe + the ingest deps /health/deps already reports
# ---------------------------------------------------------------------------


def check_ffmpeg(
    which: Callable[[str], str | None] | None = None,
    run: Callable[..., Any] | None = None,
    exists: Callable[[str], bool] | None = None,
) -> list[CheckResult]:
    """ffmpeg must be on PATH (``dance process`` preflight uses PATH) and must
    actually run; ffprobe sits next to it and yt-dlp's audio extraction needs
    it. Resolution mirrors ``csv_importer._resolve_ffmpeg`` so this never
    disagrees with the ingest worker or the /health/deps chip."""
    which = which or shutil.which
    run = run or subprocess.run
    exists = exists or os.path.exists
    homebrew = ("/opt/homebrew/bin", "/usr/local/bin")

    on_path = which("ffmpeg")
    resolved = on_path or next((f"{d}/ffmpeg" for d in homebrew if exists(f"{d}/ffmpeg")), None)
    results: list[CheckResult] = []
    if resolved is None:
        results.append(
            CheckResult(
                "ffmpeg",
                "ffmpeg",
                FAIL,
                "not found",
                fix="brew install ffmpeg — needed for downloads (spotDL / yt-dlp) and decoding.",
            )
        )
    else:
        version, err = _tool_version(run, resolved)
        if err is not None:
            results.append(
                CheckResult(
                    "ffmpeg",
                    "ffmpeg",
                    FAIL,
                    f"{resolved} is present but won't run: {err}",
                    fix="brew reinstall ffmpeg — a half-broken ffmpeg shows up as abrupt native "
                    "exits during decode (docs/troubleshooting.md).",
                )
            )
        elif on_path is None:
            results.append(
                CheckResult(
                    "ffmpeg",
                    "ffmpeg",
                    WARN,
                    f"{resolved} works but is not on PATH",
                    details=[version] if version else [],
                    fix=f"Add {Path(resolved).parent} to PATH — `dance process`/`dance sync` "
                    "preflight looks ffmpeg up on PATH and will refuse to start.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "ffmpeg",
                    "ffmpeg",
                    OK,
                    version or resolved,
                    details=[resolved],
                )
            )

    probe = which("ffprobe")
    if probe is None and resolved is not None:
        sibling = str(Path(resolved).parent / "ffprobe")
        probe = sibling if exists(sibling) else None
    if probe is None:
        results.append(
            CheckResult(
                "ffprobe",
                "ffprobe",
                WARN,
                "not found",
                fix="brew install ffmpeg (ships ffprobe) — yt-dlp needs it to extract audio.",
            )
        )
    else:
        results.append(CheckResult("ffprobe", "ffprobe", OK, probe))
    return results


def _tool_version(run: Callable[..., Any], exe: str) -> tuple[str | None, str | None]:
    try:
        proc = run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
        return None, f"exit {proc.returncode}" + (f": {tail[0]}" if tail else "")
    first = (proc.stdout or "").strip().splitlines()[:1]
    return (first[0] if first else None), None


def check_ingest_deps(settings: Any) -> list[CheckResult]:
    """yt-dlp, YouTube cookies and Spotify credentials — straight from the
    same ``_deps_status`` the companion's MasterStrip chip renders (ffmpeg is
    handled by :func:`check_ffmpeg`, which also checks PATH and ffprobe)."""
    from dance.api.app import _deps_status

    out: list[CheckResult] = []
    for c in _deps_status(settings)["checks"]:
        if c["key"] == "ffmpeg":
            continue
        if c["status"] == "ok":
            out.append(CheckResult(c["key"], c["label"], OK, str(c["detail"])))
        elif c["required"]:
            out.append(CheckResult(c["key"], c["label"], FAIL, "missing", fix=str(c["detail"])))
        else:
            out.append(
                CheckResult(c["key"], c["label"], INFO, "not configured", fix=str(c["detail"]))
            )
    return out


# ---------------------------------------------------------------------------
# 3. Ableton Live installs vs the .als template
# ---------------------------------------------------------------------------

_LIMITED_EDITIONS = frozenset({"Lite", "Intro"})


@dataclass(frozen=True)
class TemplateInfo:
    creator: str
    version: str
    revision: str
    minor_version: str
    sha256: str


@dataclass(frozen=True)
class LiveInstall:
    app: Path
    edition: str
    version: str | None
    build: str | None
    bundle_id: str | None
    error: str | None = None

    @property
    def build_hash(self) -> str | None:
        if not self.build:
            return None
        return self.build.rsplit("_", 1)[-1]


def read_template_info(path: Path | None = None) -> TemplateInfo:
    """Root attributes + SHA-256 of the bundled ``blank_live12.xml``."""
    if path is None:
        from importlib import resources

        ref = resources.files("dance.als") / "templates" / "blank_live12.xml"
        data = ref.read_bytes()
    else:
        data = path.read_bytes()
    head = data[:4096].decode("utf-8", errors="replace")
    m = re.search(r"<Ableton\b([^>]*)>", head)
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1))) if m else {}
    creator = attrs.get("Creator", "")
    vm = re.search(r"(\d+(?:\.\d+)+)", creator)
    return TemplateInfo(
        creator=creator,
        version=vm.group(1) if vm else "",
        revision=attrs.get("Revision", ""),
        minor_version=attrs.get("MinorVersion", ""),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _version_tuple(v: str | None) -> tuple[int, ...]:
    if not v:
        return ()
    parts = [int(p) for p in re.findall(r"\d+", v)]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def find_live_installs(apps_dirs: Sequence[Path]) -> list[LiveInstall]:
    out: list[LiveInstall] = []
    seen: set[Path] = set()
    for d in apps_dirs:
        if not d.is_dir():
            continue
        for app in sorted(d.glob("Ableton Live*.app")):
            if app.resolve() in seen:
                continue
            seen.add(app.resolve())
            em = re.match(r"Ableton Live(?: \d+)?\s*(.*)\.app$", app.name)
            edition = (em.group(1).strip() if em else "") or "?"
            try:
                with open(app / "Contents" / "Info.plist", "rb") as fh:
                    info = plistlib.load(fh)
            except (OSError, plistlib.InvalidFileException, ValueError) as exc:
                out.append(LiveInstall(app, edition, None, None, None, error=str(exc)))
                continue
            raw = str(info.get("CFBundleShortVersionString") or info.get("CFBundleVersion") or "")
            vm = re.match(r"\s*(\d+(?:\.\d+)*)\s*(?:\(([^)]*)\))?", raw)
            out.append(
                LiveInstall(
                    app=app,
                    edition=edition,
                    version=vm.group(1) if vm else (raw or None),
                    build=(vm.group(2) if vm else None),
                    bundle_id=info.get("CFBundleIdentifier"),
                )
            )
    return out


def check_live_installs(
    apps_dirs: Sequence[Path] | None = None,
    template: TemplateInfo | None = None,
) -> CheckResult:
    key, label = "live", "Ableton Live"
    apps_dirs = list(apps_dirs if apps_dirs is not None else DEFAULT_APPS_DIRS)
    template = template or read_template_info()
    tpl_v = _version_tuple(template.version)
    tpl_hash = template.revision[:10]
    installs = find_live_installs(apps_dirs)

    details = [
        f"template blank_live12.xml: {template.creator!r}, Revision {template.revision[:12]}…, "
        f"sha256 {template.sha256[:12]}…"
    ]
    if not installs:
        if not any(d.is_dir() for d in apps_dirs):
            return CheckResult(
                key,
                label,
                INFO,
                "no Applications folder to search (not macOS?)",
                details=details,
            )
        return CheckResult(
            key,
            label,
            FAIL,
            "no 'Ableton Live*.app' in " + ", ".join(map(str, apps_dirs)),
            details=details,
            fix=f"Install Live {template.version} Standard or Suite — the .als template "
            f"was captured from {template.creator}.",
        )

    usable: list[LiveInstall] = []
    exact: list[LiveInstall] = []
    newer: list[LiveInstall] = []
    for inst in installs:
        notes: list[str] = []
        if inst.error:
            notes.append(f"unreadable Info.plist ({inst.error})")
        else:
            v = _version_tuple(inst.version)
            if inst.edition in _LIMITED_EDITIONS:
                notes.append(f"{inst.edition} is track-capped")
            if v and tpl_v and v < tpl_v:
                notes.append(f"older than the {template.version} template")
            elif v and tpl_v and v > tpl_v:
                notes.append(f"newer than the {template.version} template")
            elif inst.build_hash and tpl_hash and inst.build_hash == tpl_hash:
                notes.append("exact build the template was saved from")
            elif v == tpl_v:
                notes.append(f"same version as the template, different build ({inst.build})")
            ok_edition = inst.edition not in _LIMITED_EDITIONS
            if ok_edition and v and tpl_v and v >= tpl_v:
                usable.append(inst)
                if v > tpl_v:
                    newer.append(inst)
                elif inst.build_hash == tpl_hash:
                    exact.append(inst)
        details.append(
            f"{inst.app.name}: {inst.version or '?'}"
            + (f" ({inst.build})" if inst.build else "")
            + (f" [{inst.bundle_id}]" if inst.bundle_id else "")
            + (" — " + "; ".join(notes) if notes else "")
        )

    if not usable:
        return CheckResult(
            key,
            label,
            FAIL,
            f"no installed Live can be trusted with dance's {template.version} .als sets",
            details=details,
            fix=f"Install Live {template.version} Standard or Suite (Lite/Intro are track-capped; "
            "older versions refuse newer Sets).",
        )

    best = (exact or sorted(usable, key=lambda i: _version_tuple(i.version))[-1:])[0]
    app_name = best.app.name[: -len(".app")]
    open_cmd = f'open -a "{app_name}" <file>.als'
    shared = [i for i in installs if i.bundle_id and i.bundle_id == best.bundle_id]
    if len(shared) > 1:
        others = ", ".join(f"{i.app.name[:-4]} {i.version}" for i in shared if i is not best)
        return CheckResult(
            key,
            label,
            WARN,
            f"{len(shared)} installs share {best.bundle_id} — `open <file>.als` may launch "
            f"{others} instead of {app_name} {best.version}",
            details=details,
            fix=open_cmd,
        )
    if best in newer:
        return CheckResult(
            key,
            label,
            WARN,
            f"{app_name} {best.version} is newer than the {template.version} template",
            details=details,
            fix="Live usually upgrades older Sets fine; if it rejects a generated .als, "
            "refresh the template (docs/troubleshooting.md 'Live rejects the .als entirely') "
            "and re-verify in Live (docs/live-compat.md).",
        )
    return CheckResult(
        key,
        label,
        OK,
        f"{app_name} {best.version} matches the .als template",
        details=details,
    )


# ---------------------------------------------------------------------------
# 4. AbletonOSC: which handlers dance sends to are missing from the install
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OscPatch:
    key: str
    title: str
    where: str  # doc section (or commit) that describes the patch
    addresses: frozenset[str]


# Fork patches dance depends on, cross-checked against the addresses
# ``dance/osc/client.py`` sends (tests/test_doctor.py asserts every address
# here is really sent). Order = rough importance.
KNOWN_PATCHES: tuple[OscPatch, ...] = (
    OscPatch(
        "create_audio_clip",
        "load a sample into a clip slot",
        f"{_SETUP_DOC} § 'One-time AbletonOSC patch for auto-loading stems'",
        frozenset({"/live/clip_slot/create_audio_clip"}),
    ),
    OscPatch(
        "crossfade_assign",
        "per-track crossfader assignment",
        f"{_SETUP_DOC} § 'One-time AbletonOSC patch for crossfader assignment'",
        frozenset({"/live/track/get/crossfade_assign", "/live/track/set/crossfade_assign"}),
    ),
    OscPatch(
        "master_crossfader",
        "master crossfader get/set/listen",
        "commit 63f6b0b ('osc: master crossfader handler') — handler source is not "
        "in this repo or the setup doc",
        frozenset(
            {
                "/live/song/get/crossfader",
                "/live/song/set/crossfader",
                "/live/song/start_listen/crossfader",
                "/live/song/stop_listen/crossfader",
            }
        ),
    ),
    OscPatch(
        "warp_markers",
        "warp-marker read/repair (scripts/warp_probe.py)",
        "docs/proposals/warp-guard.md § 'Option C' (commit d54c002) — handler source is "
        "not in this repo",
        frozenset(
            {
                "/live/clip/add_warp_marker",
                "/live/clip/remove_warp_marker",
                "/live/clip/get/warp_marker_times",
            }
        ),
    ),
)

_LOG_FIX_DOC = f"{_SETUP_DOC} § 'Fix the clip_slot_callback logging bug'"
_LOG_BUG_RE = re.compile(r"logger\.info\(\s*track_index\s*,")


class _HandlerScanner(ast.NodeVisitor):
    """Collect the OSC addresses an AbletonOSC module registers.

    Understands the patterns upstream uses: literal ``add_handler("/live/…")``
    and ``for x in <list>: add_handler("/live/…/%s" % x)`` where ``<list>`` is
    a list literal, a name bound to one, or a ``+`` of those (f-strings and
    string concatenation work too). Anything else is counted in
    ``unresolved`` so the report can say the scan was partial.
    """

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.bindings: dict[str, list[str]] = {}
        self.addresses: set[str] = set()
        self.unresolved = 0

    # -- list tracking --------------------------------------------------
    def _strs(self, node: ast.AST) -> list[str] | None:
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if len(vals) == len(node.elts) and all(isinstance(v, str) for v in vals):
                return [str(v) for v in vals]
            return None
        if isinstance(node, ast.Name):
            return list(self.lists[node.id]) if node.id in self.lists else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._strs(node.left), self._strs(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        vals = self._strs(node.value)
        for t in node.targets:
            if isinstance(t, ast.Name) and vals is not None:
                self.lists[t.id] = vals
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            vals = self._strs(node.value)
            if vals is not None and node.target.id in self.lists:
                self.lists[node.target.id] = self.lists[node.target.id] + vals
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        vals = self._strs(node.iter)
        if isinstance(node.target, ast.Name) and vals is not None:
            name = node.target.id
            saved = self.bindings.get(name)
            self.bindings[name] = vals
            for stmt in node.body:
                self.visit(stmt)
            if saved is None:
                self.bindings.pop(name, None)
            else:
                self.bindings[name] = saved
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    # -- address evaluation ---------------------------------------------
    def _eval(self, node: ast.AST) -> list[str] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.Name) and node.id in self.bindings:
            return list(self.bindings[node.id])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            fmt = self._eval(node.left)
            args = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
            arg_vals = [self._eval(a) for a in args]
            if fmt is None or any(v is None for v in arg_vals):
                return None
            out: list[str] = []
            for f in fmt:
                for combo in itertools.product(*[v for v in arg_vals if v is not None]):
                    try:
                        out.append(f % tuple(combo))
                    except (TypeError, ValueError):
                        return None
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._eval(node.left), self._eval(node.right)
            if left is None or right is None:
                return None
            return [a + b for a in left for b in right]
        if isinstance(node, ast.JoinedStr):
            pieces: list[list[str]] = []
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    ev = self._eval(v.value)
                    if ev is None:
                        return None
                    pieces.append(ev)
                else:
                    ev = self._eval(v)
                    if ev is None:
                        return None
                    pieces.append(ev)
            return ["".join(p) for p in itertools.product(*pieces)]
        return None

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_handler" and node.args:
            vals = self._eval(node.args[0])
            if vals is None:
                self.unresolved += 1
            else:
                self.addresses.update(vals)
        self.generic_visit(node)


def scan_abletonosc_handlers(root: Path) -> tuple[set[str], int, list[str]]:
    """Statically scan an AbletonOSC checkout.

    Returns ``(addresses, unresolved_add_handler_calls, parse_errors)``.
    """
    files = sorted((root / "abletonosc").glob("*.py"))
    if (root / "manager.py").exists():
        files.append(root / "manager.py")
    addresses: set[str] = set()
    unresolved = 0
    errors: list[str] = []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"), filename=str(f))
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"{f.name}: {type(exc).__name__}: {exc}")
            continue
        scanner = _HandlerScanner()
        scanner.visit(tree)
        addresses |= scanner.addresses
        unresolved += scanner.unresolved
    return addresses, unresolved, errors


def dance_sent_addresses(client_path: Path | None = None) -> dict[str, set[str]]:
    """``AbletonOSCClient`` method name -> OSC addresses it sends, read from
    the source of ``dance/osc/client.py`` (every send goes through ``_send``)."""
    path = client_path or (_DANCE_PKG_DIR / "osc" / "client.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, set[str]] = {}
    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and cls.name == "AbletonOSCClient"):
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name == "_send":
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_send"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    out.setdefault(fn.name, set()).add(node.args[0].value)
    return out


_CLIENT_CALL_RE = re.compile(r"\bclient\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def dance_client_usage(src_root: Path | None = None) -> dict[str, set[str]]:
    """Client method name -> dance modules (relative paths) that call it.

    Distinguishes handlers the live app depends on (bridge / API) from ones
    only standalone probes use.
    """
    root = src_root or _DANCE_PKG_DIR
    out: dict[str, set[str]] = {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel == "osc/client.py":
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _CLIENT_CALL_RE.finditer(text):
            out.setdefault(m.group(1), set()).add(rel)
    return out


def _read_git_head(root: Path) -> dict[str, str] | None:
    """Read HEAD of a git checkout from its files (no ``git`` subprocess)."""
    git = root / ".git"
    if git.is_file():  # worktree / submodule: "gitdir: <path>"
        m = re.match(r"gitdir:\s*(.+)", git.read_text(encoding="utf-8").strip())
        if not m:
            return None
        git = (root / m.group(1)).resolve()
    if not git.is_dir():
        return None
    info: dict[str, str] = {}
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        info["ref"] = ref
        ref_file = git / ref
        sha = ref_file.read_text(encoding="utf-8").strip() if ref_file.is_file() else ""
        if not sha and (git / "packed-refs").is_file():
            for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    sha = parts[0]
        if sha:
            info["commit"] = sha
    else:
        info["commit"] = head
    cfg = git / "config"
    if cfg.is_file():
        m = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*(\S+)', cfg.read_text(encoding="utf-8"))
        if m:
            info["origin"] = m.group(1)
    if (git / "shallow").is_file():
        info["shallow"] = "yes"
    return info


def check_abletonosc(
    root: Path | None = None,
    client_path: Path | None = None,
    src_root: Path | None = None,
) -> CheckResult:
    key, label = "abletonosc", "AbletonOSC"
    root = root or DEFAULT_ABLETONOSC_DIR
    if not (root / "abletonosc").is_dir():
        return CheckResult(
            key,
            label,
            FAIL,
            f"not installed at {root}",
            fix=f"{_SETUP_DOC} § 'Install steps', then apply the dance patches in that doc "
            "and fully quit + reopen Live.",
        )

    details: list[str] = []
    git = _read_git_head(root)
    commit = (git or {}).get("commit")
    if git:
        details.append(
            f"git: {git.get('origin', '?')} @ {commit or '?'}"
            + (f" ({git['ref']})" if "ref" in git else " (detached)")
            + (", shallow clone" if git.get("shallow") else "")
        )
    else:
        details.append("not a git checkout — upstream commit unknown")

    served, unresolved, errors = scan_abletonosc_handlers(root)
    sent = dance_sent_addresses(client_path)
    usage = dance_client_usage(src_root)
    addr_methods: dict[str, set[str]] = {}
    for meth, addrs in sent.items():
        for a in addrs:
            addr_methods.setdefault(a, set()).add(meth)
    all_sent = set(addr_methods)
    missing = all_sent - served

    def users(addr: str) -> set[str]:
        return {u for m in addr_methods.get(addr, ()) for u in usage.get(m, ())}

    statuses: list[str] = []
    fixes: list[str] = []
    installed_patches: list[str] = []
    covered: set[str] = set()
    for p in KNOWN_PATCHES:
        relevant = p.addresses & all_sent
        if not relevant:
            continue
        absent = sorted(relevant & missing)
        covered |= relevant
        if not absent:
            installed_patches.append(p.key)
            continue
        used_by = sorted({u for a in absent for u in users(a)})
        sev = FAIL if used_by else WARN
        statuses.append(sev)
        details.append(
            f"MISSING {p.key} ({p.title}): {', '.join(absent)} — "
            + (f"sent by {', '.join(used_by)}" if used_by else "only standalone probes send these")
        )
        fixes.append(f"{p.key}: {p.where}")

    stray = sorted(missing - covered)
    for a in stray:
        used_by = sorted(users(a))
        sev = FAIL if used_by else WARN
        statuses.append(sev)
        details.append(
            f"MISSING {a} (no documented patch) — "
            + (f"sent by {', '.join(used_by)}" if used_by else "client method unused by the app")
        )
    if stray:
        fixes.append("undocumented gaps: compare with upstream AbletonOSC and document the patch")

    clip_slot = root / "abletonosc" / "clip_slot.py"
    log_bug = False
    if clip_slot.is_file():
        try:
            log_bug = bool(_LOG_BUG_RE.search(clip_slot.read_text(encoding="utf-8")))
        except OSError:
            log_bug = False
    if log_bug:
        statuses.append(WARN)
        details.append(
            "clip_slot_callback logging bug present (every clip_slot call logs a "
            "TypeError into Live's log)"
        )
        fixes.append(f"logging fix: {_LOG_FIX_DOC}")
    else:
        installed_patches.append("clip_slot_log_fix")

    if installed_patches:
        details.insert(1, "patches present: " + ", ".join(installed_patches))
    if unresolved:
        details.append(
            f"{unresolved} add_handler call(s) could not be resolved statically; "
            "handlers registered that way are not counted"
        )
    details.extend(f"parse error: {e}" for e in errors)

    status = _worst(statuses) if statuses else OK
    n_missing = len(missing)
    short = (commit or "")[:10]
    summary = (f"upstream {short}; " if short else "") + (
        f"all {len(all_sent)} addresses dance sends have handlers"
        if not n_missing
        else f"{n_missing} of {len(all_sent)} addresses dance sends have no handler "
        "(Live drops them and the bridge silently no-ops)"
    )
    fix = None
    if fixes:
        fix = "\n".join(
            [f"Patch {root / 'abletonosc'}:"]
            + [f"  - {f}" for f in fixes]
            + ["then fully quit and reopen Live (toggling the Control Surface is not enough)."]
        )
    return CheckResult(key, label, status, summary, details=details, fix=fix)


# ---------------------------------------------------------------------------
# 5. Model weights in local caches (presence only — never downloads)
# ---------------------------------------------------------------------------


def hf_hub_cache(env: dict[str, str] | None = None) -> Path:
    """Mirror huggingface_hub's cache resolution without importing it."""
    env = dict(os.environ if env is None else env)
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    base = (
        Path(env["XDG_CACHE_HOME"]).expanduser()
        if env.get("XDG_CACHE_HOME")
        else (Path.home() / ".cache")
    )
    return base / "huggingface" / "hub"


def torch_hub_dir(env: dict[str, str] | None = None) -> Path:
    """Mirror ``torch.hub.get_dir()`` without importing torch."""
    env = dict(os.environ if env is None else env)
    if env.get("TORCH_HOME"):
        return Path(env["TORCH_HOME"]).expanduser() / "hub"
    base = (
        Path(env["XDG_CACHE_HOME"]).expanduser()
        if env.get("XDG_CACHE_HOME")
        else (Path.home() / ".cache")
    )
    return base / "torch" / "hub"


def _mb(n: int) -> str:
    return f"{n / 1e6:.0f} MB"


def check_clap_weights(model: str, cache: Path | None = None) -> CheckResult:
    key, label = "clap_weights", "CLAP weights"
    cache = cache or hf_hub_cache()
    repo = cache / ("models--" + model.replace("/", "--"))
    fix = (
        f"First `dance process` downloads {model} (~615 MB) into {cache}; to pre-warm see "
        "docs/troubleshooting.md 'Model weights download on first run'."
    )
    if not repo.is_dir():
        return CheckResult(key, label, WARN, f"{model} not in {cache}", fix=fix)
    snapshots = repo / "snapshots"
    ref_main = repo / "refs" / "main"
    candidates: list[Path] = []
    if ref_main.is_file():
        candidates.append(snapshots / ref_main.read_text(encoding="utf-8").strip())
    if snapshots.is_dir():
        candidates.extend(sorted(p for p in snapshots.iterdir() if p not in candidates))
    for snap in candidates:
        for name in ("model.safetensors", "pytorch_model.bin"):
            w = snap / name
            if w.exists() and (snap / "config.json").exists():
                return CheckResult(
                    key,
                    label,
                    OK,
                    f"{model} cached ({name}, {_mb(w.stat().st_size)})",
                    details=[str(snap)],
                )
    return CheckResult(
        key,
        label,
        WARN,
        f"{model} cache folder exists but has no complete snapshot",
        details=[str(repo)],
        fix=fix,
    )


def _demucs_remote_dir() -> Path | None:
    spec = importlib.util.find_spec("demucs")  # locates without importing torch
    if spec is None or not spec.origin:
        return None
    remote = Path(spec.origin).parent / "remote"
    return remote if remote.is_dir() else None


def demucs_checkpoint_files(model: str, remote_dir: Path) -> list[str] | None:
    """Checkpoint filenames a pretrained Demucs model (bag) needs, from the
    package's own ``remote/<model>.yaml`` + ``remote/files.txt``."""
    yaml_path = remote_dir / f"{model}.yaml"
    if not yaml_path.is_file():
        return None
    m = re.search(r"models:\s*\[([^\]]*)\]", yaml_path.read_text(encoding="utf-8"))
    if not m:
        return None
    sigs = re.findall(r"['\"]?([0-9a-f]{8})['\"]?", m.group(1))
    names = [
        line.strip()
        for line in (remote_dir / "files.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().endswith(".th")
    ]
    out: list[str] = []
    for s in sigs:
        match = next((n for n in names if n.startswith(s + "-")), None)
        out.append(match or f"{s}-*.th")
    return out


def check_demucs_weights(
    model: str, hub_dir: Path | None = None, remote_dir: Path | None = None
) -> CheckResult:
    key, label = "demucs_weights", "Demucs weights"
    checkpoints = (hub_dir or torch_hub_dir()) / "checkpoints"
    remote = remote_dir or _demucs_remote_dir()
    if remote is None:
        return CheckResult(
            key,
            label,
            WARN,
            "demucs package not importable",
            fix="pip install -e '.[dev]' in the venv (demucs is a core dependency).",
        )
    files = demucs_checkpoint_files(model, remote)
    if files is None:
        return CheckResult(
            key,
            label,
            INFO,
            f"{model!r} is not a pretrained Demucs model name; can't check",
        )
    present = [f for f in files if (checkpoints / f).is_file()]
    missing = [f for f in files if f not in present]
    size = sum((checkpoints / f).stat().st_size for f in present)
    if not missing:
        return CheckResult(
            key,
            label,
            OK,
            f"{model}: {len(files)} checkpoint(s) cached ({_mb(size)})",
            details=[str(checkpoints)],
        )
    return CheckResult(
        key,
        label,
        WARN,
        f"{model}: {len(missing)} of {len(files)} checkpoint(s) missing from {checkpoints}",
        details=[f"missing: {', '.join(missing)}"],
        fix="The first separation downloads them (~84 MB each for the htdemucs_ft "
        "specialists); see docs/troubleshooting.md.",
    )


# ---------------------------------------------------------------------------
# 6. Backend reachability (optional — not running is fine)
# ---------------------------------------------------------------------------


def check_backend(url: str | None = None, timeout: float = 0.5) -> CheckResult:
    key, label = "backend", "Backend API"
    url = url or BACKEND_HEALTH_URL
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — local URL
            body = resp.read(4096)
            code = resp.status
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        return CheckResult(
            key,
            label,
            INFO,
            f"not reachable at {url} ({reason})",
            fix="Only needed for the companion app: "
            "uvicorn dance.api:create_app --factory --host 127.0.0.1 --port 8000",
        )
    try:
        ok = bool(json.loads(body.decode("utf-8")).get("ok"))
    except (ValueError, AttributeError):
        ok = False
    if code == 200 and ok:
        return CheckResult(key, label, OK, f"up at {url}")
    return CheckResult(
        key,
        label,
        WARN,
        f"{url} answered HTTP {code} without ok:true — something else on that port?",
    )


# ---------------------------------------------------------------------------
# Orchestration + rendering
# ---------------------------------------------------------------------------


def _guard(
    fn: Callable[[], CheckResult | list[CheckResult]], key: str, label: str
) -> list[CheckResult]:
    try:
        res = fn()
    except Exception as exc:  # noqa: BLE001 — one broken check must not hide the rest
        logger.debug("doctor check %s crashed", key, exc_info=True)
        return [CheckResult(key, label, WARN, f"check crashed: {type(exc).__name__}: {exc}")]
    return res if isinstance(res, list) else [res]


def run_doctor(
    settings: Any,
    *,
    apps_dirs: Sequence[Path] | None = None,
    abletonosc_dir: Path | None = None,
    hf_cache: Path | None = None,
    torch_hub: Path | None = None,
    backend_url: str | None = None,
) -> DoctorReport:
    checks: list[CheckResult] = []
    checks += _guard(
        lambda: check_db_migrations(settings.db_url), "db_migrations", "Database migrations"
    )
    checks += _guard(check_ffmpeg, "ffmpeg", "ffmpeg")
    checks += _guard(lambda: check_ingest_deps(settings), "ingest_deps", "Ingest deps")
    checks += _guard(lambda: check_live_installs(apps_dirs), "live", "Ableton Live")
    checks += _guard(lambda: check_abletonosc(abletonosc_dir), "abletonosc", "AbletonOSC")
    checks += _guard(
        lambda: check_clap_weights(settings.clap_model, hf_cache), "clap_weights", "CLAP weights"
    )
    checks += _guard(
        lambda: check_demucs_weights(settings.demucs_model, torch_hub),
        "demucs_weights",
        "Demucs weights",
    )
    checks += _guard(lambda: check_backend(backend_url), "backend", "Backend API")
    return DoctorReport(checks)


_STYLE = {OK: "green", INFO: "cyan", WARN: "yellow", FAIL: "bold red"}


def render_text(report: DoctorReport, console: Any) -> None:
    from rich.markup import escape

    width = max(len(c.label) for c in report.checks) if report.checks else 10
    for c in report.checks:
        tag = c.status.upper() if c.status in (WARN, FAIL) else c.status
        style = _STYLE.get(c.status, "")
        console.print(
            f"[{style}]{tag:<5}[/{style}] {escape(c.label):<{width}}  {escape(c.summary)}",
            highlight=False,
            soft_wrap=True,
        )
        for d in c.details:
            console.print(
                f"      {' ' * width}  [dim]· {escape(d)}[/dim]", highlight=False, soft_wrap=True
            )
        if c.fix and c.status != OK:
            pad = f"      {' ' * width}  "
            for i, line in enumerate(c.fix.splitlines()):
                lead = "[bold]fix:[/bold] " if i == 0 else "     "
                console.print(f"{pad}{lead}{escape(line)}", highlight=False, soft_wrap=True)
    n = report.counts()
    verdict = "[bold red]FAIL[/bold red]" if not report.ok else "[green]no failures[/green]"
    console.print(
        f"\n{len(report.checks)} checks: {n[OK]} ok, {n[INFO]} info, {n[WARN]} warn, "
        f"{n[FAIL]} fail — {verdict}",
        highlight=False,
    )


__all__ = [
    "FAIL",
    "INFO",
    "KNOWN_PATCHES",
    "OK",
    "WARN",
    "CheckResult",
    "DoctorReport",
    "check_abletonosc",
    "check_backend",
    "check_clap_weights",
    "check_db_migrations",
    "check_demucs_weights",
    "check_ffmpeg",
    "check_ingest_deps",
    "check_live_installs",
    "dance_client_usage",
    "dance_sent_addresses",
    "read_template_info",
    "render_text",
    "run_doctor",
    "scan_abletonosc_handlers",
]
