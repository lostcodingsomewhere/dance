# Development

How to work on this codebase.

## Setup

```bash
git clone <repo>
cd dance

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cd companion-app
npm install
cd -
```

Python 3.10+ required (`pyproject.toml:6`). The optional `[dev]` extras add `pytest`, `pytest-cov`, `ruff`, `mypy`, and an `httpx` pin that matches Starlette's `TestClient`.

Once installed, the `dance` CLI is on your `PATH`:

```bash
dance --version
```

## Running tests

Python — `tests/`:

```bash
source .venv/bin/activate
pytest                  # full suite (500+ tests, ~2 min)
pytest -k stage         # subset
pytest -x --pdb         # stop at first failure, drop into pdb
pytest --cov=dance      # coverage report
```

The suite uses **synthetic audio fixtures** — no audio files are committed and no downloads happen during tests. See `tests/audio_fixtures.py:24` for `TrackSpec` (default 128 BPM, 8 bars, 44.1 kHz mono). Stage tests build a few seconds of synthetic 4-on-the-floor with separable layers so even simple analyzers produce plausible results.

React — `companion-app/`:

```bash
cd companion-app
npm test                # vitest
npm test -- --watch
```

There are 11 vitest files under `companion-app/tests/` (`smoke.test.tsx` plus SceneGrid, RoleColumnsGrid, BoothColumnHeaders, Waveform, preview, seek, gridNav, gridKeys, deckArm, deckRegistration). Add one next to them as you add components.

## Adding a new pipeline stage

`src/dance/pipeline/stage.py:24` defines the protocol. Steps:

1. Create `src/dance/pipeline/stages/my_stage.py`:

   ```python
   from dance.core.database import Track, TrackState

   class MyStage:
       name = "my_stage"
       input_state = TrackState.COMPLETE
       output_state = TrackState.COMPLETE   # or a new state you add
       error_state = TrackState.ERROR

       def process(self, session, track, settings) -> bool:
           # do work, write rows, set state, commit
           track.state = self.output_state.value
           session.commit()
           return True
   ```

2. Register it. Either add a `register()` call in `Dispatcher._register_default_stages()` (`src/dance/pipeline/dispatcher.py:68`) — wrap in `try/except ImportError` if it has optional deps — or register dynamically:

   ```python
   dispatcher = Dispatcher(settings, session)
   dispatcher.register(MyStage())
   ```

3. If you added a new `TrackState`, add it to the enum in `src/dance/core/database.py:68` and write an Alembic migration (`alembic revision -m "..."` — see below).

4. Add a test under `tests/test_my_stage.py`. Pattern: copy `tests/test_embed_stage.py`. Use the synthetic fixtures.

The dispatcher will pick up your stage automatically — no other file changes needed. If `input_state == output_state`, the dispatcher's outer loop will run forever on that stage; bound the work in `process()` itself.

## Adding a new API endpoint

Routers live in `src/dance/api/routers/`. Each file is a `APIRouter(prefix=..., tags=[...])` exposed via `router` and wired in `src/dance/api/app.py:93`.

Pattern:

```python
# src/dance/api/routers/myroutes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from dance.api.deps import get_session, get_settings
from dance.api.schemas import MyResponseSchema   # add to schemas.py
from dance.config import Settings

router = APIRouter(prefix="/things", tags=["things"])

@router.get("/{id}", response_model=MyResponseSchema)
def get_thing(
    id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    ...
```

Then `app.py`:

```python
from dance.api.routers import myroutes
app.include_router(myroutes.router, prefix=API_PREFIX)
```

Dependencies (`get_session`, `get_bridge`, `get_settings`) are in `src/dance/api/deps.py`. They read from `app.state.*` which is set in `create_app`. Tests can override via `app.dependency_overrides[get_session] = ...`.

Test it: copy a test from `tests/test_api.py` and use `TestClient`.

## Migrations

SQLAlchemy declarative models live in `src/dance/core/database.py`. We use Alembic for schema changes.

Generate a migration after editing the models:

```bash
alembic revision --autogenerate -m "add my_column to tracks"
# review the generated file in src/dance/alembic/versions/
alembic upgrade head
```

Apply to an existing DB:

```bash
dance doctor            # read-only: is the DB behind? which revisions are pending?
alembic upgrade head
```

Existing migrations: `src/dance/alembic/versions/` (one file per revision; `alembic history` prints the chain).

Caveats:

- The two partial unique indexes for `audio_analysis` (`database.py:721`) are created via raw DDL in `init_db()` because SQLAlchemy can't model `UNIQUE ... WHERE` portably. Autogenerate won't see them; don't be surprised when they don't show up in a diff.
- `tests/test_alembic.py` walks every migration up and down on a temp DB. Run it after any migration change.

## Pre-commit hygiene

```bash
ruff check src/dance tests          # lint
ruff format src/dance tests         # format
mypy src/dance                      # type-check
pytest -q                           # test
```

Config in `pyproject.toml`:

- Ruff line length 100, target py310, rules `E F I N W UP` minus `E501`.
- mypy strict-ish (`warn_return_any`, `warn_unused_ignores`).

Run all four before pushing. There's no pre-commit hook configured by default — set one up locally if you want.

## Synthetic audio fixture

`tests/audio_fixtures.py`. Two-line summary:

```python
from tests.audio_fixtures import TrackSpec, write_track

spec = TrackSpec(bpm=128.0, bars=8)
wav_path = write_track(tmp_path / "test.wav", spec)
```

`write_track` writes a 44.1 kHz mono WAV with separable layers: a four-on-the-floor kick, a bass note per bar, a high-hat pattern, and a sustained pad. Demucs separates them cleanly enough for the per-stem analyzers to produce non-zero RMS, and BPM detection lands within ~0.5 of the spec value.

Use it any time you need real audio without committing real audio. Keep durations short — 8 bars at 128 BPM is ~15 s and the analyze + separate stages take ~2 s on M1.

## Beat-grid differential (the grid gate)

Any change to how beats / BPM are computed is judged against **Ableton's** grid (rule 3), with djbooth's method: monotonic DP alignment (gap = 0.45 × median period), F1 at 70/100 ms, median/p95 timing error, phase, drift per 100 beats, BPM error (analysis BPM *and* the `beats` table's own tempo), bar-phase accuracy. Metrics: `src/dance/pipeline/utils/grid_metrics.py`; references: `src/dance/als/reader.py` (warp markers out of a `.als`) and `src/dance/osc/warp_grid.py` (live, shared with `scripts/warp_probe.py`).

```bash
# reference = a Live Set saved after fixing warps by ear (or --live --scene N)
python scripts/grid_diff.py --als fixed.als --json base.json
# ... change the grid code, reprocess ...
python scripts/grid_diff.py --als fixed.als --json new.json --baseline base.json   # exit 1 = regression
```

The DB is opened read-only and never created (`file:…?mode=ro`, plus `immutable=1` only when its `-wal` is empty, so a running backend can't hide fresh beats). Ship a grid change only when the gate exits 0. Every track the baseline measured is judged again (a reference that stopped matching is a `grid_missing` regression; `--track` narrows on purpose), every metric that worsens past its tolerance fails, and so does a track measured against a different reference than its baseline (`reference_changed`: other warp markers, clip kind or sample end; `--allow-reference-change` demotes it). One waiver: on a track that already fails, when the beats table's own grid-tempo error shrank past its tolerance and by at least a fifth in the same run (drift alone never triggers it), its F1 / timing / bar-phase drops are printed as warnings, because once a grid slips past half a beat those numbers swing with the phase sawtooth (a 0.5 % → 0.1 % tempo fix can lower F1). Without a tempo change nothing is waived; `--strict` waives nothing at all. Stem clips are judged against that stem's own `audio_analysis.bpm`, and a NULL BPM fails rather than falling back. Live's grid is clipped to the span of dance's beats, and how much of it is covered is reported as `reference_coverage`; its end is the sample's real length (Set or file header), with `tracks.duration_seconds` only as a flagged fallback. Two same-kind clips with different warps make a track `ambiguous_reference` until `--ref-source` picks one. A dance-exported `.als` works as a reference too, but it only encodes `audio_analysis.bpm` from 0:00 — it is a plumbing check, not ground truth.

## Repo layout (developer cheat-sheet)

```
src/dance/
  cli.py                  Click commands
  config.py               pydantic-settings
  core/database.py        SQLAlchemy models (one file by design)
  core/serialization.py   embedding pack/unpack
  pipeline/
    dispatcher.py         Stage runner
    stage.py              Stage Protocol
    events.py             EventBus
    stages/               one file per stage
    utils/                shared (audio, db, device, beats, camelot)
  spotify/downloader.py
  recommender/
  als/                    writer, generator, markers, templates/blank_live12.xml
  osc/                    client, listener, bridge
  llm/                    tagger, qwen_audio, brief
  api/
    app.py                create_app factory
    deps.py               DI providers
    schemas.py            Pydantic response models
    routers/              one file per resource
  alembic/                migrations

companion-app/src/
  App.tsx, main.tsx, store.ts, api.ts, types.ts
  components/             TrackCard, EnergyBar, KeyBadge, ...
  hooks/                  useTracks, useRecommend, useSession, useAbletonState
  views/                  NowPlaying, UpNext, Library, SessionHistory

tests/                    500+ Python tests, all use synthetic audio
```
