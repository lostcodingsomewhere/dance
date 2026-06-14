# Recommender "brain" port — isolated from the abandoned plan-grid UI

Status: **implemented 2026-06-14** (this branch). Verified on the real
353-track library, not just synthetic fixtures.

## Why

The `recommender-rework-plan-grid` branch (2026-06-05/06) built a major
recommender unification — a shared scoring core, a trend-aware "journey"
model, structural transition-fit, and a collapsed similarity graph — but
it was never merged. It rode together with a **product direction `main`
explicitly did not take**: kill the SetRail, replace per-column rec banners
with a "plan = per-role rec grid", and add a `Set.plan` JSON model with
three throwaway migrations (`set_moves` → `set_scenes` → `set_plan_queues`).

`main` kept the SetRail + `ColumnRecBanner`. So the *UI* direction is dead,
but the *brain* (better recs) is genuinely valuable and `main` still runs
the old per-recommender inline scoring. This port lifts **only the brain**
onto `main`, leaving every trace of the set-planning/UI direction behind.

## What ships (the brain)

Wholesale from the branch — all verified to have **zero** set-planning / UI
/ API imports:

- `recommender/scoring.py` (new) — one impl per signal (`embed/key/bpm/
  energy/kick_density/presence/timbre`), per-context weight `Profile`s, and
  `combine()` that renormalizes over present features only.
- `recommender/journey.py` (new) — `JourneyState` with **trend-aware vibe**
  (`normalize(target + β·trend)`, β=0.5): projects where the set is going
  rather than rewarding proximity. Plus projected-energy and shared loaders.
- `recommender/structure.py` (new) — `transition_fit` over intro/outro/
  phrase `Region`s (mix-out vs mix-in compatibility).
- `recommender/{recommender,tail,graph_builder}.py` (overwrite) — column +
  tail + seed recommenders ported onto the core; graph collapsed to
  embedding-kNN only.
- `api/routers/recommend.py` — seed `/recommend` simplified (drops the dead
  `kinds`/`weights` edge-blend params; scores live).
- Tests: `test_{scoring,journey,structure}.py` (new) +
  `test_{recommender,tail_recs}.py` (rewritten) + two seed-recommend tests
  and one e2e graph assertion updated for the collapse.

## What is left behind (the rejected direction)

`set_queues.py`, `Set.plan` + `PlanRole`, the `/sets/{id}/plan*` endpoints,
the `Plan*` Pydantic schemas, all three migrations, and the entire frontend
teardown (`SetRail`/`ColumnRecBanner` stay; `RoleColumn`/`useSetPlan` never
arrive). The only entanglement was `GET /sets/{id}/plan-recs`, which merely
*consumed* the portable `recommend_by_column` — dropping the endpoint leaves
the brain untouched.

## Contract impact (the rule-1 reason this is a proposal)

1. **`score_breakdown` keys change** from `{embedding,key,bpm[,energy]}` to
   the canonical feature set `{embedding,key,bpm,energy,kick_density,
   presence,timbre,transition_fit}` (only the present/weighted subset per
   rec). The field type (`dict[str,float]`) is unchanged. **Safe:** the
   frontend declares `score_breakdown` but never reads or renders it (it
   shows `score` and `reasons[]` only).
2. **Seed `/recommend` reasons** are still `list[dict]` (now `{kind,value}`
   instead of `{kind,from_seed,weight}`). `RecommendationOut.reasons` is
   `list[dict[str,Any]]` and the UI never reads it — no break.
3. **Graph collapse:** `dance build-graph` now materializes only
   `embedding_neighbor` edges. Key/BPM/energy/structure are scored **live**
   at query time, so nothing is lost — the precomputed harmonic/tempo/tag
   edges were redundant with live scoring. `EdgeKind.{HARMONIC,TEMPO,TAG}_*`
   stay defined (schema stability) but unbuilt. Existing DBs keep their old
   harmonic/tempo rows as harmless ignored cruft until the next rebuild;
   `DELETE FROM track_edges WHERE kind != 'embedding_neighbor'` tidies them.
4. **No migration.** Every column the brain reads already exists on `main`
   (`kick_density`, `presence_ratio`, `brightness`, `warmth`,
   `dominant_pitch_camelot`, the `*_confidence` fields, `Region.length_bars`).

## Behavior decisions

- **BPM tolerance ±20 → ±8 with halftime/doubletime folding** (70↔140 now
  match). Materially re-ranks recs — verified sane on real data.
- **Booth column-recs pass `trailing_track_ids=None`** for now: they gain
  the new features + halftime BPM + per-column profiles, but not the
  trend/anti-repetition (those need a trailing source). Set tail-recs get
  full journey internally from the set ordering. *Follow-up:* wire Booth
  trailing from session play-history for live trend-awareness.
- **Drums/other columns drop key-matching** (a drum stem has no meaningful
  key) and weight `kick_density` instead.

## Verification (real data, `/tmp` copy of `~/.dance/dance.db`, 353 tracks)

- `GraphBuilder.build()` → `{'embedding_neighbor': 10944}` only.
- seed `recommend()` → live-scored top rec 0.866 (embedding 0.919, key 1.0).
- `recommend_by_column('drums')` → `kick_density` live (0.775).
- `tail_recs_for_set` → full journey incl. `transition_fit` (0.41), 0.86–0.88.
- Backend suite: **427 passed**. ruff clean on changed files. mypy at parity
  with `main` (same pre-existing `Column[X]` SQLAlchemy fussiness).

## Follow-ups (not in this port)

- Surface `score_breakdown` in the UI (a "why this score" chip row) — the
  data is now rich; no renderer exists today.
- Wire Booth `trailing_track_ids` for live trend-awareness.
- Optional: purge stale non-embedding edges on existing DBs.
