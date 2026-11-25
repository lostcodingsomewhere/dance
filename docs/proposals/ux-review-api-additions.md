# Two API additions from the UI review

**Status:** APPROVED 2026-11-24 (both additions, including the ingest change in section 2). Built the same day, server and app, and merged to `main`.
**Date:** 2026-11-24

The UI review of 2026-11-24 found two defects that cannot be fixed in the
companion app alone. Each needs fields added to an existing response. Nothing
here touches the database models, the Stage protocol, the `.als` writer or the
OSC contract.

## 1. A queued pick says whether it can be played

**What happens today.** A pick whose download is still running, or failed, looks
the same as a ready one. The Set view shows it with **Pads**; the Booth shows it
with ⤒A and ⤒B. Seen on 2026-11-24 with a version queued by **Get** whose
download had failed: the only hint was a missing BPM line.

**Proposed.** `PlanItemOut` (returned by `GET /sets/{id}/plan`, `PUT
/sets/{id}/plan` and `POST /sets/{id}/plan/append`) gains two fields:

| field | type | meaning |
|---|---|---|
| `state` | `str` | the track's pipeline state (`pending`, `complete`, `error`, …) |
| `download_failed` | `bool` | the audio never arrived; same meaning as on `SpotifyVersionOut` |

The card then says "on its way", "processing" or "download failed", and ⤒A, ⤒B
and **Pads** are offered only for a `complete` track.

**Alternatives.**
- The app asks `GET /tracks/{id}` for each pick. One request per pick on every
  plan change, and two answers that can disagree.
- A separate `GET /sets/{id}/plan/readiness`. A second request describing the
  first, with the same risk.

**Cost.** Two fields on one response model. No migration. A caller that ignores
them sees no change.

## 2. Spotify search says which songs the library already holds

**What happens today.** ⌘K lists a song the library holds under "Add from
Spotify" with a **+ Plan** button, unmarked. `POST /pipeline/ingest/track`
recognises a song only by its Spotify id. On a copy of the library taken
2026-11-24, 160 of the 244 playable songs (complete, not marked duplicate) carry
no Spotify id, so choosing one of them starts a second download.

**Proposed.**
- Each hit in `GET /spotify/search` gains `library_track_id: int | None` and
  `library_state: str | None`, the names `GET /spotify/versions` already uses.
  They are filled by the test find-versions uses today (`_is_the_file_of` in
  `src/dance/spotify/versions.py`): equal Spotify id, or the same song by the
  version-family rule with the same version text and a length inside the
  download matcher's gate.
- ⌘K shows such a row as "in your library", and its button adds the track the
  library holds.
- `POST /pipeline/ingest/track` applies the same test before it creates a row,
  and answers with the existing track (`already_existed: true`). This is the
  part that stops a second copy whatever the caller does.

**Decision inside this one.** The ingest change can refuse a song that is a
different master with the same title, artist and length. Find-versions already
relies on the same test to say "in your library". In the real-data check of
2026-11-24 it marked all 8 results the library held, across 14 songs; that
check did not look for a result marked as held by mistake. The alternative is to mark the row
in ⌘K only and leave ingest as it is; then a caller that does not look at the
mark still downloads a second copy. Recommended: both.

**Alternatives.** Matching in the app against `/tracks/search`. That would put
the version-family rule in TypeScript beside the Python one.

**Cost.** Two optional fields per hit. One read of the library per search, which
`GET /spotify/versions` already does. No migration.

## For the record: two new routes that do not need this proposal

Both follow existing patterns and are being built with the review's other
fixes.

- `GET /ableton/health` answers `{"live_reachable": bool}` from a short query to
  Live. The header's health light uses it, because `GET /ableton/state` returns
  a cache and stays green after Live quits.
- `POST /pipeline/tracks/{id}/retry` puts a failed track back in line, under
  the dispatcher lock. Today the Pipeline view can only delete a failed track.
