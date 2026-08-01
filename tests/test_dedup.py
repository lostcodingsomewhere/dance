"""Duplicate-recording collapse in rec lists.

The library was built over three ingest runs that pulled overlapping songs
under different filenames, so ``file_hash`` differed and ingest's dedup never
fired. Measured on the real library: 82 groups, 145 redundant tracks of 353
(41%), and 5 of 40 rec slots across the five roles were a literal repeat of
another card in the same list.

The distinction these tests defend: collapse the SAME RECORDING, never a
genuine extended-vs-radio variant.
"""

from __future__ import annotations

from dataclasses import dataclass

from dance.core.database import Track
from dance.recommender.dedup import dedupe_by_recording, title_key


@dataclass
class _Rec:
    track_id: int
    score: float = 0.0


def _track(session, tid: int, title: str, artist: str, dur: float) -> None:
    session.add(
        Track(
            id=tid,
            file_hash=f"hash{tid}",
            file_path=f"/tmp/{tid}.mp3",
            file_name=f"{tid}.mp3",
            file_size_bytes=1,
            title=title,
            artist=artist,
            duration_seconds=dur,
            state="complete",
        )
    )


# --- key normalisation -----------------------------------------------------


def test_title_key_ignores_mix_name_noise():
    """Same recording, one source appended '(Original Mix)'."""
    assert title_key("JOY (In Me All The Time) (Original Mix)") == title_key(
        "JOY (In Me All The Time)"
    )


def test_title_key_keeps_genuinely_different_songs_apart():
    assert title_key("Strobe") != title_key("Ghosts Again")


def test_title_key_ignores_artist_credit_variation():
    """The real reason artist is not in the key: one release credited three
    ways across three ingest runs. Keying on artist collapsed only 2 of 3."""
    assert title_key("Navi - Kitty Amor Remix") == title_key("Navi - Kitty Amor Remix")


# --- collapse behaviour ----------------------------------------------------


def test_collapses_same_recording_keeping_the_better_scored(session):
    """Two ingests of one song: the higher-scored copy survives, and the freed
    slot is available for a real alternative."""
    _track(session, 1, "Navi - Kitty Amor Remix", "Dot Major;Kitty Amor", 211.8)
    _track(session, 2, "Navi - Kitty Amor Remix", "Dot Major, Kitty Amor", 211.8)
    _track(session, 3, "Strobe", "deadmau5", 300.0)
    session.commit()

    # Caller passes an already-sorted list, so #1 outranks #2.
    out = dedupe_by_recording(session, [_Rec(1, 0.9), _Rec(2, 0.8), _Rec(3, 0.7)])
    assert [r.track_id for r in out] == [1, 3]


def test_does_not_collapse_extended_vs_radio(session):
    """``Manifesto`` is 348s extended and 224s radio — different tools for
    different moments. Hiding the extended mix behind a radio edit that scored
    higher would be worse than showing two cards."""
    _track(session, 1, "Manifesto (Extended Mix)", "MRAK", 348.4)
    _track(session, 2, "Manifesto", "MRAK", 223.8)
    session.commit()

    out = dedupe_by_recording(session, [_Rec(1, 0.9), _Rec(2, 0.8)])
    assert [r.track_id for r in out] == [1, 2]


def test_three_way_duplicate_collapses_to_one(session):
    """#67 / #246 / #374 in the real library are the same 211.8s recording."""
    for tid, artist in (
        (67, "Dot Major;Kitty Amor"),
        (246, "Dot Major, Kitty Amor"),
        (374, "Dot Major"),
    ):
        _track(session, tid, "Navi - Kitty Amor Remix", artist, 211.8)
    session.commit()

    out = dedupe_by_recording(session, [_Rec(67, 0.9), _Rec(246, 0.85), _Rec(374, 0.8)])
    assert [r.track_id for r in out] == [67]


def test_never_collapses_on_missing_metadata(session):
    """With no title we cannot claim two things are the same recording.
    Showing one card too many beats dropping a candidate on a guess."""
    _track(session, 1, "", "", 200.0)
    _track(session, 2, "", "", 200.0)
    session.commit()

    out = dedupe_by_recording(session, [_Rec(1, 0.9), _Rec(2, 0.8)])
    assert [r.track_id for r in out] == [1, 2]


def test_unknown_track_ids_pass_through(session):
    """A result whose track row is gone must not vanish silently."""
    out = dedupe_by_recording(session, [_Rec(999, 0.5)])
    assert [r.track_id for r in out] == [999]


def test_empty_list_is_a_noop(session):
    assert dedupe_by_recording(session, []) == []


def test_order_is_preserved(session):
    _track(session, 1, "A", "X", 100.0)
    _track(session, 2, "B", "X", 100.0)
    _track(session, 3, "C", "X", 100.0)
    session.commit()
    out = dedupe_by_recording(session, [_Rec(3), _Rec(1), _Rec(2)])
    assert [r.track_id for r in out] == [3, 1, 2]
