from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dance.api import create_app
from dance.api.app import API_PREFIX as API
from dance.config import Settings
from dance.core.database import Track, TrackEmbedding, TrackState, now_utc
from dance.core.embedding_scheme import CLAP_EMBED_VERSION
from dance.core.serialization import encode_embedding
from tests.test_api import FakeAbletonBridge

WHAT_THE_QUERY_SOUNDS_LIKE = [1.0, 0.0, 0.0]
CLOSE_TO_THE_QUERY = [0.9, 0.1, 0.0]
FAR_FROM_THE_QUERY = [0.1, 0.9, 0.0]
READY = TrackState.COMPLETE.value


class TextEncoderStandIn:
    def encode_text(self, _query: str) -> np.ndarray:
        return np.asarray(WHAT_THE_QUERY_SOUNDS_LIKE, dtype=np.float32)


class Library:
    def __init__(self, session, make_track, clap_model: str) -> None:
        self.session = session
        self.make_track = make_track
        self.clap_model = clap_model

    def song(
        self,
        title: str,
        sounds_like: list[float],
        *,
        state: str = READY,
        copy_of: int | None = None,
    ) -> int:
        track: Track = self.make_track(title=title, state=state, duplicate_of=copy_of)
        self.session.add(
            TrackEmbedding(
                track_id=track.id,
                stem_file_id=None,
                model=self.clap_model,
                model_version=CLAP_EMBED_VERSION,
                dim=len(sounds_like),
                embedding=encode_embedding(np.asarray(sounds_like, dtype=np.float32)),
                created_at=now_utc(),
            )
        )
        self.session.commit()
        return int(track.id)


@pytest.fixture
def palette(session_factory, session, make_track, tmp_path):
    settings = Settings(
        library_dir=tmp_path / "library",
        stems_dir=tmp_path / "stems",
        data_dir=tmp_path / "data",
        als_output_dir=tmp_path / "sets",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, bridge=FakeAbletonBridge(), session_factory=session_factory)
    app.state.embedding_stage = TextEncoderStandIn()
    with TestClient(app) as client:
        yield client, Library(session, make_track, settings.clap_model)


def _found_by_vibe(client: TestClient, k: int) -> list[tuple[int, str]]:
    found = client.post(f"{API}/recommend/text", json={"query": "deep rolling bass", "k": k})
    assert found.status_code == 200, found.text
    return [(hit["track_id"], hit["title"]) for hit in found.json()]


def test_vibe_search_lists_a_song_once_when_the_library_holds_three_copies_of_it(palette):
    client, library = palette
    kept = library.song("Navi", CLOSE_TO_THE_QUERY)
    library.song("Navi", CLOSE_TO_THE_QUERY, copy_of=kept)
    library.song("Navi", CLOSE_TO_THE_QUERY, copy_of=kept)
    another_song = library.song("Shallow Water", FAR_FROM_THE_QUERY)

    assert _found_by_vibe(client, k=3) == [(kept, "Navi"), (another_song, "Shallow Water")]


def test_vibe_search_leaves_out_songs_that_are_not_ready_to_play(palette):
    client, library = palette
    ready = library.song("Shallow Water", FAR_FROM_THE_QUERY)
    library.song("Still In The Pipeline", CLOSE_TO_THE_QUERY, state=TrackState.SEPARATING.value)
    library.song("Waiting To Be Processed", CLOSE_TO_THE_QUERY, state=TrackState.PENDING.value)
    library.song("Failed On The Way", CLOSE_TO_THE_QUERY, state=TrackState.ERROR.value)

    assert _found_by_vibe(client, k=5) == [(ready, "Shallow Water")]
