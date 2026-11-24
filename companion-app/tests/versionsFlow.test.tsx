import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SetEditor } from "../src/views/SetEditor";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  renderApp,
  songOn,
  type SeenRequest,
} from "./boothHarness";

const SET_ID = 1;
const ORIGINAL = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive - Club Mix" };
const UNRELATED = { track_id: 9, title: "Half Light" };
const SONG_COLUMN = "mix";

function rec(song: { track_id: number; title: string }, version: object = {}) {
  return {
    track_id: song.track_id,
    stem_file_id: null,
    track_title: song.title,
    track_artist: "Test Artist",
    duration_seconds: 190,
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: { embedding: 0.9 },
    reasons: ["close vibe"],
    version_of: null,
    version_of_title: null,
    version_label: null,
    ...version,
  };
}

const CLUB_MIX_AS_A_VERSION = rec(CLUB_MIX, {
  version_of: ORIGINAL.track_id,
  version_of_title: ORIGINAL.title,
  version_label: "Club Mix",
});

function planItem(song: { track_id: number; title: string }) {
  return { ...song, artist: "Test Artist", key_camelot: "8A", bpm: 128, floor_energy: 6 };
}

function track(song: { track_id: number; title: string }, family: number | null, others: number) {
  return {
    id: song.track_id,
    file_path: `/music/${song.track_id}.mp3`,
    title: song.title,
    artist: "Test Artist",
    duration_seconds: 240,
    state: "complete",
    analysis: null,
    tags: [],
    version_count: others,
    version_family: family,
  };
}

function aSetWithTheOriginalPlanned({ path, query, body }: SeenRequest): unknown {
  const plan = {
    set_id: SET_ID,
    queues: { drums: [], bass: [], vocals: [], other: [], song: [planItem(ORIGINAL)] },
  };
  if (path === "/api/v1/sets/active" || path === `/api/v1/sets/${SET_ID}`) {
    return {
      id: SET_ID,
      name: "Friday night",
      notes: null,
      is_active: true,
      created_at: "2026-11-01T00:00:00",
      updated_at: "2026-11-01T00:00:00",
      tracks: [],
    };
  }
  if (path === "/api/v1/sets") return [];
  if (path === `/api/v1/sets/${SET_ID}/plan` || path === `/api/v1/sets/${SET_ID}/plan/append`) {
    return plan;
  }
  if (path === `/api/v1/sets/${SET_ID}/plan-recs`) {
    const forTheSongColumn = query.includes("role=song");
    return {
      column: SONG_COLUMN,
      combo_size: 0,
      recs: forTheSongColumn ? [CLUB_MIX_AS_A_VERSION, rec(UNRELATED)] : [],
    };
  }
  if (path === "/api/v1/recommend/by-column") {
    const column = (body as { column: string }).column;
    return { column, combo_size: 0, recs: column === SONG_COLUMN ? [CLUB_MIX_AS_A_VERSION] : [] };
  }
  if (path === "/api/v1/ableton/decks") {
    return { columns: LIVE_TRACK_OF, cells: songOn(0, "a", ORIGINAL) };
  }
  if (path === "/api/v1/tracks/search") {
    return [
      track(ORIGINAL, ORIGINAL.track_id, 1),
      track(CLUB_MIX, ORIGINAL.track_id, 1),
      track(UNRELATED, null, 0),
    ];
  }
  return undefined;
}

function card(title: string): HTMLElement {
  return screen.getByRole("group", { name: title });
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("versions of a song", () => {
  it("lead the Set view's Song column and queue right after their original", async () => {
    backend = new BoothBackend(aSetWithTheOriginalPlanned);
    render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <SetEditor />
      </QueryClientProvider>,
    );

    await screen.findByRole("group", { name: CLUB_MIX.title });
    expect(within(card(CLUB_MIX.title)).getByText("version")).toBeInTheDocument();
    expect(
      within(card(CLUB_MIX.title)).getByText("Club Mix of Night Drive · 3:10"),
    ).toBeInTheDocument();
    expect(within(card(UNRELATED.title)).queryByText("version")).not.toBeInTheDocument();

    fireEvent.click(within(card(CLUB_MIX.title)).getByTitle("Add to plan"));
    fireEvent.click(within(card(UNRELATED.title)).getByTitle("Add to plan"));

    await waitFor(() =>
      expect(
        backend.sent("POST", `/api/v1/sets/${SET_ID}/plan/append`).map((sent) => sent.body),
      ).toEqual([
        { role: "song", track_id: CLUB_MIX.track_id, after_track_id: ORIGINAL.track_id },
        { role: "song", track_id: UNRELATED.track_id, after_track_id: null },
      ]),
    );
  });

  it("lead the Booth's Song column for the song on a deck, offered for the other deck", async () => {
    backend = new BoothBackend(aSetWithTheOriginalPlanned);
    renderApp();
    await screen.findByRole("button", { name: "Page 1" });

    backend.liveNowPlays({});

    const songColumnAsked = () =>
      backend
        .sent("POST", "/api/v1/recommend/by-column")
        .map((sent) => sent.body as { column: string; deck_track_ids: number[] })
        .filter((asked) => asked.column === SONG_COLUMN);
    await waitFor(() =>
      expect(songColumnAsked().at(-1)?.deck_track_ids).toEqual([ORIGINAL.track_id]),
    );
    expect(
      await screen.findByTitle("Load → Deck B, the deck beside its original"),
    ).toBeInTheDocument();
    const version = within(card(CLUB_MIX.title));
    expect(version.getByText("Club Mix of Night Drive · 3:10")).toBeInTheDocument();
    expect(version.getByTitle("Load → Deck A")).toBeInTheDocument();
  });

  it("sit together in the search palette", async () => {
    backend = new BoothBackend(aSetWithTheOriginalPlanned);
    renderApp();
    await screen.findByRole("button", { name: "Page 1" });

    fireEvent.keyDown(window, { key: "k", metaKey: true });
    fireEvent.change(await screen.findByRole("textbox"), { target: { value: "night" } });

    const original = await screen.findByRole("group", { name: `Search result: ${ORIGINAL.title}` });
    const clubMix = screen.getByRole("group", { name: `Search result: ${CLUB_MIX.title}` });
    const unrelated = screen.getByRole("group", { name: `Search result: ${UNRELATED.title}` });
    expect(within(original).getByText("2 versions")).toBeInTheDocument();
    expect(within(clubMix).getByText("version")).toBeInTheDocument();
    expect(within(unrelated).queryByText(/version/)).not.toBeInTheDocument();
  });
});
