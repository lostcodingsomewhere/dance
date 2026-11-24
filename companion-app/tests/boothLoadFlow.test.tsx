import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  BoothBackend,
  LIVE_TRACK_OF,
  PADS_PER_PAGE,
  deck,
  grid,
  renderApp,
  songOn,
  type Cell,
  type SeenRequest,
} from "./boothHarness";

const ORIGINAL = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
const RECOMMENDED = { track_id: 9, title: "Half Light" };
const SONG_COLUMN = "mix";

let backend: BoothBackend;
let inLive: Cell[];

function recommendation() {
  return {
    track_id: RECOMMENDED.track_id,
    stem_file_id: null,
    track_title: RECOMMENDED.title,
    track_artist: "Test Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: { embedding: 0.9, bpm: 0.7 },
    reasons: ["close vibe"],
  };
}

function liveWithBothDecksOfPageOneTaken({ method, path, body }: SeenRequest): unknown {
  if (path.endsWith("/ableton/decks")) return { columns: LIVE_TRACK_OF, cells: inLive };
  if (path.endsWith("/recommend/by-column")) {
    const column = (body as { column: string }).column;
    return { column, combo_size: 0, recs: column === SONG_COLUMN ? [recommendation()] : [] };
  }
  if (method === "POST" && path.endsWith("/ableton/load-track")) {
    inLive = [...inLive, ...songOn(1, "a", RECOMMENDED)];
    return {
      ok: true,
      scene_index: PADS_PER_PAGE,
      page: 1,
      side: "a",
      track_indices: LIVE_TRACK_OF,
      stems_loaded: 4,
      message: null,
      warnings: [],
    };
  }
  return undefined;
}

beforeEach(() => {
  inLive = [...songOn(0, "a", ORIGINAL), ...songOn(0, "b", CLUB_MIX)];
  backend = new BoothBackend(liveWithBothDecksOfPageOneTaken);
});

afterEach(() => {
  backend.restore();
});

describe("loading a song from the Booth during a set", () => {
  it("asks for the page on screen, follows the load to its page, and plays it from a cue", async () => {
    renderApp();
    expect(await screen.findByRole("button", { name: "Page 1" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.queryByRole("button", { name: "Page 2" })).not.toBeInTheDocument();

    backend.liveNowPlays({});
    fireEvent.click(await screen.findByTitle("Load → Deck A"));

    await waitFor(() =>
      expect(backend.sent("POST", "/api/v1/ableton/load-track")).toMatchObject([
        { body: { track_id: RECOMMENDED.track_id, kinds: null, side: "a", page: 0 } },
      ]),
    );

    const pageTwo = await screen.findByRole("button", { name: "Page 2" });
    await waitFor(() => expect(pageTwo).toHaveAttribute("aria-pressed", "true"));
    expect(within(grid()).getAllByText("Half Light")).toHaveLength(4 * PADS_PER_PAGE);
    expect(within(grid()).queryByText("Night Drive")).not.toBeInTheDocument();
    expect(within(grid()).getAllByText("3 drop")).toHaveLength(4);

    fireEvent.click(within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" }));
    await waitFor(() =>
      expect(backend.sent("POST", "/api/v1/ableton/transport/fire-deck/a/7")).toMatchObject([
        { query: "loaded_only=true" },
      ]),
    );
  });
});
