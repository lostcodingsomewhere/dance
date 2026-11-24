import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  LIVE_TRACK_OF,
  PADS_PER_PAGE,
  renderApp,
  songOn,
  songOnDeck,
  type BoothBackend,
  type Cell,
  type SeenRequest,
} from "./boothHarness";
import {
  CLUB_MIX,
  HALF_LIGHT,
  NIGHT_DRIVE,
  SessionLog,
  boothWith,
  decksThisBrowserLoaded,
  minutesAgo,
} from "./session-logBackend";

const SONG_COLUMN = "mix";

let backend: BoothBackend;
let inLive: Cell[];

function pageOneTaken(): Cell[] {
  return [...songOn(0, "a", NIGHT_DRIVE), ...songOn(0, "b", CLUB_MIX)];
}

function recommendation() {
  return {
    track_id: HALF_LIGHT.track_id,
    stem_file_id: null,
    track_title: HALF_LIGHT.title,
    track_artist: "Test Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: { embedding: 0.9, bpm: 0.7 },
    reasons: ["close vibe"],
  };
}

function oneRecommendationThatLoadsOntoPageTwo({ method, path, body }: SeenRequest): unknown {
  if (path.endsWith("/recommend/by-column")) {
    const column = (body as { column: string }).column;
    return { column, combo_size: 0, recs: column === SONG_COLUMN ? [recommendation()] : [] };
  }
  if (method === "POST" && path.endsWith("/ableton/load-track")) {
    inLive = [...inLive, ...songOn(1, "a", HALF_LIGHT)];
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

function liveSaysItsDecksChanged(): void {
  backend.liveNowPlays({});
}

beforeEach(() => {
  inLive = pageOneTaken();
  const log = new SessionLog();
  log.open(minutesAgo(10), [{ track_id: NIGHT_DRIVE.track_id, played_at: minutesAgo(5) }]);
  backend = boothWith(log, () => inLive, oneRecommendationThatLoadsOntoPageTwo);
});

afterEach(() => {
  backend.restore();
});

describe("a deck loaded from this browser", () => {
  it("is remembered while Live holds it and forgotten once Live does not", async () => {
    renderApp();
    fireEvent.click(await screen.findByTitle("Load → Deck A"));

    const pageTwo = await screen.findByRole("button", { name: "Page 2" });
    await waitFor(() => expect(pageTwo).toHaveAttribute("aria-pressed", "true"));
    expect(Object.values(decksThisBrowserLoaded())).toMatchObject([
      { track_id: HALF_LIGHT.track_id, scene_index: PADS_PER_PAGE, side: "a" },
    ]);

    inLive = [...pageOneTaken(), ...songOn(1, "b", CLUB_MIX)];
    liveSaysItsDecksChanged();

    await waitFor(() => expect(songOnDeck("B", CLUB_MIX.title)).not.toHaveLength(0));
    expect(songOnDeck("A", HALF_LIGHT.title)).toHaveLength(0);
    await waitFor(() => expect(Object.values(decksThisBrowserLoaded())).toEqual([]));
  });
});
