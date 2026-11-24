import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  PADS_PER_PAGE,
  deck,
  renderApp,
  songOn,
  songOnDeck,
  type Cell,
  type SeenRequest,
} from "./boothHarness";

const ON_DECK_A = { track_id: 7, title: "Night Drive" };
const ON_DECK_B = { track_id: 8, title: "Night Drive (Club Mix)" };
const BASS_BORROWED_ON_DECK_B = { track_id: 9, title: "Half Light" };
const NEXT_PICK = { track_id: 10, title: "Low Tide" };
const SONG_COLUMN = "mix";
const SLOWER_THAN_LIVE_BPM = 124;
const BOTH_DECKS_ON_CUE_ONE = { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0 };
const SIDES = ["A", "B"] as const;

let backend: BoothBackend;
let inLive: Cell[];

function slowerThanLive(cells: Cell[]): Cell[] {
  return cells.map((cell) => ({ ...cell, bpm: SLOWER_THAN_LIVE_BPM }));
}

function withBorrowedBass(cells: Cell[]): Cell[] {
  return cells.map((cell) =>
    cell.kind === "bass_b" ? { ...cell, ...BASS_BORROWED_ON_DECK_B } : cell,
  );
}

function nextPickAsRecommendation() {
  return {
    track_id: NEXT_PICK.track_id,
    stem_file_id: null,
    track_title: NEXT_PICK.title,
    track_artist: "Test Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: { embedding: 0.9, bpm: 0.7 },
    reasons: ["close vibe"],
  };
}

function aSetWithEveryFxAuthored({ method, path, body }: SeenRequest): unknown {
  if (path.endsWith("/ableton/decks")) return { columns: LIVE_TRACK_OF, cells: inLive };
  if (path.endsWith("/ableton/fx/state")) {
    const nothingEngaged = { a: false, b: false };
    return {
      filter: nothingEngaged,
      reverb: nothingEngaged,
      delay: nothingEngaged,
      discovered: { returns: { filter: 10, reverb: 11, delay: 12, riser: 13 }, scenes: { riser: 0 } },
    };
  }
  if (path.endsWith(`/tracks/${ON_DECK_A.track_id}/waveform`)) {
    return { peaks: [0.2, 0.9, 0.4], duration_seconds: 240, num_peaks: 3 };
  }
  if (path.endsWith(`/tracks/${ON_DECK_A.track_id}/regions`)) {
    return [
      {
        id: 1,
        position_ms: 0,
        length_ms: 60_000,
        region_type: "section",
        section_label: "intro",
        length_bars: 32,
        name: null,
        color: null,
        confidence: 0.9,
        source: "auto",
        stem_file_id: null,
      },
    ];
  }
  if (path.endsWith("/recommend/by-column")) {
    const column = (body as { column: string }).column;
    return {
      column,
      combo_size: 0,
      recs: column === SONG_COLUMN ? [nextPickAsRecommendation()] : [],
    };
  }
  if (method === "POST" && path.endsWith("/ableton/load-track")) {
    inLive = [...inLive, ...songOn(1, "a", NEXT_PICK)];
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

async function bothDecksPlayingWithTheNextPickArmed(): Promise<void> {
  renderApp();
  await screen.findByRole("region", { name: "Deck A" });
  backend.liveNowPlays(BOTH_DECKS_ON_CUE_ONE);
  fireEvent.click(await screen.findByTitle("Load → Deck A"));
  await screen.findByTestId("armed-a");
  await within(deck("A")).findByRole("button", { name: "SYNC" });
  await within(deck("A")).findByTestId("waveform-section-icon");
  await screen.findByRole("button", { name: "Fire riser" });
}

function titleOn(side: "A" | "B", title: string): HTMLElement {
  return songOnDeck(side, title).find((shown) => !shown.closest("button"))!;
}

function charactersOfRoomKept(element: HTMLElement): number {
  const reserved = /(?:^|\s)min-w-\[(\d+)ch\]/.exec(element.className);
  return reserved ? Number(reserved[1]) : 0;
}

function expectToGiveWay(chip: HTMLElement): void {
  expect(chip).not.toHaveClass("shrink-0");
  expect(chip.className).toMatch(/(^|\s)min-w-/);
}

beforeEach(async () => {
  inLive = slowerThanLive([
    ...songOn(0, "a", ON_DECK_A),
    ...withBorrowedBass(songOn(0, "b", ON_DECK_B)),
  ]);
  backend = new BoothBackend(aSetWithEveryFxAuthored);
  await bothDecksPlayingWithTheNextPickArmed();
});

afterEach(() => {
  backend.restore();
  store.clearArm("a");
  store.clearDecks();
  store.setBoothPage(0);
});

describe("a deck panel in the middle of a set", () => {
  it("keeps room for the song title and lets the chips beside it give way first", () => {
    for (const [side, song] of [["A", ON_DECK_A], ["B", ON_DECK_B]] as const) {
      const panel = within(deck(side));
      expect(charactersOfRoomKept(titleOn(side, song.title))).toBeGreaterThanOrEqual(12);

      const artist = panel.getByText("Test Artist");
      const bpmKeyAndEnergy = panel.getByText(`${SLOWER_THAN_LIVE_BPM}`).parentElement!;
      const originalChip = panel.getByRole("button", { name: `Play original mix on Deck ${side}` });
      expect(bpmKeyAndEnergy).toHaveTextContent("· 8A");
      expect(bpmKeyAndEnergy).toHaveTextContent("· E6");
      for (const chip of [artist, bpmKeyAndEnergy, originalChip]) expectToGiveWay(chip);
    }
  });

  it("gives every control in the deck header and the cue row a target tall enough to hit", () => {
    for (const side of SIDES) {
      const panel = within(deck(side));
      const inHeader = [
        `Play Deck ${side}`,
        `Stop Deck ${side}`,
        `PFL Deck ${side}`,
        `HPF Deck ${side}`,
        `Reverb throw Deck ${side}`,
        `Delay throw Deck ${side}`,
        "SYNC",
        `Play original mix on Deck ${side}`,
      ].map((name) => panel.getByRole("button", { name }));
      const cues = panel.getAllByRole("button", { name: /^Deck [AB] cue \d/ });

      expect(cues).toHaveLength(PADS_PER_PAGE);
      expect(panel.getAllByRole("button")).toHaveLength(inHeader.length + cues.length);
      for (const headerControl of inHeader) expect(headerControl).toHaveClass("h-6");
      for (const cue of cues) expect(cue).toHaveClass("h-7");
    }
    expect(screen.getByRole("button", { name: "Fire riser" })).toHaveClass("h-6");
  });
});
