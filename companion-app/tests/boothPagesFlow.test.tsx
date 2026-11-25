import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  BoothBackend,
  LIVE_TRACK_OF,
  deck,
  grid,
  renderApp,
  songOn,
  songOnDeck,
  type Cell,
} from "./boothHarness";

const ORIGINAL = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
const THIRD_SONG = { track_id: 9, title: "Half Light" };
const STEM_COLUMNS_PER_DECK = 4;
const PADS_WITHOUT_A_CUE = 1;

function preparedSetInLive(): Cell[] {
  const cells = [...songOn(0, "a", ORIGINAL), ...songOn(0, "b", CLUB_MIX), ...songOn(1, "a", THIRD_SONG)];
  const droppedInByHand = cells.find((cell) => cell.kind === "bass_b" && cell.scene_index === 3)!;
  Object.assign(droppedInByHand, { cue: null, cue_label: null, cue_silent: false });
  return cells;
}

let backend: BoothBackend;

beforeEach(() => {
  backend = new BoothBackend(({ path }) =>
    path.endsWith("/ableton/decks")
      ? { columns: LIVE_TRACK_OF, cells: preparedSetInLive() }
      : undefined,
  );
});

afterEach(() => {
  backend.restore();
});

describe("a prepared Set in the Booth", () => {
  it("shows one page of five cues at a time and plays a deck from the cue you pick", async () => {
    renderApp();

    const pageOne = await screen.findByRole("button", { name: "Page 1" });
    const pageTwo = screen.getByRole("button", { name: "Page 2" });
    expect(pageOne).toHaveAttribute("aria-pressed", "true");
    expect(pageTwo).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("button", { name: "Page 3" })).not.toBeInTheDocument();

    expect(within(grid()).getAllByText("Night Drive")).toHaveLength(STEM_COLUMNS_PER_DECK);
    expect(within(grid()).getAllByText("Night Drive (Club Mix)")).toHaveLength(
      STEM_COLUMNS_PER_DECK + PADS_WITHOUT_A_CUE,
    );
    expect(within(grid()).queryByText("Half Light")).not.toBeInTheDocument();
    expect(within(grid()).getAllByText("3 drop")).toHaveLength(8);
    expect(within(grid()).getAllByText("4 breakdown")).toHaveLength(7);
    expect(
      within(grid()).getByTitle("Vocals A: Night Drive — Test Artist — cue 1, intro, silent here — tap to fire"),
    ).toBeInTheDocument();
    expect(
      within(grid()).getByTitle("Bass B: Night Drive (Club Mix) — Test Artist — tap to fire"),
    ).toBeInTheDocument();

    expect(songOnDeck("A", "Night Drive")).not.toHaveLength(0);
    expect(songOnDeck("B", "Night Drive (Club Mix)")).not.toHaveLength(0);

    fireEvent.click(pageTwo);
    expect(pageTwo).toHaveAttribute("aria-pressed", "true");
    expect(within(grid()).getAllByText("Half Light")).toHaveLength(STEM_COLUMNS_PER_DECK);
    expect(within(grid()).queryByText("Night Drive")).not.toBeInTheDocument();
    expect(songOnDeck("A", "Half Light")).not.toHaveLength(0);
    expect(songOnDeck("B", "Night Drive (Club Mix)")).toHaveLength(0);
    expect(within(deck("B")).queryByRole("button", { name: /cue/i })).not.toBeInTheDocument();

    fireEvent.click(within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" }));
    await waitFor(() =>
      expect(backend.sent("POST", "/api/v1/ableton/transport/fire-deck/a/7")).toMatchObject([
        { query: "loaded_only=true" },
      ]),
    );

    backend.liveNowPlays({ "0": 2, "1": 2, "2": 2, "3": 2 });
    const pageOnePlaying = await screen.findByRole("button", { name: "Page 1, playing" });
    await waitFor(() => expect(pageOnePlaying).toHaveAttribute("aria-pressed", "true"));
    expect(screen.getByRole("button", { name: "Page 2" })).toHaveAttribute("aria-pressed", "false");
    expect(songOnDeck("A", "Night Drive")).not.toHaveLength(0);
    expect(songOnDeck("A", "Half Light")).toHaveLength(0);
    expect(within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(within(deck("A")).getByRole("button", { name: "Deck A cue 1, intro" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});
