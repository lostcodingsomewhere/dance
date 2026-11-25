import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PADS_PER_PAGE, deck, songOnDeck } from "./boothHarness";
import {
  BoothMidSet,
  ORIGINAL,
  RECOMMENDED,
  WARP_CHECK_DELAY_MS,
  expectWarningLines,
  isCommand,
  loadAnswer,
  loadButton,
  loadOnto,
  press,
  warningLines,
} from "./failuresFlowBooth";

const SESSION_ID = 3;
const LIVE_DID_NOT_ANSWER = "Live did not answer, so nothing was loaded.";
const NO_STEMS_ON_DISK = ["drums", "bass", "vocals", "other"].map(
  (stem) => `No ${stem} stem available for track ${RECOMMENDED.track_id}`,
);

let booth: BoothMidSet;

beforeEach(() => {
  booth = new BoothMidSet();
});

afterEach(() => {
  booth.close();
});

describe("a deck load that loads nothing", () => {
  it("says so, leaves the deck unarmed and the page where it was, and asks for no warp check", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"], shouldAdvanceTime: true });
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/load-track")
        ? loadAnswer({ page: 1, side: "a", stems: 0, warnings: NO_STEMS_ON_DISK })
        : undefined;
    await booth.open();

    loadOnto("A");
    await waitFor(() => expect(booth.sent("POST", "/ableton/load-track")).toHaveLength(1));
    await waitFor(() => expect(loadButton("A")).toBeEnabled());

    expect(within(deck("A")).queryByText(/armed/)).not.toBeInTheDocument();
    expect(songOnDeck("A", ORIGINAL.title)).not.toHaveLength(0);
    await expectWarningLines(["Couldn't load Half Light onto Deck A"]);
    for (const missing of NO_STEMS_ON_DISK) expect(warningLines()[0]).toHaveTextContent(missing);

    act(() => {
      vi.advanceTimersByTime(WARP_CHECK_DELAY_MS);
    });
    expect(booth.sent("POST", `/ableton/warp-check/${PADS_PER_PAGE}`)).toEqual([]);
  });

  it("does not log the song as played when the deck then plays what was already on it", async () => {
    booth.liveAnswers = (request) => {
      if (request.path.endsWith("/sessions/current")) {
        return {
          id: SESSION_ID,
          name: null,
          notes: null,
          started_at: new Date().toISOString(),
          ended_at: null,
          plays: [],
        };
      }
      if (isCommand(request, "/ableton/load-track")) {
        return loadAnswer({ page: 0, side: "a", stems: 0, warnings: [LIVE_DID_NOT_ANSWER] });
      }
      return undefined;
    };
    await booth.open();

    loadOnto("A");
    await waitFor(() => expect(booth.sent("POST", "/ableton/load-track")).toHaveLength(1));
    await waitFor(() => expect(loadButton("A")).toBeEnabled());
    booth.liveNowPlays({ "0": 0, "1": 0, "2": 0, "3": 0 });
    await screen.findByRole("button", { name: "Page 1, playing" });
    press("Stop Deck A");
    await waitFor(() =>
      expect(booth.sent("POST", "/ableton/transport/stop-deck/a")).toHaveLength(1),
    );

    expect(booth.sent("POST", `/sessions/${SESSION_ID}/plays`)).toEqual([]);
    await expectWarningLines([`Couldn't load Half Light onto Deck A: ${LIVE_DID_NOT_ANSWER}`]);
  });
});
