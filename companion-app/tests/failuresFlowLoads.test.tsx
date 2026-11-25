import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LIVE_TRACK_OF, PADS_PER_PAGE, deck, songOn, songOnDeck } from "./boothHarness";
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

const ONLY_THE_SONG = "No stems were loaded, only the song.";

function sessionUnderWay() {
  return {
    id: SESSION_ID,
    name: null,
    notes: null,
    started_at: new Date().toISOString(),
    ended_at: null,
    plays: [],
  };
}

function songColumnOn(page: number, side: "a" | "b", song: { track_id: number; title: string }) {
  return songOn(page, side, song).filter((cell) => cell.kind === `mix_${side}`);
}

let booth: BoothMidSet;

beforeEach(() => {
  booth = new BoothMidSet();
});

afterEach(async () => {
  await booth.close();
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

  it("logs the song that was already on the deck, not the one that failed to load, when the deck plays", async () => {
    booth.liveAnswers = (request) => {
      if (request.path.endsWith("/sessions/current")) return sessionUnderWay();
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

    await waitFor(() =>
      expect(booth.sent("POST", `/sessions/${SESSION_ID}/plays`).map((sent) => sent.body)).toEqual([
        { track_id: ORIGINAL.track_id },
      ]),
    );
    await expectWarningLines([`Couldn't load Half Light onto Deck A: ${LIVE_DID_NOT_ANSWER}`]);
  });
});

describe("a whole-song load of a song that has no stems", () => {
  it("arms the deck, turns to its page, logs the play and says only the song was loaded", async () => {
    booth.liveAnswers = (request) => {
      if (request.path.endsWith("/sessions/current")) return sessionUnderWay();
      if (isCommand(request, "/ableton/load-track")) {
        booth.inLive = [...booth.inLive, ...songColumnOn(1, "a", RECOMMENDED)];
        return loadAnswer({ page: 1, side: "a", stems: 0, warnings: NO_STEMS_ON_DISK });
      }
      return undefined;
    };
    await booth.open();

    loadOnto("A");

    expect(await within(deck("A")).findByText(`armed · ${PADS_PER_PAGE + 1}`)).toBeInTheDocument();
    const pageTwo = await screen.findByRole("button", { name: "Page 2" });
    await waitFor(() => expect(pageTwo).toHaveAttribute("aria-pressed", "true"));
    await expectWarningLines(["Half Light → Deck A"]);
    expect(warningLines()[0]).toHaveTextContent(ONLY_THE_SONG);
    for (const missing of NO_STEMS_ON_DISK) expect(warningLines()[0]).toHaveTextContent(missing);
    expect(warningLines()[0]).not.toHaveTextContent("Couldn't load");

    booth.liveNowPlays({ [LIVE_TRACK_OF.mix_a]: PADS_PER_PAGE });
    await waitFor(() =>
      expect(booth.sent("POST", `/sessions/${SESSION_ID}/plays`).map((sent) => sent.body)).toEqual([
        { track_id: RECOMMENDED.track_id },
      ]),
    );
  });

  it("still says the load failed when the song did not reach Live either", async () => {
    const noMixFile = "mix file missing on disk: '/music/half-light.mp3'";
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/load-track")
        ? loadAnswer({ page: 1, side: "a", stems: 0, warnings: [...NO_STEMS_ON_DISK, noMixFile] })
        : undefined;
    await booth.open();

    loadOnto("A");

    await expectWarningLines(["Couldn't load Half Light onto Deck A"]);
    expect(warningLines()[0]).toHaveTextContent(noMixFile);
    expect(warningLines()[0]).not.toHaveTextContent(ONLY_THE_SONG);
    expect(within(deck("A")).queryByText(/armed/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Page 1" })).toHaveAttribute("aria-pressed", "true");
  });

  it("says the load failed when the song was already on that deck and nothing new reached Live", async () => {
    booth.inLive = [...booth.inLive, ...songColumnOn(1, "a", RECOMMENDED)];
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/load-track")
        ? loadAnswer({ page: 1, side: "a", stems: 0, warnings: [LIVE_DID_NOT_ANSWER] })
        : undefined;
    await booth.open();
    await screen.findByRole("button", { name: "Page 2" });

    loadOnto("A");

    await expectWarningLines([`Couldn't load Half Light onto Deck A: ${LIVE_DID_NOT_ANSWER}`]);
    expect(within(deck("A")).queryByText(/armed/)).not.toBeInTheDocument();
  });
});
