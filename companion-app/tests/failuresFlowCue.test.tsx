import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PADS_PER_PAGE, deck, songOn } from "./boothHarness";
import {
  BoothMidSet,
  LIVE_IS_DOWN,
  RECOMMENDED,
  WARP_CHECK_DELAY_MS,
  expectWarningLines,
  isCommand,
  loadAnswer,
  recommended,
  refused,
} from "./failuresFlowBooth";

const CUE_TRACK_IN_LIVE = 10;
const NO_VOCALS = "No vocals stem available for track 9";
const BASS_AT_HALF_TEMPO = "Bass on Deck B is warped at half the tempo of the other stems.";
const BACKEND_DID_NOT_ANSWER = "The Dance backend did not answer. Check it is still running.";

const PREVIEW_PLAYING = {
  ok: true,
  cue_track_idx: CUE_TRACK_IN_LIVE,
  slot: 0,
  audio_path: "/music/half-light.mp3",
  label: "PREVIEW · Half Light (mix)",
  warnings: [],
};

let booth: BoothMidSet;

function startPreview(): void {
  fireEvent.click(within(recommended()).getByTitle("Preview in headphones (cue)"));
}

function cueStrip(): HTMLElement {
  return screen.getByTestId("cue-strip");
}

function stopButton(): HTMLElement {
  return within(cueStrip()).getByRole("button", { name: "⏹ stop" });
}

beforeEach(() => {
  booth = new BoothMidSet();
});

afterEach(async () => {
  await booth.close();
});

describe("the cue preview", () => {
  it("says so when a preview will not start", async () => {
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/preview") ? refused(503, LIVE_IS_DOWN) : undefined;
    await booth.open();

    startPreview();

    await expectWarningLines([`Couldn't start the preview: ${LIVE_IS_DOWN}`]);
    expect(screen.queryByTestId("cue-strip")).not.toBeInTheDocument();
  });

  it("says the backend is not answering when a preview gets no answer at all", async () => {
    booth.liveAnswers = (request) => {
      if (isCommand(request, "/ableton/preview")) throw new TypeError("Failed to fetch");
      return undefined;
    };
    await booth.open();

    startPreview();

    await expectWarningLines([`Couldn't start the preview: ${BACKEND_DID_NOT_ANSWER}`]);
    expect(screen.queryByTestId("cue-strip")).not.toBeInTheDocument();
  });

  it("keeps the cue strip when a preview will not stop, so it can be stopped again", async () => {
    let liveIsBack = false;
    booth.liveAnswers = (request) => {
      if (isCommand(request, "/ableton/preview")) return PREVIEW_PLAYING;
      if (isCommand(request, "/ableton/preview/stop")) {
        return liveIsBack ? { ok: true, warnings: [] } : refused(503, LIVE_IS_DOWN);
      }
      return undefined;
    };
    await booth.open();
    startPreview();
    await screen.findByTestId("cue-strip");

    fireEvent.click(stopButton());

    await expectWarningLines([`Couldn't stop the preview: ${LIVE_IS_DOWN}`]);
    expect(cueStrip()).toBeInTheDocument();

    liveIsBack = true;
    await waitFor(() => expect(stopButton()).toBeEnabled());
    fireEvent.click(stopButton());

    await waitFor(() => expect(screen.queryByTestId("cue-strip")).not.toBeInTheDocument());
    expect(booth.sent("POST", "/ableton/preview/stop")).toHaveLength(2);
  });

  it("loads from the cue strip the way a deck button does", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"], shouldAdvanceTime: true });
    booth.liveAnswers = (request) => {
      if (isCommand(request, "/ableton/preview")) return PREVIEW_PLAYING;
      if (isCommand(request, "/ableton/load-track")) {
        booth.inLive = [...booth.inLive, ...songOn(1, "b", RECOMMENDED)];
        return loadAnswer({ page: 1, side: "b", stems: 3, warnings: [NO_VOCALS] });
      }
      if (isCommand(request, `/ableton/warp-check/${PADS_PER_PAGE}`)) {
        return { ok: true, scene_index: PADS_PER_PAGE, checked: 3, warnings: [BASS_AT_HALF_TEMPO] };
      }
      return undefined;
    };
    await booth.open();
    startPreview();
    await screen.findByTestId("cue-strip");
    const loadToMaster = within(cueStrip()).getByRole("button", { name: "→ Load song to master" });
    await waitFor(() => expect(loadToMaster).toBeEnabled());

    fireEvent.click(loadToMaster);

    await waitFor(() =>
      expect(booth.sent("POST", "/ableton/load-track")).toMatchObject([
        { body: { track_id: RECOMMENDED.track_id, kinds: null, side: null, page: 0 } },
      ]),
    );
    await expectWarningLines([`Half Light → Deck B: ${NO_VOCALS}`]);
    expect(await within(deck("B")).findByText(`armed · ${PADS_PER_PAGE + 1}`)).toBeInTheDocument();
    const pageTwo = await screen.findByRole("button", { name: "Page 2" });
    await waitFor(() => expect(pageTwo).toHaveAttribute("aria-pressed", "true"));
    await waitFor(() => expect(booth.sent("POST", "/ableton/preview/stop")).toHaveLength(1));

    act(() => {
      vi.advanceTimersByTime(WARP_CHECK_DELAY_MS);
    });
    await expectWarningLines([
      `Half Light → Deck B: ${BASS_AT_HALF_TEMPO}`,
      `Half Light → Deck B: ${NO_VOCALS}`,
    ]);
  });
});
