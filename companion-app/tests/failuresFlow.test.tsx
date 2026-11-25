import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PADS_PER_PAGE, deck, grid } from "./boothHarness";
import {
  BoothMidSet,
  LIVE_IS_DOWN,
  WARP_CHECK_DELAY_MS,
  everyCommandFails,
  expectWarningLines,
  isCommand,
  loadAnswer,
  loadButton,
  loadOnto,
  press,
  refused,
  warningLines,
  warnings,
} from "./failuresFlowBooth";

const NO_DECK_COLUMNS = "no deck columns staged yet";
const NO_REVERB_RETURN = "No 'Reverb' return track in Live.";
const NO_ARMED_RECORDER =
  "Could not confirm an armed Recorder track — Live may capture nothing. Check Live is reachable, then retry.";
const NO_VOCALS = "No vocals stem available for track 9";
const NO_MIX_FILE = "mix file missing on disk: '/music/half-light.mp3'";
const BACKEND_DID_NOT_ANSWER = "The Dance backend did not answer. Check it is still running.";
const BASS_AT_HALF_TEMPO = "Bass on Deck B is warped at half the tempo of the other stems.";

let booth: BoothMidSet;

beforeEach(() => {
  booth = new BoothMidSet();
});

afterEach(async () => {
  await booth.close();
});

const BOOTH_COMMANDS = [
  {
    command: "loading a song onto a deck",
    press: () => loadOnto("A"),
    sends: ["POST", "/ableton/load-track"],
    says: "Couldn't load Half Light onto Deck A",
  },
  {
    command: "playing a clip from the grid",
    press: () =>
      fireEvent.click(
        within(grid()).getByTitle("Drums A: Night Drive — Test Artist — cue 1, intro — tap to fire"),
      ),
    sends: ["POST", "/ableton/transport/fire-clip/0/0"],
    says: "Couldn't play that clip",
  },
  {
    command: "stopping a clip from the grid",
    press: async () => {
      booth.liveNowPlays({ "0": 0 });
      fireEvent.click(await within(grid()).findByTitle("Stop drums a (Night Drive)"));
    },
    sends: ["POST", "/ableton/transport/stop-cell/0/0"],
    says: "Couldn't stop that clip",
  },
  {
    command: "removing a clip from the grid",
    press: () => {
      fireEvent.click(
        within(grid()).getAllByRole("button", { name: "Remove Drums clip from scene" })[0],
      );
      fireEvent.click(
        within(grid()).getByRole("button", { name: "Press again to remove Drums clip from scene" }),
      );
    },
    sends: ["DELETE", "/ableton/decks/cell/0/0"],
    says: "Couldn't remove that clip",
  },
  {
    command: "soloing a column",
    press: () => press("solo drums_a"),
    sends: ["POST", "/ableton/transport/solo-track/0"],
    says: "Couldn't solo that column",
  },
  {
    command: "playing a deck",
    press: () => press("Play Deck A"),
    sends: ["POST", "/ableton/transport/fire-deck/a/0"],
    says: "Couldn't play Deck A",
  },
  {
    command: "playing a deck from a cue",
    press: () => press("Deck B cue 3, drop"),
    sends: ["POST", "/ableton/transport/fire-deck/b/2"],
    says: "Couldn't play Deck B",
  },
  {
    command: "stopping a deck",
    press: () => press("Stop Deck B"),
    sends: ["POST", "/ableton/transport/stop-deck/b"],
    says: "Couldn't stop Deck B",
  },
  {
    command: "monitoring a deck in headphones",
    press: () => press("PFL Deck A"),
    sends: ["POST", "/ableton/pfl/a"],
    says: "Couldn't monitor Deck A in headphones",
  },
  {
    command: "switching a deck's filter",
    press: () => press("HPF Deck A"),
    sends: ["POST", "/ableton/fx/filter/a"],
    says: "Couldn't switch the filter on Deck A",
  },
  {
    command: "firing the riser",
    press: () => press("Fire riser"),
    sends: ["POST", "/ableton/fx/riser"],
    says: "Couldn't fire the riser",
  },
  {
    command: "changing the tempo",
    press: () => press("÷2"),
    sends: ["POST", "/ableton/tempo"],
    says: "Couldn't set the tempo",
  },
  {
    command: "moving the crossfader",
    press: () =>
      fireEvent.pointerDown(screen.getByRole("slider", { name: "Master crossfader" }), {
        pointerId: 1,
      }),
    sends: ["POST", "/ableton/crossfader"],
    says: "Couldn't move the crossfader",
  },
  {
    command: "starting Live's transport",
    press: () => press("Play"),
    sends: ["POST", "/ableton/play"],
    says: "Couldn't start Live's transport",
  },
  {
    command: "stopping Live's transport",
    press: () => press("Stop transport"),
    sends: ["POST", "/ableton/stop"],
    says: "Couldn't stop Live's transport",
  },
  {
    command: "panic",
    press: () => {
      press("Panic — stop all clips");
      press("PANIC");
    },
    sends: ["POST", "/ableton/transport/stop-all"],
    says: "Couldn't stop all clips",
  },
  {
    command: "recording",
    press: () => press("Start recording"),
    sends: ["POST", "/ableton/record/on"],
    says: "Couldn't start recording",
  },
  {
    command: "a resync",
    press: () => press("↻ resync"),
    sends: ["POST", "/ableton/decks/resync"],
    says: "That didn't work",
  },
];

describe("a Booth command that fails", () => {
  it.each(BOOTH_COMMANDS)("says so in the banner: $command", async ({ press, sends, says }) => {
    booth.liveAnswers = everyCommandFails;
    await booth.open();

    await press();

    await waitFor(() => expect(booth.sent(sends[0], sends[1])).toHaveLength(1));
    await expectWarningLines([`${says}: ${LIVE_IS_DOWN}`]);
  });

  it("gives the server's reason for each deck load that is refused", async () => {
    let reason = refused(503, LIVE_IS_DOWN);
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/load-track") ? reason : undefined;
    await booth.open();

    loadOnto("A");
    await expectWarningLines([`Couldn't load Half Light onto Deck A: ${LIVE_IS_DOWN}`]);
    await waitFor(() => expect(loadButton("B")).toBeEnabled());

    reason = refused(404, "track not found");
    loadOnto("B");
    await expectWarningLines([
      "Couldn't load Half Light onto Deck B: track not found",
      `Couldn't load Half Light onto Deck A: ${LIVE_IS_DOWN}`,
    ]);
    expect(within(deck("A")).queryByText(/armed/)).not.toBeInTheDocument();
    expect(within(deck("B")).queryByText(/armed/)).not.toBeInTheDocument();
  });

  it("says the backend is not answering when a command gets no answer at all", async () => {
    booth.liveAnswers = (request) => {
      if (isCommand(request, "/transport/fire-deck/a/0")) throw new TypeError("Failed to fetch");
      return undefined;
    };
    await booth.open();

    press("Play Deck A");

    await expectWarningLines([`Couldn't play Deck A: ${BACKEND_DID_NOT_ANSWER}`]);
  });

  it("says what Live refused when the answer comes back refused", async () => {
    booth.liveAnswers = ({ method, path }) => {
      if (method !== "POST") return undefined;
      if (path.includes("/transport/fire-deck/")) return { ok: false, warning: NO_DECK_COLUMNS };
      if (path.includes("/ableton/pfl/")) return { ok: false, warning: NO_DECK_COLUMNS };
      if (path.includes("/ableton/fx/reverb/")) return { ok: false, warning: NO_REVERB_RETURN };
      return undefined;
    };
    await booth.open();

    press("Play Deck A");
    await expectWarningLines([`Couldn't play Deck A: ${NO_DECK_COLUMNS}`]);

    press("PFL Deck B");
    await expectWarningLines([
      `Couldn't monitor Deck B in headphones: ${NO_DECK_COLUMNS}`,
      `Couldn't play Deck A: ${NO_DECK_COLUMNS}`,
    ]);

    press("Reverb throw Deck A");
    await expectWarningLines([
      `Couldn't switch the reverb on Deck A: ${NO_REVERB_RETURN}`,
      `Couldn't monitor Deck B in headphones: ${NO_DECK_COLUMNS}`,
      `Couldn't play Deck A: ${NO_DECK_COLUMNS}`,
    ]);
  });

  it("says so when Live may be recording nothing", async () => {
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/record/on")
        ? { ok: true, recording: true, armed_track: null, warnings: [NO_ARMED_RECORDER] }
        : undefined;
    await booth.open();

    press("Start recording");

    expect(await screen.findAllByText(NO_ARMED_RECORDER)).toHaveLength(1);
    expect(screen.getByRole("button", { name: /^Stop recording/ })).toBeInTheDocument();
  });
});

describe("the warnings banner", () => {
  function dismiss(line: HTMLElement): void {
    fireEvent.click(within(line).getByRole("button", { name: "Dismiss" }));
  }

  it("lists the newest four, counts the rest, and drops a line only when it is dismissed", async () => {
    booth.liveAnswers = everyCommandFails;
    await booth.open();

    const failures: string[] = [];
    for (const [button, says] of [
      ["Play Deck A", "Couldn't play Deck A"],
      ["Stop Deck A", "Couldn't stop Deck A"],
      ["Play Deck B", "Couldn't play Deck B"],
      ["Stop Deck B", "Couldn't stop Deck B"],
    ]) {
      press(button);
      failures.unshift(`${says}: ${LIVE_IS_DOWN}`);
      await expectWarningLines(failures);
    }
    expect(await warnings()).not.toHaveTextContent(/more/);

    press("PFL Deck A");
    await expectWarningLines([
      `Couldn't monitor Deck A in headphones: ${LIVE_IS_DOWN}`,
      ...failures.slice(0, 3),
    ]);
    expect(await warnings()).toHaveTextContent("+1 more");

    press("PFL Deck B");
    await expectWarningLines([
      `Couldn't monitor Deck B in headphones: ${LIVE_IS_DOWN}`,
      `Couldn't monitor Deck A in headphones: ${LIVE_IS_DOWN}`,
      ...failures.slice(0, 2),
    ]);
    expect(await warnings()).toHaveTextContent("+2 more");

    press("PFL Deck B");
    press("HPF Deck A");
    await expectWarningLines([
      `Couldn't switch the filter on Deck A: ${LIVE_IS_DOWN}`,
      `Couldn't monitor Deck B in headphones: ${LIVE_IS_DOWN}`,
      `Couldn't monitor Deck A in headphones: ${LIVE_IS_DOWN}`,
      failures[0],
    ]);
    expect(booth.sent("POST", "/ableton/pfl/b")).toHaveLength(2);
    expect(await warnings()).toHaveTextContent("+3 more");

    dismiss(warningLines()[0]);
    await expectWarningLines([
      `Couldn't monitor Deck B in headphones: ${LIVE_IS_DOWN}`,
      `Couldn't monitor Deck A in headphones: ${LIVE_IS_DOWN}`,
      ...failures.slice(0, 2),
    ]);
    expect(await warnings()).toHaveTextContent("+2 more");

    dismiss(warningLines()[3]);
    await expectWarningLines([
      `Couldn't monitor Deck B in headphones: ${LIVE_IS_DOWN}`,
      `Couldn't monitor Deck A in headphones: ${LIVE_IS_DOWN}`,
      failures[0],
      failures[2],
    ]);
    expect(await warnings()).toHaveTextContent("+1 more");

    for (let left = 5; left > 0; left--) {
      dismiss(warningLines()[0]);
      await waitFor(() => expect(warningLines()).toHaveLength(Math.min(left - 1, 4)));
    }
    expect(screen.queryByRole("status", { name: "Warnings" })).not.toBeInTheDocument();
  });

  it("counts a failure that happens again, so a retry that failed does not look like one that worked", async () => {
    let liveIsBack = false;
    booth.liveAnswers = (request) => (liveIsBack ? undefined : everyCommandFails(request));
    await booth.open();

    press("Play Deck A");
    await expectWarningLines([`Couldn't play Deck A: ${LIVE_IS_DOWN}`]);
    expect(warningLines()[0]).not.toHaveTextContent(/times/);

    press("Play Deck A");
    await waitFor(() => expect(warningLines()[0]).toHaveTextContent("2 times"));
    await expectWarningLines([`Couldn't play Deck A: ${LIVE_IS_DOWN}`]);

    press("Stop Deck A");
    await expectWarningLines([
      `Couldn't stop Deck A: ${LIVE_IS_DOWN}`,
      `Couldn't play Deck A: ${LIVE_IS_DOWN}`,
    ]);
    press("Play Deck A");
    await waitFor(() => expect(warningLines()[0]).toHaveTextContent("3 times"));
    await expectWarningLines([
      `Couldn't play Deck A: ${LIVE_IS_DOWN}`,
      `Couldn't stop Deck A: ${LIVE_IS_DOWN}`,
    ]);
    expect(warningLines()[1]).not.toHaveTextContent(/times/);

    liveIsBack = true;
    press("Play Deck A");
    await waitFor(() => expect(booth.sent("POST", "/ableton/transport/fire-deck/a/0")).toHaveLength(4));
    await waitFor(() => expect(screen.getByRole("button", { name: "Play Deck A" })).toBeEnabled());
    expect(warningLines()[0]).toHaveTextContent("3 times");
  });

  it("clears every line at once", async () => {
    booth.liveAnswers = everyCommandFails;
    await booth.open();

    press("Play Deck A");
    await expectWarningLines([`Couldn't play Deck A: ${LIVE_IS_DOWN}`]);
    expect(screen.queryByRole("button", { name: "Dismiss all" })).not.toBeInTheDocument();

    press("Stop Deck A");
    press("PFL Deck B");
    await waitFor(() => expect(warningLines()).toHaveLength(3));
    press("Dismiss all");

    await waitFor(() =>
      expect(screen.queryByRole("status", { name: "Warnings" })).not.toBeInTheDocument(),
    );
  });

  it("still shows a load's warning after the next load and the warp check report theirs", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"], shouldAdvanceTime: true });
    let landing = loadAnswer({ page: 1, side: "a", stems: 3, warnings: [NO_VOCALS] });
    booth.liveAnswers = (request) => {
      if (isCommand(request, "/ableton/load-track")) return landing;
      if (isCommand(request, `/ableton/warp-check/${PADS_PER_PAGE}`)) {
        return { ok: true, scene_index: PADS_PER_PAGE, checked: 4, warnings: [BASS_AT_HALF_TEMPO] };
      }
      return undefined;
    };
    await booth.open();

    loadOnto("A");
    expect(await warnings()).toHaveTextContent(NO_VOCALS);
    await waitFor(() => expect(loadButton("B")).toBeEnabled());

    landing = loadAnswer({ page: 1, side: "b", stems: 4, warnings: [NO_MIX_FILE] });
    loadOnto("B");
    await waitFor(async () => expect(await warnings()).toHaveTextContent(NO_MIX_FILE));
    expect(await warnings()).toHaveTextContent(NO_VOCALS);
    await expectWarningLines([
      `Half Light → Deck B: ${NO_MIX_FILE}`,
      `Half Light → Deck A: ${NO_VOCALS}`,
    ]);

    act(() => {
      vi.advanceTimersByTime(WARP_CHECK_DELAY_MS);
    });
    await waitFor(async () => expect(await warnings()).toHaveTextContent(BASS_AT_HALF_TEMPO));
    expect(await warnings()).toHaveTextContent(NO_MIX_FILE);
    expect(await warnings()).toHaveTextContent(NO_VOCALS);
    await expectWarningLines([
      `Half Light → Deck B: ${BASS_AT_HALF_TEMPO}`,
      `Half Light → Deck B: ${NO_MIX_FILE}`,
      `Half Light → Deck A: ${NO_VOCALS}`,
    ]);
  });
});

describe("a resync that leaves clips unmatched", () => {
  function unmatched(clipNames: string[]) {
    return {
      ok: true,
      scanned: 12,
      adopted: 12 - clipNames.length,
      unmatched: clipNames.map((clip_name, row) => ({
        scene_index: row,
        kind: "drums_a",
        clip_name,
        tried_title: clip_name,
      })),
    };
  }

  it("says how many and names up to three", async () => {
    let leftOver = ["Bootleg (drums a)", "Bootleg (bass a)", "Bootleg (vocals a)", "Edit", "Loop 4"];
    booth.liveAnswers = (request) =>
      isCommand(request, "/ableton/decks/resync") ? unmatched(leftOver) : undefined;
    await booth.open();

    press("↻ resync");
    await expectWarningLines([
      "Resync couldn't match 5 clips to your library: Bootleg (drums a), Bootleg (bass a), Bootleg (vocals a) and 2 more",
    ]);

    leftOver = ["Edit"];
    press("↻ 7/12");
    await expectWarningLines([
      "Resync couldn't match 1 clip to your library: Edit",
      "Resync couldn't match 5 clips to your library",
    ]);

    leftOver = [];
    press("↻ 11/12");
    await screen.findByRole("button", { name: "↻ 12/12" });
    expect(warningLines()).toHaveLength(2);
  });
});
