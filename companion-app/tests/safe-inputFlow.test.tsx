import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { store } from "../src/store";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  PADS_PER_PAGE,
  deck,
  grid,
  renderApp,
  songOn,
  type SeenRequest,
} from "./boothHarness";
import {
  clickWithTheMouse,
  holdAnswersTo,
  letRequestsGoOut,
  press,
  tabTo,
} from "./safe-inputHelpers";

const ORIGINAL = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
const RECOMMENDED = { track_id: 9, title: "Half Light" };
const SONG_COLUMN = "mix";
const SONG_SECONDS = 240;
const WAVEFORM_WIDTH_PX = 1000;
const MISSING_STEM = "The bass stem is missing on disk.";

const LOAD = "/api/v1/ableton/load-track";
const PREVIEW = "/api/v1/ableton/preview";
const STOP_PREVIEW = "/api/v1/ableton/preview/stop";
const TEMPO = "/api/v1/ableton/tempo";
const STOP_ALL_CLIPS = "/api/v1/ableton/transport/stop-all";
const FIRE_DECK_A_CUE_3 = "/api/v1/ableton/transport/fire-deck/a/2";
const SEEK_ON_DECK_A = `/api/v1/ableton/transport/seek/${LIVE_TRACK_OF.mix_a}/0`;

const DRUMS_A_INTRO_PAD = "Drums A: Night Drive — Test Artist — cue 1, intro — tap to fire";
const REMOVE_IDLE = "Remove Drums clip from scene";
const REMOVE_ARMED = "Press again to remove Drums clip from scene";
const REMOVE_STAYS_ARMED_MS = 2500;

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

function boothWithBothDecksLoaded({ method, path, body }: SeenRequest): unknown {
  if (path.endsWith("/ableton/decks")) {
    return {
      columns: LIVE_TRACK_OF,
      cells: [...songOn(0, "a", ORIGINAL), ...songOn(0, "b", CLUB_MIX)],
    };
  }
  if (/\/tracks\/\d+\/waveform$/.test(path)) {
    return { peaks: [0.2, 0.8, 0.5, 0.3], duration_seconds: SONG_SECONDS, num_peaks: 4 };
  }
  if (path.endsWith("/tracks/search")) return [];
  if (path.endsWith("/recommend/by-column")) {
    const column = (body as { column: string }).column;
    return { column, combo_size: 0, recs: column === SONG_COLUMN ? [recommendation()] : [] };
  }
  if (method === "POST" && path === LOAD) {
    return {
      ok: true,
      scene_index: PADS_PER_PAGE,
      page: 1,
      side: (body as { side: "a" | "b" }).side,
      track_indices: LIVE_TRACK_OF,
      stems_loaded: 3,
      message: null,
      warnings: [MISSING_STEM],
    };
  }
  return undefined;
}

function pointerEvent(type: string, clientX: number): MouseEvent {
  return Object.assign(new MouseEvent(type, { bubbles: true, clientX }), { pointerId: 1 });
}

async function waveformOf(side: "A" | "B"): Promise<Element> {
  const bars = await within(deck(side)).findAllByTestId("waveform-bar");
  return bars[0].closest("div") as Element;
}

function removeButtonOf(padTitle: string): HTMLElement {
  const pad = within(grid()).getByTitle(padTitle).parentElement as HTMLElement;
  return within(pad).getByRole("button", { name: /remove Drums clip/i });
}

function seeksSent(): SeenRequest[] {
  return backend.requests.filter((request) => request.path.includes("/ableton/transport/seek/"));
}

function clipsRemoved(): SeenRequest[] {
  return backend.requests.filter((request) => request.method === "DELETE");
}

function decksLoaded(): string[] {
  return backend.sent("POST", LOAD).map((sent) => (sent.body as { side: string }).side);
}

async function openBoothWithTheCursorOnTheRecommendedSong(): Promise<void> {
  renderApp();
  await screen.findByRole("button", { name: "Page 1" });
  backend.liveNowPlays({});
  await screen.findByRole("group", { name: RECOMMENDED.title });
  press("ArrowDown");
}

async function openBpmPopoverWithADraftOf(bpm: number): Promise<HTMLElement> {
  renderApp();
  await screen.findByRole("button", { name: "Page 1" });
  backend.liveNowPlays({});
  await screen.findByRole("button", { name: "128.0" });
  return draftInTheBpmPopover(bpm);
}

function draftInTheBpmPopover(bpm: number): HTMLElement {
  fireEvent.click(screen.getByRole("button", { name: "128.0" }));
  const slider = screen.getByRole("slider", { name: "Master BPM" });
  fireEvent.change(slider, { target: { value: String(bpm) } });
  return slider;
}

function tempoChanges(): unknown[] {
  return backend.sent("POST", TEMPO).map((sent) => sent.body);
}

function startFromACleanBooth(): void {
  store.setView("booth");
  store.setBoothPage(0);
  store.setGridFocus(null);
  store.setPreviewing(null);
  store.clearLoadWarnings();
  store.closeCommandBar();
  store.clearDecks();
  store.clearArm("a");
  store.clearArm("b");
}

let backend: BoothBackend;

beforeEach(() => {
  startFromACleanBooth();
  backend = new BoothBackend(boothWithBothDecksLoaded);
  // jsdom has no layout and no pointer capture, both of which the waveform's click-to-seek needs.
  Object.assign(Element.prototype, {
    setPointerCapture: () => {},
    releasePointerCapture: () => {},
    hasPointerCapture: () => false,
  });
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: WAVEFORM_WIDTH_PX,
    bottom: 56,
    width: WAVEFORM_WIDTH_PX,
    height: 56,
    toJSON: () => ({}),
  });
});

afterEach(() => {
  vi.useRealTimers();
  backend.restore();
});

describe("stray input in the Booth", () => {
  it("seeks a deck only when the press began on its waveform", async () => {
    renderApp();
    await screen.findByRole("button", { name: "Page 1" });
    const waveform = await waveformOf("A");
    const cueOnTheSameDeck = within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" });

    fireEvent(cueOnTheSameDeck, pointerEvent("pointerdown", 300));
    fireEvent(waveform, pointerEvent("pointerup", 800));

    fireEvent(waveform, pointerEvent("pointerdown", 600));
    fireEvent(waveform, pointerEvent("pointercancel", 600));
    fireEvent(waveform, pointerEvent("pointerup", 800));

    fireEvent(waveform, pointerEvent("pointerdown", 250));
    fireEvent(waveform, pointerEvent("pointerup", 250));

    await waitFor(() => expect(seeksSent()).not.toHaveLength(0));
    expect(seeksSent()).toMatchObject([
      { method: "POST", path: SEEK_ON_DECK_A, query: "position=128" },
    ]);
  });

  it("removes a pad's clip from Live on the second click, not the first", async () => {
    renderApp();
    await screen.findByRole("button", { name: "Page 1" });
    const remove = () => removeButtonOf(DRUMS_A_INTRO_PAD);
    expect(remove()).toHaveAccessibleName(REMOVE_IDLE);
    expect(remove()).toHaveTextContent("×");

    fireEvent.click(remove());
    await letRequestsGoOut();
    expect(clipsRemoved()).toEqual([]);
    expect(remove()).toHaveAccessibleName(REMOVE_ARMED);
    expect(remove()).toHaveTextContent("remove?");

    fireEvent.click(remove());
    await waitFor(() =>
      expect(clipsRemoved()).toMatchObject([
        { path: `/api/v1/ableton/decks/cell/${LIVE_TRACK_OF.drums_a}/0` },
      ]),
    );
    expect(remove()).toHaveAccessibleName(REMOVE_IDLE);

    vi.useFakeTimers();
    fireEvent.click(remove());
    expect(remove()).toHaveAccessibleName(REMOVE_ARMED);
    act(() => {
      vi.advanceTimersByTime(REMOVE_STAYS_ARMED_MS);
    });
    expect(remove()).toHaveAccessibleName(REMOVE_IDLE);
    expect(remove()).toHaveTextContent("×");
    vi.useRealTimers();

    fireEvent.click(remove());
    await letRequestsGoOut();
    expect(remove()).toHaveAccessibleName(REMOVE_ARMED);
    expect(clipsRemoved()).toHaveLength(1);
  });
});

describe("Enter and Space on the Booth's plan grid", () => {
  it("loads the pick once when Enter is pressed again before the load has finished", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();
    const finishLoading = holdAnswersTo(LOAD);

    press("Enter");
    press("Enter");
    press("Enter", { shiftKey: true });
    await letRequestsGoOut();
    expect(backend.sent("POST", LOAD)).toMatchObject([
      { body: { track_id: RECOMMENDED.track_id, kinds: null, side: "a", page: 0 } },
    ]);

    finishLoading();
    await screen.findByText(MISSING_STEM);
    press("Enter", { shiftKey: true });
    await waitFor(() => expect(decksLoaded()).toEqual(["a", "b"]));
  });

  it("says which pick is loading onto which deck until the load has finished", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();
    const finishLoading = holdAnswersTo(LOAD);

    press("Enter");
    const loading = await screen.findByText("loading Half Light onto Deck A…");
    expect(loading).toHaveAttribute("role", "status");
    expect(loading.closest("[data-plan-grid]")).not.toBeNull();

    finishLoading();
    await waitFor(() => expect(screen.queryByText(/^loading /)).not.toBeInTheDocument());
    expect(screen.getByText(MISSING_STEM)).toBeInTheDocument();
  });

  it("names the pick in the warning a load leaves behind", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();

    press("Enter", { shiftKey: true });

    const warning = await screen.findByTestId("load-warnings");
    expect(within(warning).getByText("Half Light → Deck B:")).toBeInTheDocument();
    expect(within(warning).getByText(MISSING_STEM)).toBeInTheDocument();
  });

  it("loads the pick once however long Enter is held", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();

    press("Enter");
    await screen.findByText(MISSING_STEM);
    press("Enter", { repeat: true });
    press("Enter", { repeat: true });
    press("Enter", { repeat: true });
    press("Enter", { shiftKey: true });

    await waitFor(() => expect(decksLoaded()).toContain("b"));
    expect(decksLoaded()).toEqual(["a", "b"]);
  });

  it("auditions the pick once however long Space is held", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();

    press(" ");
    press(" ", { repeat: true });
    press(" ", { repeat: true });
    press(" ", { repeat: true });
    press("Escape");

    await waitFor(() => expect(backend.sent("POST", STOP_PREVIEW)).not.toHaveLength(0));
    expect(backend.sent("POST", PREVIEW)).toMatchObject([
      { body: { track_id: RECOMMENDED.track_id, column: SONG_COLUMN } },
    ]);
    expect(backend.sent("POST", STOP_PREVIEW)).toHaveLength(1);
  });

  it("leaves Enter and Space to a button outside the plan grid until an arrow key moves the cursor", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();

    tabTo(screen.getByRole("button", { name: "Panic — stop all clips" }));
    press("Enter");
    expect(screen.getByRole("button", { name: "PANIC" })).toHaveFocus();
    press("Enter");
    await waitFor(() => expect(backend.sent("POST", STOP_ALL_CLIPS)).toHaveLength(1));

    const cue = within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" });
    tabTo(cue);
    press(" ");
    await waitFor(() => expect(backend.sent("POST", FIRE_DECK_A_CUE_3)).toHaveLength(1));
    expect(backend.sent("POST", LOAD)).toEqual([]);
    expect(backend.sent("POST", PREVIEW)).toEqual([]);

    press("ArrowDown");
    expect(cue).not.toHaveFocus();
    press(" ");
    await waitFor(() =>
      expect(backend.sent("POST", PREVIEW)).toMatchObject([
        { body: { track_id: RECOMMENDED.track_id, column: SONG_COLUMN } },
      ]),
    );
    expect(backend.sent("POST", FIRE_DECK_A_CUE_3)).toHaveLength(1);
  });

  it("leaves Enter and Space to a button clicked with the mouse outside the plan grid", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();

    clickWithTheMouse(screen.getByRole("button", { name: "Panic — stop all clips" }));
    expect(screen.getByRole("button", { name: "PANIC" })).toHaveFocus();
    press("Enter");
    await waitFor(() => expect(backend.sent("POST", STOP_ALL_CLIPS)).toHaveLength(1));

    const cue = within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" });
    clickWithTheMouse(cue);
    expect(cue).toHaveFocus();
    press(" ");
    await waitFor(() => expect(backend.sent("POST", FIRE_DECK_A_CUE_3)).toHaveLength(2));
    expect(backend.sent("POST", LOAD)).toEqual([]);
    expect(backend.sent("POST", PREVIEW)).toEqual([]);

    press("ArrowDown");
    expect(cue).not.toHaveFocus();
    press("Enter");
    await waitFor(() => expect(decksLoaded()).toEqual(["a"]));
    expect(backend.sent("POST", FIRE_DECK_A_CUE_3)).toHaveLength(2);
  });

  it("still loads the pick under the cursor when Enter lands on a button inside the plan grid", async () => {
    await openBoothWithTheCursorOnTheRecommendedSong();
    const pick = screen.getByRole("group", { name: RECOMMENDED.title });
    expect(pick.closest("[data-plan-grid]")).not.toBeNull();

    within(pick).getByTitle("Preview in headphones (cue)").focus();
    press("Enter", { shiftKey: true });

    await waitFor(() => expect(decksLoaded()).toEqual(["b"]));
    expect(backend.sent("POST", PREVIEW)).toEqual([]);
  });
});

describe("Enter with the BPM popover open", () => {
  it("applies the draft tempo only when it is pressed inside the popover", async () => {
    const slider = await openBpmPopoverWithADraftOf(120);
    expect(slider).toHaveFocus();

    press("k", { metaKey: true });
    const search = await screen.findByRole("textbox");
    await waitFor(() => expect(search).toHaveFocus());
    press("Enter");
    await letRequestsGoOut();
    expect(tempoChanges()).toEqual([]);
    expect(slider).toBeInTheDocument();

    slider.focus();
    press("Enter");
    await waitFor(() => expect(tempoChanges()).toEqual([{ bpm: 120 }]));
    expect(slider).not.toBeInTheDocument();

    draftInTheBpmPopover(110);
    clickWithTheMouse(within(screen.getByTestId("bpm-slider")).getByText("110.0"));
    press("Enter");
    await waitFor(() => expect(tempoChanges()).toEqual([{ bpm: 120 }, { bpm: 110 }]));
  });

  it("applies the draft without also loading the pick under the plan grid's cursor", async () => {
    await openBpmPopoverWithADraftOf(120);
    await screen.findByRole("group", { name: RECOMMENDED.title });
    clickWithTheMouse(within(screen.getByTestId("bpm-slider")).getByText("120.0"));
    press("ArrowDown");

    press("Enter");

    await waitFor(() => expect(tempoChanges()).toEqual([{ bpm: 120 }]));
    await letRequestsGoOut();
    expect(backend.sent("POST", LOAD)).toEqual([]);
  });

  it("cancels, and applies nothing, when it is pressed on Cancel", async () => {
    const slider = await openBpmPopoverWithADraftOf(120);

    tabTo(within(screen.getByTestId("bpm-slider")).getByRole("button", { name: "Cancel" }));
    press("Enter");

    await letRequestsGoOut();
    expect(slider).not.toBeInTheDocument();
    expect(tempoChanges()).toEqual([]);
  });
});
