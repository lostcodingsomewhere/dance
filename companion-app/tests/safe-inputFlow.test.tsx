import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { store } from "../src/store";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  deck,
  grid,
  renderApp,
  songOn,
  type SeenRequest,
} from "./boothHarness";

const ORIGINAL = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
const SONG_SECONDS = 240;
const WAVEFORM_WIDTH_PX = 1000;
const SEEK_ON_DECK_A = `/api/v1/ableton/transport/seek/${LIVE_TRACK_OF.mix_a}/0`;
const DRUMS_A_INTRO_PAD = "Drums A: Night Drive — Test Artist — cue 1, intro — tap to fire";
const REMOVE_IDLE = "Remove Drums clip from scene";
const REMOVE_ARMED = "Press again to remove Drums clip from scene";
const REMOVE_STAYS_ARMED_MS = 2500;

function boothWithBothDecksLoaded({ path }: SeenRequest): unknown {
  if (path.endsWith("/ableton/decks")) {
    return {
      columns: LIVE_TRACK_OF,
      cells: [...songOn(0, "a", ORIGINAL), ...songOn(0, "b", CLUB_MIX)],
    };
  }
  if (/\/tracks\/\d+\/waveform$/.test(path)) {
    return { peaks: [0.2, 0.8, 0.5, 0.3], duration_seconds: SONG_SECONDS, num_peaks: 4 };
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

function seeksSent(): SeenRequest[] {
  return backend.requests.filter((request) => request.path.includes("/ableton/transport/seek/"));
}

function removeButtonOf(padTitle: string): HTMLElement {
  const pad = within(grid()).getByTitle(padTitle).parentElement as HTMLElement;
  return within(pad).getByRole("button", { name: /remove Drums clip/i });
}

function clipsRemoved(): SeenRequest[] {
  return backend.requests.filter((request) => request.method === "DELETE");
}

async function letRequestsGoOut(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function startFromACleanBooth() {
  store.setView("booth");
  store.setBoothPage(0);
  store.setGridFocus(null);
  store.setPreviewing(null);
  store.clearLoadWarnings();
  store.closeCommandBar();
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
