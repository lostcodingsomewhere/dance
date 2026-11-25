import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import { LIVE_TRACK_OF, PADS_PER_PAGE, renderApp, songOn, type BoothBackend } from "./boothHarness";
import { BoothMidSet, CLUB_MIX, ORIGINAL, RECOMMENDED, press } from "./failuresFlowBooth";
import {
  NIGHT_DRIVE,
  CLUB_MIX as CLUB_MIX_IN_THE_LOG,
  SessionLog,
  backendTime,
  boothWith,
  clips,
  liveNowPlays,
  liveStops,
  playsLogged,
} from "./session-logBackend";

const THIRD_SONG = { track_id: 11, title: "Low Tide" };
const FULL_A = { crossfader: -1 };
const FULL_B = { crossfader: 1 };

function deckAPlays(scene: number): Record<string, number> {
  return Object.fromEntries(
    ["drums_a", "bass_a", "vocals_a", "other_a"].map((kind) => [LIVE_TRACK_OF[kind], scene]),
  );
}

function pageShown(): string | null {
  const pages = screen.getByRole("navigation", { name: "Pages" });
  const shown = within(pages)
    .getAllByRole("button")
    .find((page) => page.getAttribute("aria-pressed") === "true");
  return shown?.textContent ?? null;
}

describe("the Booth while the APC40 plays a prepared set", () => {
  let booth: BoothMidSet;

  afterEach(async () => {
    await booth.close();
  });

  it("turns to the page whose pads were pressed, and keeps a page you turned to until the next press", async () => {
    booth = new BoothMidSet();
    booth.inLive = [
      ...songOn(0, "a", ORIGINAL),
      ...songOn(0, "b", CLUB_MIX),
      ...songOn(1, "a", RECOMMENDED),
      ...songOn(2, "a", THIRD_SONG),
    ];
    store.setBoothPage(2);
    await booth.open();
    expect(pageShown()).toBe("3");

    booth.liveNowPlays(deckAPlays(0));
    await waitFor(() => expect(pageShown()).toBe("1"));

    fireEvent.click(screen.getByRole("button", { name: /^Page 3/ }));
    booth.liveNowPlays(deckAPlays(0));
    expect(pageShown()).toBe("3");

    booth.liveNowPlays(deckAPlays(PADS_PER_PAGE + 1));
    await waitFor(() => expect(pageShown()).toBe("2"));
  });

  it("aims a deck's ▶ at the row that started on it, not at a row armed before", async () => {
    booth = new BoothMidSet();
    booth.inLive = [...songOn(0, "a", ORIGINAL), ...songOn(1, "a", RECOMMENDED)];
    store.armDeck("a", PADS_PER_PAGE);
    await booth.open();

    booth.liveNowPlays(deckAPlays(2));
    await waitFor(() => expect(playDeckA()).toHaveAttribute("title", "Play Deck A (fire scene 3)"));
    press("Play Deck A");

    await waitFor(() =>
      expect(booth.sent("POST", "/ableton/transport/fire-deck/a/2")).toHaveLength(1),
    );
    expect(booth.sent("POST", `/ableton/transport/fire-deck/a/${PADS_PER_PAGE}`)).toEqual([]);
  });

  it("forgets a row armed in another set once the set open in Live has no such row", async () => {
    booth = new BoothMidSet();
    booth.inLive = songOn(0, "a", ORIGINAL);
    store.armDeck("a", 35);
    await booth.open();

    await waitFor(() => expect(playDeckA()).toHaveAttribute("title", "Play Deck A (fire scene 1)"));
    press("Play Deck A");

    await waitFor(() =>
      expect(booth.sent("POST", "/ableton/transport/fire-deck/a/0")).toHaveLength(1),
    );
    expect(booth.sent("POST", "/ableton/transport/fire-deck/a/35")).toEqual([]);
  });
});

function playDeckA(): HTMLElement {
  return screen.getByRole("button", { name: "Play Deck A" });
}

describe("a song cued behind the crossfader", () => {
  let backend: BoothBackend;
  let log: SessionLog;

  afterEach(async () => {
    await liveStops();
    backend.restore();
  });

  function keyShown(): string | null {
    return within(screen.getByRole("banner")).getByText("Key").nextElementSibling?.textContent ?? null;
  }

  it("is not logged or read as the key until the crossfader lets it through", async () => {
    log = new SessionLog();
    log.open(backendTime());
    const clubMixInAnotherKey = () => [
      ...songOn(0, "a", NIGHT_DRIVE),
      ...songOn(0, "b", CLUB_MIX_IN_THE_LOG).map((cell) => ({ ...cell, key_camelot: "6A" })),
    ];
    backend = boothWith(log, clubMixInAnotherKey);
    renderApp();
    await screen.findByRole("button", { name: /^Page 1/ });

    await liveNowPlays(backend, clips("a", 0), FULL_A);
    await waitFor(() => expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]));

    await liveNowPlays(backend, { ...clips("a", 0), ...clips("b", 0) }, FULL_A);
    await new Promise((settled) => setTimeout(settled, 50));
    expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id]);
    expect(keyShown()).toBe("8A");

    await liveNowPlays(backend, { ...clips("a", 0), ...clips("b", 0) }, FULL_B);
    await waitFor(() =>
      expect(playsLogged(backend)).toEqual([NIGHT_DRIVE.track_id, CLUB_MIX_IN_THE_LOG.track_id]),
    );
    expect(keyShown()).toBe("6A");
  });
});
