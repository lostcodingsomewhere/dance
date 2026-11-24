import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  BoothBackend,
  LIVE_TRACK_OF,
  deck,
  renderApp,
  songOn,
  type SeenRequest,
} from "./boothHarness";

const ON_DECK_A = { track_id: 7, title: "Night Drive" };
const ON_DECK_B = { track_id: 8, title: "Half Light" };
const FX_STATE_PATH = "/api/v1/ableton/fx/state";
const SIDES = ["A", "B"] as const;

interface FxInLive {
  returns: Record<string, number>;
  scenes: Record<string, number>;
}

const NO_FX: FxInLive = { returns: {}, scenes: {} };
const A_FILTER_RETURN_ONLY: FxInLive = { returns: { filter: 10 }, scenes: {} };
const EVERY_FX: FxInLive = {
  returns: { filter: 10, reverb: 11, delay: 12, riser: 13 },
  scenes: { riser: 0 },
};

function aLiveSetWith(fx: FxInLive) {
  const nothingEngaged = { a: false, b: false };
  return ({ path }: SeenRequest): unknown => {
    if (path.endsWith("/ableton/decks")) {
      return {
        columns: LIVE_TRACK_OF,
        cells: [...songOn(0, "a", ON_DECK_A), ...songOn(0, "b", ON_DECK_B)],
      };
    }
    if (path === FX_STATE_PATH) {
      return {
        filter: nothingEngaged,
        reverb: nothingEngaged,
        delay: nothingEngaged,
        discovered: fx,
      };
    }
    return undefined;
  };
}

async function bothDecksOnScreen(): Promise<void> {
  await screen.findByRole("region", { name: "Deck A" });
  await waitFor(() => expect(backend.sent("GET", FX_STATE_PATH)).not.toHaveLength(0));
}

function control(side: "A" | "B", name: string): HTMLElement | null {
  return within(deck(side)).queryByRole("button", { name: `${name} Deck ${side}` });
}

function riser(): HTMLElement | null {
  return screen.queryByRole("button", { name: "Fire riser" });
}

function sideBySideWithTheDecks(): Element[] {
  return Array.from(deck("A").parentElement!.children);
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("the FX controls on the decks", () => {
  it("are left out when the Live Set has no FX returns and no Riser clip", async () => {
    backend = new BoothBackend(aLiveSetWith(NO_FX));
    renderApp();
    await bothDecksOnScreen();

    for (const side of SIDES) {
      expect(control(side, "Play")).toBeInTheDocument();
      expect(control(side, "Stop")).toBeInTheDocument();
      expect(control(side, "PFL")).toBeInTheDocument();
      expect(control(side, "HPF")).not.toBeInTheDocument();
      expect(control(side, "Reverb throw")).not.toBeInTheDocument();
      expect(control(side, "Delay throw")).not.toBeInTheDocument();
    }
    expect(riser()).not.toBeInTheDocument();

    const columns = sideBySideWithTheDecks();
    expect(columns).toHaveLength(2);
    expect(columns[0]).toBe(deck("A"));
    expect(columns[1]).toBe(deck("B"));
  });

  it("show only the FX the Live Set has", async () => {
    backend = new BoothBackend(aLiveSetWith(A_FILTER_RETURN_ONLY));
    renderApp();
    await bothDecksOnScreen();

    expect(await within(deck("A")).findByRole("button", { name: "HPF Deck A" })).toBeEnabled();
    for (const side of SIDES) {
      expect(control(side, "HPF")).toBeEnabled();
      expect(control(side, "Reverb throw")).not.toBeInTheDocument();
      expect(control(side, "Delay throw")).not.toBeInTheDocument();
    }
    expect(riser()).not.toBeInTheDocument();

    fireEvent.click(control("B", "HPF")!);
    await waitFor(() =>
      expect(backend.sent("POST", "/api/v1/ableton/fx/filter/b")).toHaveLength(1),
    );
  });

  it("put the Riser in a column of its own between the two decks", async () => {
    backend = new BoothBackend(aLiveSetWith(EVERY_FX));
    renderApp();
    await bothDecksOnScreen();

    expect(await screen.findByRole("button", { name: "Fire riser" })).toBeEnabled();
    for (const side of SIDES) {
      expect(control(side, "HPF")).toBeEnabled();
      expect(control(side, "Reverb throw")).toBeEnabled();
      expect(control(side, "Delay throw")).toBeEnabled();
      expect(deck(side)).not.toContainElement(riser());
    }

    const columns = sideBySideWithTheDecks();
    expect(columns).toHaveLength(3);
    expect(columns[0]).toBe(deck("A"));
    expect(columns[1]).toContainElement(riser());
    expect(columns[1]).not.toHaveClass("absolute");
    expect(columns[2]).toBe(deck("B"));

    fireEvent.click(riser()!);
    await waitFor(() => expect(backend.sent("POST", "/api/v1/ableton/fx/riser")).toHaveLength(1));
  });
});
