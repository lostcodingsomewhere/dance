import { onlineManager } from "@tanstack/react-query";
import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { songOn } from "./boothHarness";
import {
  BoothMidSet,
  LIVE_IS_DOWN,
  RECOMMENDED,
  everyCommandFails,
  expectWarningLines,
  isCommand,
  loadAnswer,
  loadOnto,
  press,
} from "./failuresFlowBooth";

let booth: BoothMidSet;

function theBrowserGoesOffline(): void {
  act(() => {
    window.dispatchEvent(new Event("offline"));
  });
}

beforeEach(() => {
  booth = new BoothMidSet();
});

afterEach(() => {
  onlineManager.setOnline(true);
  booth.close();
});

describe("the venue's network dropping mid-set", () => {
  it("still sends Booth commands to the backend on this machine, and says so when one fails", async () => {
    await booth.open();
    theBrowserGoesOffline();

    press("Play Deck A");
    await waitFor(() => expect(booth.sent("POST", "/ableton/transport/fire-deck/a/0")).toHaveLength(1));

    booth.liveAnswers = everyCommandFails;
    press("Stop Deck A");
    await expectWarningLines([`Couldn't stop Deck A: ${LIVE_IS_DOWN}`]);
  });

  it("still shows what a load put in Live", async () => {
    booth.liveAnswers = (request) => {
      if (!isCommand(request, "/ableton/load-track")) return undefined;
      booth.inLive = [...booth.inLive, ...songOn(1, "b", RECOMMENDED)];
      return loadAnswer({ page: 1, side: "b", stems: 4, warnings: [] });
    };
    await booth.open();
    theBrowserGoesOffline();

    loadOnto("B");

    const pageTwo = await screen.findByRole("button", { name: "Page 2" });
    await waitFor(() => expect(pageTwo).toHaveAttribute("aria-pressed", "true"));
  });
});
