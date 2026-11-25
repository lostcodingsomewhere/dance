import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { BoothBackend, LIVE_TRACK_OF, renderApp, type SeenRequest } from "./boothHarness";
import { API, SetsOnTheServer } from "./setsFlowBackend";
import {
  HALF_LIGHT,
  LIBRARY,
  LOW_TIDE,
  NIGHT_DRIVE,
  SLOW_BURN,
  planned,
  renderSetView,
} from "./setsFlowView";

const WHERE_EACH_SONG_STANDS: Record<number, { state: string; download_failed: boolean }> = {
  [NIGHT_DRIVE.track_id]: { state: "complete", download_failed: false },
  [HALF_LIGHT.track_id]: { state: "pending", download_failed: false },
  [LOW_TIDE.track_id]: { state: "error", download_failed: true },
  [SLOW_BURN.track_id]: { state: "error", download_failed: false },
};
const FOUR_SONGS = {
  song: [NIGHT_DRIVE.track_id, HALF_LIGHT.track_id, LOW_TIDE.track_id, SLOW_BURN.track_id],
};
const LOAD = `${API}/ableton/load-track`;

interface PlanAnswer {
  queues: Record<string, { track_id: number }[]>;
}

function isAPlan(answer: unknown): answer is PlanAnswer {
  return typeof answer === "object" && answer !== null && "queues" in answer;
}

function withStates(answer: unknown): unknown {
  if (!isAPlan(answer)) return answer;
  const queues = Object.fromEntries(
    Object.entries(answer.queues).map(([role, items]) => [
      role,
      items.map((item) => ({ ...item, ...WHERE_EACH_SONG_STANDS[item.track_id] })),
    ]),
  );
  return { ...answer, queues };
}

function aSetWhoseSongsAreNotAllReady(inTheBooth: boolean) {
  const server = new SetsOnTheServer(LIBRARY);
  server.has("Friday night", FOUR_SONGS, true);
  return (request: SeenRequest): unknown => {
    if (inTheBooth && request.path === `${API}/ableton/decks`) {
      return { columns: LIVE_TRACK_OF, cells: [] };
    }
    if (inTheBooth && request.path === `${API}/recommend/by-column`) {
      return { column: "mix", combo_size: 0, recs: [] };
    }
    return withStates(server.answer(request));
  };
}

function pick(title: string): HTMLElement {
  return within(planned("song")).getByRole("group", { name: title });
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("a planned song that cannot be played yet", () => {
  it("says where it stands in the Set view and offers pads only for a finished song", async () => {
    backend = new BoothBackend(aSetWhoseSongsAreNotAllReady(false));
    renderSetView();
    await screen.findByRole("group", { name: NIGHT_DRIVE.title });

    expect(within(pick(HALF_LIGHT.title)).getByText("on its way")).toBeInTheDocument();
    expect(within(pick(LOW_TIDE.title)).getByText("download failed")).toBeInTheDocument();
    expect(within(pick(SLOW_BURN.title)).getByText("failed, see Pipeline")).toBeInTheDocument();
    expect(within(pick(NIGHT_DRIVE.title)).queryByText(/on its way|failed/)).not.toBeInTheDocument();

    expect(within(pick(NIGHT_DRIVE.title)).getByRole("button", { name: "Pads" })).toBeInTheDocument();
    for (const notReady of [HALF_LIGHT, LOW_TIDE, SLOW_BURN]) {
      expect(within(pick(notReady.title)).queryByRole("button", { name: "Pads" })).not.toBeInTheDocument();
      expect(within(pick(notReady.title)).getByTitle("Not ready to preview")).toBeDisabled();
      expect(within(pick(notReady.title)).getByRole("button", { name: "Versions" })).toBeInTheDocument();
    }
  });

  it("cannot be loaded onto a deck in the Booth, by button or by Enter", async () => {
    backend = new BoothBackend(aSetWhoseSongsAreNotAllReady(true));
    renderApp();
    await screen.findByRole("group", { name: NIGHT_DRIVE.title });

    expect(within(pick(NIGHT_DRIVE.title)).getByTitle("Load → Deck A")).toBeInTheDocument();
    for (const notReady of [HALF_LIGHT, LOW_TIDE, SLOW_BURN]) {
      expect(within(pick(notReady.title)).queryByTitle("Load → Deck A")).not.toBeInTheDocument();
      expect(within(pick(notReady.title)).queryByTitle("Load → Deck B")).not.toBeInTheDocument();
    }
    expect(within(pick(LOW_TIDE.title)).getByText("download failed")).toBeInTheDocument();

    fireEvent.mouseDown(pick(LOW_TIDE.title));
    fireEvent.keyDown(document.body, { key: "Enter" });

    const warnings = await screen.findByTestId("load-warnings");
    expect(within(warnings).getByText("Couldn't load Low Tide:")).toBeInTheDocument();
    expect(within(warnings).getByText("download failed")).toBeInTheDocument();
    expect(backend.sent("POST", LOAD)).toEqual([]);

    fireEvent.mouseDown(pick(NIGHT_DRIVE.title));
    fireEvent.keyDown(document.body, { key: "Enter" });
    await waitFor(() => expect(backend.sent("POST", LOAD)).toHaveLength(1));
  });
});
