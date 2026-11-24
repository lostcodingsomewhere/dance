import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import { SetEditor } from "../src/views/SetEditor";
import { BoothBackend, type SeenRequest } from "./boothHarness";
import { press } from "./safe-inputHelpers";

const SET_ID = 1;
const PLANNED = { track_id: 7, title: "Night Drive" };
const FIRST_REC = { track_id: 8, title: "Night Drive (Club Mix)" };
const SECOND_REC = { track_id: 9, title: "Half Light" };
const APPEND = `/api/v1/sets/${SET_ID}/plan/append`;

function planItem(song: { track_id: number; title: string }) {
  return { ...song, artist: "Test Artist", key_camelot: "8A", bpm: 128, floor_energy: 6 };
}

function rec(song: { track_id: number; title: string }) {
  return {
    track_id: song.track_id,
    stem_file_id: null,
    track_title: song.title,
    track_artist: "Test Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: { embedding: 0.9 },
    reasons: ["close vibe"],
  };
}

function aSetWithOneSongPlanned({ path, query }: SeenRequest): unknown {
  const plan = {
    set_id: SET_ID,
    queues: { drums: [], bass: [], vocals: [], other: [], song: [planItem(PLANNED)] },
  };
  if (path === "/api/v1/sets/active" || path === `/api/v1/sets/${SET_ID}`) {
    return {
      id: SET_ID,
      name: "Friday night",
      notes: null,
      is_active: true,
      created_at: "2026-11-01T00:00:00",
      updated_at: "2026-11-01T00:00:00",
      tracks: [],
    };
  }
  if (path === "/api/v1/sets") return [];
  if (path === `/api/v1/sets/${SET_ID}/plan` || path === APPEND) return plan;
  if (path === `/api/v1/sets/${SET_ID}/plan-recs`) {
    const forTheSongColumn = query.includes("role=song");
    return {
      column: "mix",
      combo_size: 0,
      recs: forTheSongColumn ? [rec(FIRST_REC), rec(SECOND_REC)] : [],
    };
  }
  return undefined;
}

function picksAdded(): number[] {
  return backend.sent("POST", APPEND).map((sent) => (sent.body as { track_id: number }).track_id);
}

async function openSetViewWithTheCursorOnTheFirstRec(): Promise<void> {
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <SetEditor />
    </QueryClientProvider>,
  );
  await screen.findByRole("group", { name: SECOND_REC.title });
  press("ArrowDown");
}

let backend: BoothBackend;

beforeEach(() => {
  store.setGridFocus(null);
  store.setPreviewing(null);
  store.closeCommandBar();
  backend = new BoothBackend(aSetWithOneSongPlanned);
});

afterEach(() => {
  backend.restore();
});

describe("Enter on the Set view's plan grid", () => {
  it("adds the pick under the cursor once however long Enter is held", async () => {
    await openSetViewWithTheCursorOnTheFirstRec();

    press("Enter");
    press("Enter", { repeat: true });
    press("Enter", { repeat: true });
    press("ArrowDown");
    press("Enter");

    await waitFor(() => expect(picksAdded()).toContain(SECOND_REC.track_id));
    expect(picksAdded()).toEqual([FIRST_REC.track_id, SECOND_REC.track_id]);
  });
});
