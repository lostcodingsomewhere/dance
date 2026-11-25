import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import { SetEditor } from "../src/views/SetEditor";
import { BoothBackend, renderApp, type SeenRequest } from "./boothHarness";
import { letRequestsGoOut, press, tabTo } from "./safe-inputHelpers";

const SET_ID = 1;
const SET_NAME = "Friday night";
const PLANNED = { track_id: 7, title: "Night Drive" };
const FIRST_REC = { track_id: 8, title: "Night Drive (Club Mix)" };
const SECOND_REC = { track_id: 9, title: "Half Light" };
const APPEND = `/api/v1/sets/${SET_ID}/plan/append`;
const REMIX_ON_SPOTIFY = {
  spotify_id: "sp-remix",
  title: "Night Drive - Tinlicker Remix",
  artist: "Test Artist, Tinlicker",
  album: "Night Drive",
  duration_ms: 210_000,
  image_url: null,
  explicit: false,
  label: "Tinlicker Remix",
  library_track_id: null,
  library_state: null,
  download_failed: false,
};
const PADS = [1, 2, 3].map((slot) => ({
  slot,
  role: "mix",
  stem_file_id: null,
  position_ms: slot * 30_000,
  length_ms: 15_000,
  length_bars: 8,
  section_label: "drop",
  source: "auto",
  silent: false,
}));

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
      name: SET_NAME,
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
  if (path === "/api/v1/spotify/versions") {
    return { track_id: PLANNED.track_id, query: "q", versions: [REMIX_ON_SPOTIFY] };
  }
  if (path === `/api/v1/tracks/${PLANNED.track_id}/pads`) return PADS;
  if (path === `/api/v1/tracks/${PLANNED.track_id}/waveform`) {
    return { peaks: [0.2, 0.8, 0.5, 0.3], duration_seconds: 200, num_peaks: 4 };
  }
  if (path === "/api/v1/tracks/search") return [];
  return undefined;
}

function picksAdded(): number[] {
  return backend.sent("POST", APPEND).map((sent) => (sent.body as { track_id: number }).track_id);
}

function renderSetView(): void {
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <SetEditor />
    </QueryClientProvider>,
  );
}

async function openSetViewWithTheCursorOnTheFirstRec(): Promise<void> {
  renderSetView();
  await screen.findByRole("group", { name: SECOND_REC.title });
  press("ArrowDown");
}

async function buttonOnThePlannedSong(name: string): Promise<HTMLElement> {
  const planned = await screen.findByRole("group", { name: PLANNED.title });
  return within(planned).getByRole("button", { name });
}

async function dialogHasClosed(): Promise<void> {
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
}

let backend: BoothBackend;

beforeEach(() => {
  store.setView("booth");
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

  it("leaves Enter to a button outside the plan grid until an arrow key moves the cursor", async () => {
    await openSetViewWithTheCursorOnTheFirstRec();

    tabTo(screen.getByRole("button", { name: SET_NAME }));
    press("Enter");
    expect(await screen.findByRole("menu")).toBeInTheDocument();
    await letRequestsGoOut();
    expect(picksAdded()).toEqual([]);

    press("ArrowDown");
    press("Enter");
    await waitFor(() => expect(picksAdded()).toEqual([SECOND_REC.track_id]));
  });
});

describe("a dialog opened from the Set view", () => {
  it("takes focus, keeps Tab inside, hands focus back when it closes, and leaves Tab alone under the search palette", async () => {
    store.setView("set");
    renderApp();
    const versions = await buttonOnThePlannedSong("Versions");
    tabTo(versions);
    press("Enter");

    const dialog = await screen.findByRole("dialog", { name: "Versions · Night Drive" });
    const close = within(dialog).getByRole("button", { name: "Close" });
    expect(close).toHaveFocus();

    const get = await within(dialog).findByRole("button", { name: "Get" });
    get.focus();
    press("Tab");
    expect(close).toHaveFocus();
    press("Tab", { shiftKey: true });
    expect(get).toHaveFocus();

    get.blur();
    press("Tab");
    expect(close).toHaveFocus();

    press("Escape");
    await dialogHasClosed();
    expect(versions).toHaveFocus();

    press("Enter");
    await screen.findByRole("dialog", { name: "Versions · Night Drive" });
    press("k", { metaKey: true });
    const search = await screen.findByRole("textbox");
    await waitFor(() => expect(search).toHaveFocus());
    press("Tab");
    expect(search).toHaveFocus();
  });

  it("leaves focus where the dialog's own content put it", async () => {
    renderSetView();
    const pads = await buttonOnThePlannedSong("Pads");
    tabTo(pads);
    press("Enter");

    const firstTime = await screen.findByRole("dialog", { name: "Pads · Night Drive" });
    const allStems = await within(firstTime).findByRole("button", { name: "All" });
    await waitFor(() => expect(allStems).toHaveFocus());
    // jsdom never moves focus on Tab; between two controls the dialog must not move it either.
    press("Tab");
    expect(allStems).toHaveFocus();
    press("Escape");
    await dialogHasClosed();
    expect(pads).toHaveFocus();

    press("Enter");
    const again = screen.getByRole("dialog", { name: "Pads · Night Drive" });
    expect(within(again).getByRole("button", { name: "All" })).toHaveFocus();
    press("Escape");
    await dialogHasClosed();
    expect(pads).toHaveFocus();
  });
});
