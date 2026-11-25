import { act, configure, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { store } from "../src/store";
import type { Recommendation, SpotifyTrackHit, Track } from "../src/types";
import { BoothBackend, LIVE_TRACK_OF, renderApp, type SeenRequest } from "./boothHarness";

const SET_ID = 1;
const PALETTE = "Find & vibe search";
const APPEND_PATH = `/api/v1/sets/${SET_ID}/plan/append`;
const LOAD_PATH = "/api/v1/ableton/load-track";
const INGEST_PATH = "/api/v1/pipeline/ingest/track";
const NEW_TRACK_ID = 31;
const WARP_CHECK_PATH = "/api/v1/ableton/warp-check/0";
const WARP_CHECK_DELAY_MS = 18_000;
const BASS_AT_HALF_TEMPO = "bass is warped at half the tempo of the other stems";
const LIVE_IS_NOT_THERE = "Could not reach Ableton Live over OSC: [Errno 61] Connection refused";

interface Song {
  track_id: number;
  title: string;
}

const NIGHT_DRIVE: Song = { track_id: 7, title: "Night Drive" };
const HALF_LIGHT: Song = { track_id: 9, title: "Half Light" };
const SLOW_TIDE: Song = { track_id: 11, title: "Slow Tide" };
const LOW_SUN: Song = { track_id: 12, title: "Low Sun" };
const EVERY_SONG = [NIGHT_DRIVE, HALF_LIGHT, SLOW_TIDE, LOW_SUN];

const REMIX_ON_SPOTIFY: SpotifyTrackHit = {
  spotify_id: "sp-remix",
  title: "Night Drive - Tinlicker Remix",
  artist: "Test Artist, Tinlicker",
  album: "Night Drive",
  duration_ms: 210_000,
  preview_url: null,
  image_url: null,
  explicit: false,
  popularity: 40,
};

const THE_ACTIVE_SET = {
  id: SET_ID,
  name: "Friday night",
  notes: null,
  is_active: true,
  created_at: "2026-11-01T00:00:00",
  updated_at: "2026-11-01T00:00:00",
  tracks: [],
};

function inLibrary(song: Song, state = "complete"): Track {
  return {
    id: song.track_id,
    file_path: `/music/${song.track_id}.mp3`,
    title: song.title,
    artist: "Test Artist",
    duration_seconds: 240,
    state,
    analysis: null,
    tags: [],
    version_count: 0,
    version_family: null,
  };
}

function foundByVibe(song: Song): Recommendation {
  return {
    track_id: song.track_id,
    score: 0.41,
    reasons: [{ kind: "text_query" }],
    title: song.title,
    artist: "Test Artist",
    file_path: `/music/${song.track_id}.mp3`,
    bpm: 124,
    key_camelot: "8A",
    floor_energy: 6,
  };
}

function planItem(trackId: number) {
  const title = EVERY_SONG.find((song) => song.track_id === trackId)?.title ?? `Track ${trackId}`;
  return {
    track_id: trackId,
    title,
    artist: "Test Artist",
    key_camelot: "8A",
    bpm: 128,
    floor_energy: 6,
  };
}

function loadedOnDeckA() {
  return {
    ok: true,
    scene_index: 0,
    page: 0,
    side: "a",
    track_indices: LIVE_TRACK_OF,
    stems_loaded: 4,
    message: null,
    warnings: [],
  };
}

function failure(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

interface Library {
  tracks: Track[];
  tracksWhenTyped: Record<string, Track[]>;
  byVibe: Recommendation[];
  onSpotify: SpotifyTrackHit[];
  alreadyPlanned: number[];
  whenLoading: () => unknown;
  whenPlanning: (() => Response) | null;
}

function aLibrary(given: Partial<Library> = {}) {
  const library: Library = {
    tracks: [inLibrary(NIGHT_DRIVE), inLibrary(HALF_LIGHT)],
    tracksWhenTyped: {},
    byVibe: [],
    onSpotify: [],
    alreadyPlanned: [],
    whenLoading: loadedOnDeckA,
    whenPlanning: null,
    ...given,
  };
  const songQueue = [...library.alreadyPlanned];
  const plan = () => ({
    set_id: SET_ID,
    queues: { drums: [], bass: [], vocals: [], other: [], song: songQueue.map(planItem) },
  });
  return ({ method, path, query, body }: SeenRequest): unknown => {
    if (path === "/api/v1/sets/active" || path === `/api/v1/sets/${SET_ID}`) return THE_ACTIVE_SET;
    if (path === "/api/v1/sets") return [];
    if (path === `/api/v1/sets/${SET_ID}/plan`) return plan();
    if (path === APPEND_PATH) {
      if (library.whenPlanning) return library.whenPlanning();
      songQueue.push((body as { track_id: number }).track_id);
      return plan();
    }
    if (path === `/api/v1/sets/${SET_ID}/plan-recs`) {
      return { column: "mix", combo_size: 0, recs: [] };
    }
    if (path === "/api/v1/recommend/by-column") {
      return { column: (body as { column: string }).column, combo_size: 0, recs: [] };
    }
    if (path === "/api/v1/ableton/decks") return { columns: LIVE_TRACK_OF, cells: [] };
    if (path === "/api/v1/tracks/search") {
      const typed = new URLSearchParams(query).get("q") ?? "";
      return library.tracksWhenTyped[typed] ?? library.tracks;
    }
    if (path === "/api/v1/recommend/text") return library.byVibe;
    if (path === "/api/v1/spotify/search") return { query: "night drive", hits: library.onSpotify };
    if (path === LOAD_PATH && method === "POST") return library.whenLoading();
    if (path === INGEST_PATH && method === "POST") {
      return { track_id: NEW_TRACK_ID, state: "pending", job_id: "job-1", already_existed: false };
    }
    return undefined;
  };
}

function palette(): HTMLElement {
  return screen.getByRole("dialog", { name: PALETTE });
}

function searchField(): HTMLElement {
  return within(palette()).getByRole("textbox");
}

function searchesFor(text: string): SeenRequest[] {
  return backend
    .sent("GET", "/api/v1/tracks/search")
    .filter((sent) => new URLSearchParams(sent.query).get("q") === text);
}

async function openPaletteAndType(text: string): Promise<void> {
  fireEvent.keyDown(window, { key: "k", metaKey: true });
  await screen.findByRole("dialog", { name: PALETTE });
  fireEvent.change(searchField(), { target: { value: text } });
  await waitFor(() => expect(searchesFor(text)).not.toHaveLength(0));
}

function press(key: string, held: KeyboardEventInit = {}): void {
  fireEvent.keyDown(searchField(), { key, ...held });
}

function libraryRow(title: string): HTMLElement {
  return within(palette()).getByRole("group", { name: `Search result: ${title}` });
}

function spotifyRow(title: string): HTMLElement {
  return within(palette()).getByRole("group", { name: `Spotify result: ${title}` });
}

function highlighted(): (string | null)[] {
  return within(palette())
    .queryAllByRole("group", { current: true })
    .map((row) => row.getAttribute("aria-label"));
}

function button(row: HTMLElement, name: string): HTMLElement {
  return within(row).getByRole("button", { name });
}

function clickWithTheMouse(target: HTMLElement): void {
  act(() => target.focus());
  fireEvent.click(target);
}

function bodiesSentTo(path: string): unknown[] {
  return backend.sent("POST", path).map((sent) => sent.body);
}

function openSetView(): void {
  fireEvent.click(screen.getByRole("tab", { name: "Set" }));
}

configure({ asyncUtilTimeout: 3000 });

let backend: BoothBackend;
let scrolledIntoView: Element[];

beforeEach(() => {
  store.closeCommandBar();
  store.setView("booth");
  store.clearDecks();
  store.clearArm("a");
  store.clearArm("b");
  scrolledIntoView = [];
  Element.prototype.scrollIntoView = function scrollIntoView(this: Element) {
    scrolledIntoView.push(this);
  };
});

afterEach(() => {
  vi.useRealTimers();
  store.clearLoadWarnings();
  backend.restore();
  Reflect.deleteProperty(Element.prototype, "scrollIntoView");
});

describe("the search palette, from the keyboard", () => {
  it("moves a highlight through every row in the order shown", async () => {
    backend = new BoothBackend(
      aLibrary({ byVibe: [foundByVibe(SLOW_TIDE)], onSpotify: [REMIX_ON_SPOTIFY] }),
    );
    renderApp();
    await openPaletteAndType("night drive");
    await within(palette()).findByRole("group", { name: `Search result: ${SLOW_TIDE.title}` });
    await within(palette()).findByText(REMIX_ON_SPOTIFY.title);
    await waitFor(() => expect(button(libraryRow("Night Drive"), "+ Plan")).toBeEnabled());

    expect(highlighted()).toEqual([]);
    press("ArrowUp");
    expect(highlighted()).toEqual([]);

    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Night Drive"]);
    expect(scrolledIntoView.at(-1)).toBe(libraryRow("Night Drive"));
    expect(within(button(libraryRow("Night Drive"), "Load")).getByText("⏎")).toBeInTheDocument();
    expect(within(libraryRow("Half Light")).queryByText("⏎")).not.toBeInTheDocument();

    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Half Light"]);
    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Slow Tide"]);
    press("ArrowDown");
    expect(highlighted()).toEqual([`Spotify result: ${REMIX_ON_SPOTIFY.title}`]);
    expect(scrolledIntoView.at(-1)).toBe(spotifyRow(REMIX_ON_SPOTIFY.title));
    expect(
      within(button(spotifyRow(REMIX_ON_SPOTIFY.title), "+ Plan")).getByText("⏎"),
    ).toBeInTheDocument();
    press("ArrowDown");
    expect(highlighted()).toEqual([`Spotify result: ${REMIX_ON_SPOTIFY.title}`]);

    press("ArrowUp");
    expect(highlighted()).toEqual(["Search result: Slow Tide"]);

    const reachedWithTab = button(libraryRow("Night Drive"), "+ Plan");
    act(() => reachedWithTab.focus());
    fireEvent.keyDown(reachedWithTab, { key: "ArrowUp" });
    expect(highlighted()).toEqual(["Search result: Half Light"]);
    expect(searchField()).toHaveFocus();
    press("ArrowDown");

    fireEvent.click(button(palette(), "BPM range"));
    fireEvent.keyDown(within(palette()).getByLabelText("BPM min"), { key: "ArrowDown" });
    expect(highlighted()).toEqual(["Search result: Slow Tide"]);

    fireEvent.change(searchField(), { target: { value: "night drive " } });
    expect(highlighted()).toEqual([]);

    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Night Drive"]);
    fireEvent.keyDown(window, { key: "k", metaKey: true });
    expect(screen.queryByRole("dialog", { name: PALETTE })).not.toBeInTheDocument();
    fireEvent.keyDown(window, { key: "k", metaKey: true });
    await screen.findByRole("group", { name: "Search result: Night Drive" });
    expect(highlighted()).toEqual([]);
  });

  it("runs the highlighted row with Enter in the Booth, and plans a song with ⌘Enter", async () => {
    backend = new BoothBackend(aLibrary({ onSpotify: [REMIX_ON_SPOTIFY] }));
    renderApp();
    await openPaletteAndType("night drive");
    await within(palette()).findByText(REMIX_ON_SPOTIFY.title);
    await waitFor(() => expect(button(libraryRow("Night Drive"), "+ Plan")).toBeEnabled());

    press("Enter");
    press("Enter", { metaKey: true });

    press("ArrowDown");
    press("ArrowDown");
    press("ArrowDown");
    expect(highlighted()).toEqual([`Spotify result: ${REMIX_ON_SPOTIFY.title}`]);
    press("Enter");
    await waitFor(() =>
      expect(bodiesSentTo(INGEST_PATH)).toEqual([
        {
          spotify_id: "sp-remix",
          title: REMIX_ON_SPOTIFY.title,
          artist: "Test Artist, Tinlicker",
          duration_ms: 210_000,
        },
      ]),
    );
    await within(spotifyRow(REMIX_ON_SPOTIFY.title)).findByRole("button", { name: "✓ added" });

    press("ArrowUp");
    press("ArrowUp");
    expect(highlighted()).toEqual(["Search result: Night Drive"]);
    press("Enter", { shiftKey: true });
    fireEvent.keyDown(button(libraryRow("Half Light"), "Load"), { key: "Enter" });
    press("Enter", { metaKey: true });
    await waitFor(() =>
      expect(bodiesSentTo(APPEND_PATH)).toEqual([
        { role: "song", track_id: NEW_TRACK_ID, after_track_id: null },
        { role: "song", track_id: NIGHT_DRIVE.track_id, after_track_id: null },
      ]),
    );
    expect(backend.sent("POST", LOAD_PATH)).toEqual([]);

    press("Enter");
    await waitFor(() =>
      expect(backend.sent("POST", LOAD_PATH)).toMatchObject([
        { body: { track_id: NIGHT_DRIVE.track_id, kinds: null, side: null, page: 0 } },
      ]),
    );
    expect(bodiesSentTo(APPEND_PATH)).toHaveLength(2);
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: PALETTE })).not.toBeInTheDocument(),
    );
  });

  it("adds the highlighted song to the plan with Enter in the Set view", async () => {
    backend = new BoothBackend(aLibrary());
    renderApp();
    openSetView();
    await screen.findByRole("button", { name: "Open in Live" });
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Half Light" });
    await waitFor(() => expect(button(libraryRow("Half Light"), "+ Plan")).toBeEnabled());

    press("ArrowDown");
    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Half Light"]);
    const row = libraryRow("Half Light");
    expect(within(button(row, "+ Plan")).getByText("⏎")).toBeInTheDocument();
    expect(within(button(row, "Load")).queryByText("⏎")).not.toBeInTheDocument();

    press("Enter");

    await waitFor(() =>
      expect(bodiesSentTo(APPEND_PATH)).toEqual([
        { role: "song", track_id: HALF_LIGHT.track_id, after_track_id: null },
      ]),
    );
    expect(
      await within(libraryRow("Half Light")).findByRole("button", { name: "✓ in plan" }),
    ).toBeDisabled();
    expect(backend.sent("POST", LOAD_PATH)).toEqual([]);
  });

  it("adds the highlighted song to the plan with Enter in the Pipeline view, and loads nothing", async () => {
    backend = new BoothBackend(aLibrary());
    renderApp();
    fireEvent.click(screen.getByRole("tab", { name: "Pipeline" }));
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });
    await waitFor(() => expect(button(libraryRow("Night Drive"), "+ Plan")).toBeEnabled());

    press("ArrowDown");
    press("Enter");

    await waitFor(() =>
      expect(bodiesSentTo(APPEND_PATH)).toEqual([
        { role: "song", track_id: NIGHT_DRIVE.track_id, after_track_id: null },
      ]),
    );
    expect(backend.sent("POST", LOAD_PATH)).toEqual([]);
  });

  it("says there is no active set when Enter has no plan to add to", async () => {
    const library = aLibrary();
    backend = new BoothBackend((request) =>
      request.path === "/api/v1/sets/active" ? failure(404, "no active set") : library(request),
    );
    renderApp();
    openSetView();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });

    press("ArrowDown");
    press("Enter");

    expect(
      await within(libraryRow("Night Drive")).findByText(
        "No active set. Make one in the Set view first.",
      ),
    ).toBeInTheDocument();
    expect(bodiesSentTo(APPEND_PATH)).toEqual([]);
    expect(backend.sent("POST", LOAD_PATH)).toEqual([]);
  });

  it("leaves the last search's rows alone until the rows for the new text are in", async () => {
    backend = new BoothBackend(aLibrary({ tracksWhenTyped: { slow: [inLibrary(SLOW_TIDE)] } }));
    renderApp();
    fireEvent.keyDown(window, { key: "k", metaKey: true });
    await screen.findByRole("group", { name: "Search result: Night Drive" });

    fireEvent.change(searchField(), { target: { value: "slow" } });
    press("ArrowDown");
    press("Enter");

    expect(libraryRow("Night Drive")).toBeInTheDocument();
    expect(highlighted()).toEqual([]);
    await within(palette()).findByRole("group", { name: "Search result: Slow Tide" });
    expect(highlighted()).toEqual([]);
    expect(backend.sent("POST", LOAD_PATH)).toEqual([]);

    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Slow Tide"]);
    press("Enter");
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: PALETTE })).not.toBeInTheDocument(),
    );
    expect(backend.sent("POST", LOAD_PATH)).toMatchObject([
      { body: { track_id: SLOW_TIDE.track_id } },
    ]);
  });

  it("loads a song once, however often Enter or Load is pressed before the palette closes", async () => {
    backend = new BoothBackend(aLibrary());
    renderApp();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });

    press("ArrowDown");
    press("Enter");
    const loaded = await within(libraryRow("Night Drive")).findByRole("button", {
      name: "scene 1",
    });
    press("Enter");
    clickWithTheMouse(loaded);

    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: PALETTE })).not.toBeInTheDocument(),
    );
    expect(backend.sent("POST", LOAD_PATH)).toHaveLength(1);
  });
});

describe("the search palette's rows", () => {
  it("say which songs the plan already has, and mark a song once it is added", async () => {
    backend = new BoothBackend(aLibrary({ alreadyPlanned: [NIGHT_DRIVE.track_id] }));
    renderApp();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });

    expect(
      await within(libraryRow("Night Drive")).findByRole("button", { name: "✓ in plan" }),
    ).toBeDisabled();

    act(() => searchField().blur());
    fireEvent.click(libraryRow("Half Light"));
    expect(searchField()).not.toHaveFocus();

    clickWithTheMouse(button(libraryRow("Half Light"), "+ Plan"));

    expect(
      await within(libraryRow("Half Light")).findByRole("button", { name: "✓ in plan" }),
    ).toBeDisabled();
    expect(bodiesSentTo(APPEND_PATH)).toEqual([
      { role: "song", track_id: HALF_LIGHT.track_id, after_track_id: null },
    ]);
    expect(searchField()).toHaveFocus();
  });

  it("say why a song could not be added to the plan", async () => {
    backend = new BoothBackend(aLibrary({ whenPlanning: () => failure(404, "track not found") }));
    renderApp();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });
    await waitFor(() => expect(button(libraryRow("Night Drive"), "+ Plan")).toBeEnabled());

    fireEvent.click(button(libraryRow("Night Drive"), "+ Plan"));

    expect(await within(libraryRow("Night Drive")).findByRole("alert")).toHaveTextContent(
      "track not found",
    );
    expect(within(palette()).queryByText(/API 404/)).not.toBeInTheDocument();
    expect(button(libraryRow("Night Drive"), "+ Plan")).toBeEnabled();
  });

  it("say why a song cannot be loaded yet, and offer no Load for it", async () => {
    backend = new BoothBackend(
      aLibrary({
        tracks: [
          inLibrary(NIGHT_DRIVE),
          inLibrary(HALF_LIGHT, "pending"),
          inLibrary(SLOW_TIDE, "separating"),
          inLibrary(LOW_SUN, "error"),
        ],
      }),
    );
    renderApp();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });
    await waitFor(() => expect(button(libraryRow("Half Light"), "+ Plan")).toBeEnabled());

    expect(button(libraryRow("Night Drive"), "Load")).toBeEnabled();
    const notReady: [Song, string][] = [
      [HALF_LIGHT, "not processed yet"],
      [SLOW_TIDE, "processing"],
      [LOW_SUN, "failed, see Pipeline"],
    ];
    for (const [song, inWords] of notReady) {
      const row = libraryRow(song.title);
      expect(within(row).getByText(inWords)).toBeInTheDocument();
      expect(within(row).queryByRole("button", { name: "Load" })).not.toBeInTheDocument();
      expect(button(row, "+ Plan")).toBeEnabled();
    }

    press("ArrowDown");
    press("ArrowDown");
    expect(highlighted()).toEqual(["Search result: Half Light"]);
    expect(within(libraryRow("Half Light")).queryByText("⏎")).not.toBeInTheDocument();
    press("Enter");
    press("Enter", { metaKey: true });

    await waitFor(() =>
      expect(bodiesSentTo(APPEND_PATH)).toEqual([
        { role: "song", track_id: HALF_LIGHT.track_id, after_track_id: null },
      ]),
    );
    expect(backend.sent("POST", LOAD_PATH)).toEqual([]);
  });

  it("say in the server's words why a Load failed", async () => {
    backend = new BoothBackend(aLibrary({ whenLoading: () => failure(503, LIVE_IS_NOT_THERE) }));
    renderApp();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });

    fireEvent.click(button(libraryRow("Night Drive"), "Load"));

    expect(await within(libraryRow("Night Drive")).findByRole("alert")).toHaveTextContent(
      LIVE_IS_NOT_THERE,
    );
    expect(within(palette()).queryByText(/API 503/)).not.toBeInTheDocument();
    expect(button(libraryRow("Night Drive"), "Load")).toBeEnabled();
  });
});

describe("a song loaded from the palette", () => {
  it("still gets its warp check after the palette has closed", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"], shouldAdvanceTime: true });
    const library = aLibrary();
    backend = new BoothBackend((request) =>
      request.path === WARP_CHECK_PATH
        ? { ok: true, scene_index: 0, checked: 4, warnings: [BASS_AT_HALF_TEMPO] }
        : library(request),
    );
    renderApp();
    await openPaletteAndType("night");
    await within(palette()).findByRole("group", { name: "Search result: Night Drive" });

    press("ArrowDown");
    press("Enter");
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: PALETTE })).not.toBeInTheDocument(),
    );
    act(() => {
      vi.advanceTimersByTime(WARP_CHECK_DELAY_MS);
    });

    const warnings = await screen.findByRole("status", { name: "Warnings" });
    expect(warnings).toHaveTextContent(`Night Drive → scene 1: ${BASS_AT_HALF_TEMPO}`);
    expect(backend.sent("POST", WARP_CHECK_PATH)).toHaveLength(1);
  });
});

describe("the BPM range popover in the search palette", () => {
  it("closes on the first Escape and leaves the palette for the second", async () => {
    backend = new BoothBackend(aLibrary());
    renderApp();
    await openPaletteAndType("night");
    fireEvent.click(button(palette(), "BPM range"));
    const popover = within(palette()).getByRole("dialog", { name: "BPM range" });

    fireEvent.keyDown(within(popover).getByLabelText("BPM min"), { key: "Escape" });

    expect(screen.queryByRole("dialog", { name: "BPM range" })).not.toBeInTheDocument();
    expect(palette()).toBeInTheDocument();

    press("Escape");

    expect(screen.queryByRole("dialog", { name: PALETTE })).not.toBeInTheDocument();
  });
});

describe("a song on Spotify that the library already holds", () => {
  const HELD_ON_SPOTIFY: SpotifyTrackHit = {
    ...REMIX_ON_SPOTIFY,
    spotify_id: "sp-half-light",
    title: HALF_LIGHT.title,
    artist: "Test Artist",
    library_track_id: HALF_LIGHT.track_id,
    library_state: "complete",
  };

  it("is marked as held, and planning it adds the library's track without a download", async () => {
    backend = new BoothBackend(aLibrary({ onSpotify: [HELD_ON_SPOTIFY, REMIX_ON_SPOTIFY] }));
    renderApp();
    await openPaletteAndType("half light");
    await within(palette()).findByText(REMIX_ON_SPOTIFY.title);

    const held = spotifyRow(HELD_ON_SPOTIFY.title);
    expect(within(held).getByText("in your library")).toBeInTheDocument();
    expect(within(spotifyRow(REMIX_ON_SPOTIFY.title)).queryByText("in your library")).not.toBeInTheDocument();

    clickWithTheMouse(button(held, "+ Plan"));

    await waitFor(() =>
      expect(bodiesSentTo(APPEND_PATH)).toEqual([
        { role: "song", track_id: HALF_LIGHT.track_id, after_track_id: null },
      ]),
    );
    expect(bodiesSentTo(INGEST_PATH)).toEqual([]);
    await within(held).findByRole("button", { name: "✓ in plan" });
  });

  it("says so when the held copy is not ready to play", async () => {
    const stillProcessing = { ...HELD_ON_SPOTIFY, library_state: "analyzing" };
    backend = new BoothBackend(aLibrary({ onSpotify: [stillProcessing] }));
    renderApp();
    await openPaletteAndType("half light");

    const held = await within(palette()).findByRole("group", {
      name: `Spotify result: ${HELD_ON_SPOTIFY.title}`,
    });
    expect(within(held).getByText("in your library, processing")).toBeInTheDocument();
  });
});
