import { act, configure, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { store } from "../src/store";
import type { Recommendation, SpotifyTrackHit, Track } from "../src/types";
import { BoothBackend, LIVE_TRACK_OF, renderApp, type SeenRequest } from "./boothHarness";

const SET_ID = 1;
const PALETTE = "Find & vibe search";
const APPEND_PATH = `/api/v1/sets/${SET_ID}/plan/append`;
const LOAD_PATH = "/api/v1/ableton/load-track";
const INGEST_PATH = "/api/v1/pipeline/ingest/track";
const NEW_TRACK_ID = 31;
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
  byVibe: Recommendation[];
  onSpotify: SpotifyTrackHit[];
  alreadyPlanned: number[];
  whenLoading: () => unknown;
  whenPlanning: (() => Response) | null;
}

function aLibrary(given: Partial<Library> = {}) {
  const library: Library = {
    tracks: [inLibrary(NIGHT_DRIVE), inLibrary(HALF_LIGHT)],
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
  return ({ method, path, body }: SeenRequest): unknown => {
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
    if (path === "/api/v1/tracks/search") return library.tracks;
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

async function openPaletteAndType(text: string): Promise<void> {
  fireEvent.keyDown(window, { key: "k", metaKey: true });
  await screen.findByRole("dialog", { name: PALETTE });
  fireEvent.change(searchField(), { target: { value: text } });
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
  fireEvent.click(screen.getByTitle("Set, Pipeline, Course"));
  fireEvent.click(screen.getByRole("button", { name: "Set" }));
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
  backend.restore();
  Reflect.deleteProperty(Element.prototype, "scrollIntoView");
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
