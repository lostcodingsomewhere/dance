import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";

const PADS_PER_PAGE = 5;
const SONG_ROLES = ["drums", "bass", "vocals", "other", "mix"] as const;
const CUE_LABELS = ["intro", "buildup", "drop", "breakdown", "outro"];
const LIVE_TRACK_OF: Record<string, number> = {
  drums_a: 0, bass_a: 1, vocals_a: 2, other_a: 3,
  drums_b: 4, bass_b: 5, vocals_b: 6, other_b: 7,
  mix_a: 8, mix_b: 9,
};
const ORIGINAL = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
const THIRD_SONG = { track_id: 9, title: "Half Light" };

interface Cell {
  scene_index: number;
  kind: string;
  track_id: number;
  title: string;
  cue: number | null;
  cue_label: string | null;
  cue_silent: boolean;
  [field: string]: unknown;
}

function songOn(page: number, side: "a" | "b", song: { track_id: number; title: string }): Cell[] {
  return SONG_ROLES.flatMap((role) =>
    CUE_LABELS.map((label, row) => ({
      scene_index: page * PADS_PER_PAGE + row,
      kind: `${role}_${side}`,
      ...song,
      stem_file_id: null,
      artist: "Test Artist",
      bpm: 128,
      key_camelot: "8A",
      floor_energy: 6,
      duration_seconds: 240,
      cue: row + 1,
      cue_label: label,
      cue_silent: role === "vocals" && row === 0,
    })),
  );
}

function preparedSetInLive(): Cell[] {
  const cells = [...songOn(0, "a", ORIGINAL), ...songOn(0, "b", CLUB_MIX), ...songOn(1, "a", THIRD_SONG)];
  const droppedInByHand = cells.find((cell) => cell.kind === "bass_b" && cell.scene_index === 3)!;
  Object.assign(droppedInByHand, { cue: null, cue_label: null, cue_silent: false });
  return cells;
}

interface SeenRequest {
  method: string;
  path: string;
  query: string;
}

let requests: SeenRequest[];
let sockets: RecordingSocket[];

class RecordingSocket {
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onclose: (() => void) | null = null;
  public onerror: (() => void) | null = null;
  constructor(public url: string) {
    sockets.push(this);
  }
  close() {}
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function fakeBackend(url: string, init: RequestInit = {}): Promise<Response> {
  const [path, query = ""] = url.split("?");
  requests.push({ method: init.method ?? "GET", path, query });
  if (path.endsWith("/ableton/decks")) {
    return json({ columns: LIVE_TRACK_OF, cells: preparedSetInLive() });
  }
  if (path.includes("/sessions/current")) return json({ detail: "no active session" }, 404);
  if (path.includes("/sets/active")) return json({ detail: "no active set" }, 404);
  if (path.endsWith("/tracks") || path.endsWith("/regions")) return json([]);
  return json({});
}

function liveNowPlays(playingClips: Record<string, number>) {
  const frame = {
    tempo: 128,
    is_playing: true,
    beat: 0,
    playing_clips: playingClips,
    track_volumes: {},
    track_meters: {},
    playing_positions: {},
    crossfader: 0,
    soloed_kinds: [],
    deck_map_revision: 1,
  };
  act(() => {
    for (const socket of sockets) {
      if (socket.url.endsWith("/ws")) socket.onmessage?.({ data: JSON.stringify(frame) } as MessageEvent);
    }
  });
}

function renderApp() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

function grid(): HTMLElement {
  return screen.getByTestId("scene-grid");
}

function deck(side: "A" | "B"): HTMLElement {
  return screen.getByRole("region", { name: `Deck ${side}` });
}

function songOnDeck(side: "A" | "B", title: string): HTMLElement[] {
  return within(deck(side)).queryAllByText(title);
}

const originalFetch = globalThis.fetch;
const originalWebSocket = globalThis.WebSocket;

beforeEach(() => {
  requests = [];
  sockets = [];
  window.localStorage.clear();
  globalThis.WebSocket = RecordingSocket as unknown as typeof WebSocket;
  globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
    fakeBackend(typeof input === "string" ? input : input.toString(), init),
  ) as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  globalThis.WebSocket = originalWebSocket;
  vi.restoreAllMocks();
});

describe("a prepared Set in the Booth", () => {
  it("shows one page of five cues at a time and plays a deck from the cue you pick", async () => {
    renderApp();

    const pageOne = await screen.findByRole("button", { name: "Page 1" });
    const pageTwo = screen.getByRole("button", { name: "Page 2" });
    expect(pageOne).toHaveAttribute("aria-pressed", "true");
    expect(pageTwo).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("button", { name: "Page 3" })).not.toBeInTheDocument();

    expect(within(grid()).getAllByText("Night Drive")).toHaveLength(4 * PADS_PER_PAGE);
    expect(within(grid()).getAllByText("Night Drive (Club Mix)")).toHaveLength(4 * PADS_PER_PAGE);
    expect(within(grid()).queryByText("Half Light")).not.toBeInTheDocument();
    expect(within(grid()).getAllByText("3 drop")).toHaveLength(8);
    expect(within(grid()).getAllByText("4 breakdown")).toHaveLength(7);
    expect(
      within(grid()).getByTitle("Vocals A: Night Drive — Test Artist — cue 1, intro, silent here — tap to fire"),
    ).toBeInTheDocument();
    expect(
      within(grid()).getByTitle("Bass B: Night Drive (Club Mix) — Test Artist — tap to fire"),
    ).toBeInTheDocument();

    expect(songOnDeck("A", "Night Drive")).not.toHaveLength(0);
    expect(songOnDeck("B", "Night Drive (Club Mix)")).not.toHaveLength(0);

    fireEvent.click(pageTwo);
    expect(pageTwo).toHaveAttribute("aria-pressed", "true");
    expect(within(grid()).getAllByText("Half Light")).toHaveLength(4 * PADS_PER_PAGE);
    expect(within(grid()).queryByText("Night Drive")).not.toBeInTheDocument();
    expect(songOnDeck("A", "Half Light")).not.toHaveLength(0);
    expect(songOnDeck("B", "Night Drive (Club Mix)")).toHaveLength(0);
    expect(within(deck("B")).queryByRole("button", { name: /cue/i })).not.toBeInTheDocument();

    fireEvent.click(within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" }));
    await waitFor(() =>
      expect(requests).toContainEqual({
        method: "POST",
        path: "/api/v1/ableton/transport/fire-deck/a/7",
        query: "loaded_only=true",
      }),
    );

    liveNowPlays({ "0": 2, "1": 2, "2": 2, "3": 2 });
    expect(await screen.findByRole("button", { name: "Page 1, playing" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Page 2" })).toHaveAttribute("aria-pressed", "true");
    expect(songOnDeck("A", "Night Drive")).not.toHaveLength(0);
    expect(songOnDeck("A", "Half Light")).toHaveLength(0);
    expect(within(deck("A")).getByRole("button", { name: "Deck A cue 3, drop" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(within(deck("A")).getByRole("button", { name: "Deck A cue 1, intro" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});
