import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, within } from "@testing-library/react";
import { vi } from "vitest";

import { App } from "../src/App";

export const PADS_PER_PAGE = 5;
export const SONG_ROLES = ["drums", "bass", "vocals", "other", "mix"] as const;
export const CUE_LABELS = ["intro", "buildup", "drop", "breakdown", "outro"];
export const LIVE_TRACK_OF: Record<string, number> = {
  drums_a: 0, bass_a: 1, vocals_a: 2, other_a: 3,
  drums_b: 4, bass_b: 5, vocals_b: 6, other_b: 7,
  mix_a: 8, mix_b: 9,
};

export interface Cell {
  scene_index: number;
  kind: string;
  track_id: number;
  title: string;
  cue: number | null;
  cue_label: string | null;
  cue_silent: boolean;
  [field: string]: unknown;
}

export function songOn(
  page: number,
  side: "a" | "b",
  song: { track_id: number; title: string },
): Cell[] {
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

export interface SeenRequest {
  method: string;
  path: string;
  query: string;
  body: unknown;
}

export type Answer = (request: SeenRequest) => unknown;

// The app keeps one socket per module, so a later test must reach the one an earlier test opened.
const socketsTheAppOpened: RecordingSocket[] = [];

class RecordingSocket {
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onclose: (() => void) | null = null;
  public onerror: (() => void) | null = null;
  constructor(public url: string) {
    socketsTheAppOpened.push(this);
  }
  close() {}
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function whenNothingElseAnswers({ path }: SeenRequest): Response {
  if (path.includes("/sessions/current")) return json({ detail: "no active session" }, 404);
  if (path.includes("/sets/active")) return json({ detail: "no active set" }, 404);
  if (path.endsWith("/tracks") || path.endsWith("/regions")) return json([]);
  return json({});
}

export class BoothBackend {
  public requests: SeenRequest[] = [];
  private readonly originalFetch = globalThis.fetch;
  private readonly originalWebSocket = globalThis.WebSocket;

  constructor(answer: Answer) {
    window.localStorage.clear();
    globalThis.WebSocket = RecordingSocket as unknown as typeof WebSocket;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const [path, query = ""] = (typeof input === "string" ? input : input.toString()).split("?");
      const seen: SeenRequest = {
        method: init.method ?? "GET",
        path,
        query,
        body: typeof init.body === "string" ? JSON.parse(init.body) : null,
      };
      this.requests.push(seen);
      const answered = answer(seen);
      if (answered instanceof Response) return answered;
      return answered === undefined ? whenNothingElseAnswers(seen) : json(answered);
    }) as unknown as typeof fetch;
  }

  sent(method: string, path: string): SeenRequest[] {
    return this.requests.filter((request) => request.method === method && request.path === path);
  }

  liveNowPlays(playingClips: Record<string, number>, alsoInLive: Record<string, unknown> = {}): void {
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
      ...alsoInLive,
    };
    act(() => {
      for (const socket of socketsTheAppOpened) {
        if (socket.url.endsWith("/ws")) {
          socket.onmessage?.({ data: JSON.stringify(frame) } as MessageEvent);
        }
      }
    });
  }

  restore(): void {
    globalThis.fetch = this.originalFetch;
    globalThis.WebSocket = this.originalWebSocket;
    vi.restoreAllMocks();
  }
}

export function renderApp() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

export function grid(): HTMLElement {
  return screen.getByTestId("scene-grid");
}

export function deck(side: "A" | "B"): HTMLElement {
  return screen.getByRole("region", { name: `Deck ${side}` });
}

export function songOnDeck(side: "A" | "B", title: string): HTMLElement[] {
  return within(deck(side)).queryAllByText(title);
}
