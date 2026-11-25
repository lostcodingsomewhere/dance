import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { expect, vi } from "vitest";

import { App, createAppQueryClient } from "../src/App";
import { store } from "../src/store";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  PADS_PER_PAGE,
  songOn,
  type Answer,
  type Cell,
  type SeenRequest,
} from "./boothHarness";

export const ORIGINAL = { track_id: 7, title: "Night Drive" };
export const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
export const RECOMMENDED = { track_id: 9, title: "Half Light" };
export const SONG_COLUMN = "mix";
export const WARP_CHECK_DELAY_MS = 18_000;
export const LIVE_IS_DOWN = "Could not reach Ableton Live over OSC: [Errno 61] Connection refused";

const FX_RETURNS_IN_LIVE = {
  filter: { a: false, b: false },
  reverb: { a: false, b: false },
  delay: { a: false, b: false },
  discovered: { returns: { filter: 11, reverb: 12, delay: 13 }, scenes: { riser: 0 } },
};

export function recommendation() {
  return {
    track_id: RECOMMENDED.track_id,
    stem_file_id: null,
    track_title: RECOMMENDED.title,
    track_artist: "Test Artist",
    bpm: 128,
    key_camelot: "8A",
    floor_energy: 6,
    score: 0.82,
    score_breakdown: { embedding: 0.9, bpm: 0.7 },
    reasons: ["close vibe"],
  };
}

function recommendedTrack() {
  return {
    id: RECOMMENDED.track_id,
    file_path: "/music/half-light.mp3",
    title: RECOMMENDED.title,
    artist: "Test Artist",
    duration_seconds: 240,
    state: "complete",
    analysis: null,
    tags: [],
  };
}

function forgetThePreviousTest(): void {
  store.clearLoadWarnings();
  store.setPreviewing(null);
  store.clearDecks();
  store.clearArm("a");
  store.clearArm("b");
  store.setBoothPage(0);
  store.setView("booth");
}

// jsdom has no pointer capture, which the crossfader's drag needs.
function letPointersBeCaptured(): void {
  Object.assign(Element.prototype, {
    setPointerCapture: () => {},
    releasePointerCapture: () => {},
    hasPointerCapture: () => true,
  });
}

export class BoothMidSet {
  public inLive: Cell[] = [...songOn(0, "a", ORIGINAL), ...songOn(0, "b", CLUB_MIX)];
  public liveAnswers: Answer = () => undefined;
  private readonly backend: BoothBackend;

  constructor() {
    forgetThePreviousTest();
    letPointersBeCaptured();
    this.backend = new BoothBackend((request) => this.answer(request));
  }

  private answer(request: SeenRequest): unknown {
    const fromLive = this.liveAnswers(request);
    if (fromLive !== undefined) return fromLive;
    const { path, body } = request;
    if (path.endsWith("/ableton/decks")) return { columns: LIVE_TRACK_OF, cells: this.inLive };
    if (path.endsWith("/ableton/fx/state")) return FX_RETURNS_IN_LIVE;
    if (path.endsWith("/recommend/by-column")) {
      const column = (body as { column: string }).column;
      return { column, combo_size: 0, recs: column === SONG_COLUMN ? [recommendation()] : [] };
    }
    if (path.endsWith(`/tracks/${RECOMMENDED.track_id}`)) return recommendedTrack();
    return undefined;
  }

  show(): void {
    render(
      <QueryClientProvider client={createAppQueryClient()}>
        <App />
      </QueryClientProvider>,
    );
    this.liveNowPlays({});
  }

  async open(): Promise<void> {
    this.show();
    await screen.findByRole("button", { name: "Page 1" });
    await screen.findByRole("group", { name: RECOMMENDED.title });
  }

  liveNowPlays(playingClips: Record<string, number>): void {
    this.backend.liveNowPlays(playingClips);
  }

  sent(method: string, path: string): SeenRequest[] {
    return this.backend.sent(method, `/api/v1${path}`);
  }

  close(): void {
    vi.useRealTimers();
    this.backend.restore();
  }
}

export function refused(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function isCommand({ method, path }: SeenRequest, endOfPath: string): boolean {
  return method !== "GET" && path.endsWith(endOfPath);
}

export function everyCommandFails(request: SeenRequest): unknown {
  return request.method !== "GET" && request.path.includes("/ableton/")
    ? refused(503, LIVE_IS_DOWN)
    : undefined;
}

export function loadAnswer(landed: {
  page: number;
  side: "a" | "b";
  stems: number;
  warnings: string[];
}) {
  return {
    ok: true,
    scene_index: landed.page * PADS_PER_PAGE,
    page: landed.page,
    side: landed.side,
    track_indices: LIVE_TRACK_OF,
    stems_loaded: landed.stems,
    message: null,
    warnings: landed.warnings,
  };
}

export function recommended(): HTMLElement {
  return screen.getByRole("group", { name: RECOMMENDED.title });
}

export function loadButton(side: "A" | "B"): HTMLElement {
  return within(recommended()).getByRole("button", { name: `⤒ ${side}` });
}

export function loadOnto(side: "A" | "B"): void {
  fireEvent.click(loadButton(side));
}

export function press(name: string): void {
  fireEvent.click(screen.getByRole("button", { name }));
}

export function warnings(): Promise<HTMLElement> {
  return screen.findByRole("status", { name: "Warnings" });
}

export function warningLines(): HTMLElement[] {
  const banner = screen.queryByRole("status", { name: "Warnings" });
  return banner ? within(banner).queryAllByRole("listitem") : [];
}

export async function expectWarningLines(expected: string[]): Promise<void> {
  await waitFor(() => {
    const lines = warningLines();
    expect(lines.map((line) => line.textContent)).toHaveLength(expected.length);
    expected.forEach((text, at) => expect(lines[at]).toHaveTextContent(text));
  });
}
