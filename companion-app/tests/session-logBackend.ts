import { act } from "@testing-library/react";
import { vi } from "vitest";

import { EMPTY_ABLETON_STATE } from "../src/types";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  SONG_ROLES,
  songOn,
  type Answer,
  type Cell,
  type SeenRequest,
} from "./boothHarness";

export const NIGHT_DRIVE = { track_id: 7, title: "Night Drive" };
export const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
export const HALF_LIGHT = { track_id: 9, title: "Half Light" };
const SONGS = [NIGHT_DRIVE, CLUB_MIX, HALF_LIGHT];

const SESSIONS = "/api/v1/sessions";
const PLAYS_PATH = /^\/api\/v1\/sessions\/(\d+)\/plays$/;
const END_PATH = /^\/api\/v1\/sessions\/(\d+)\/end$/;

export function preparedSetInLive(): Cell[] {
  return [
    ...songOn(0, "a", NIGHT_DRIVE),
    ...songOn(0, "b", CLUB_MIX),
    ...songOn(1, "a", HALF_LIGHT),
  ];
}

export function clips(
  side: "a" | "b",
  scene: number,
  roles: readonly string[] = SONG_ROLES,
): Record<string, number> {
  return Object.fromEntries(roles.map((role) => [`${LIVE_TRACK_OF[`${role}_${side}`]}`, scene]));
}

// The backend writes UTC without a zone marker.
export function backendTime(ms: number = Date.now()): string {
  return new Date(ms).toISOString().slice(0, -1);
}

export function minutesAgo(minutes: number): string {
  return backendTime(Date.now() - minutes * 60_000);
}

interface Play {
  track_id: number;
  played_at: string;
  position_in_set: number;
  energy_at_play: number | null;
  transition_type: string | null;
  title: string | null;
  artist: string | null;
}

interface Session {
  id: number;
  name: string | null;
  notes: string | null;
  started_at: string;
  ended_at: string | null;
  plays: Play[];
}

function refused(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function answeredLater(body: () => unknown): { response: Response; send: () => void } {
  let stream!: ReadableStreamDefaultController<Uint8Array>;
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        stream = controller;
      },
    }),
    { headers: { "content-type": "application/json" } },
  );
  return {
    response,
    send() {
      stream.enqueue(new TextEncoder().encode(JSON.stringify(body())));
      stream.close();
    },
  };
}

export class SessionLog {
  public current: Session | null = null;
  public startsToRefuse = 0;
  public playsToRefuse = 0;
  public refusesEnds = false;
  public holdsStarts = false;
  public holdsReads = false;
  private heldStarts: Array<() => void> = [];
  private heldReads: Array<() => void> = [];
  private lastId = 0;

  letStartsThrough(): void {
    this.holdsStarts = false;
    for (const send of this.heldStarts.splice(0)) send();
  }

  letReadsThrough(): void {
    this.holdsReads = false;
    for (const send of this.heldReads.splice(0)) send();
  }

  open(startedAt: string, plays: Array<{ track_id: number; played_at: string }> = []): Session {
    this.lastId += 1;
    this.current = {
      id: this.lastId,
      name: null,
      notes: null,
      started_at: startedAt,
      ended_at: null,
      plays: [],
    };
    for (const play of plays) this.log(play.track_id, play.played_at);
    return this.current;
  }

  log(trackId: number, playedAt: string = backendTime()): void {
    const session = this.current!;
    session.plays = [
      ...session.plays,
      {
        track_id: trackId,
        played_at: playedAt,
        position_in_set: session.plays.length + 1,
        energy_at_play: 6,
        transition_type: null,
        title: SONGS.find((song) => song.track_id === trackId)?.title ?? null,
        artist: "Test Artist",
      },
    ];
  }

  answer({ method, path, body }: SeenRequest): unknown {
    if (path === `${SESSIONS}/current`) {
      if (this.holdsReads) {
        const held = answeredLater(() => this.current);
        this.heldReads.push(held.send);
        return held.response;
      }
      return this.current ?? refused(404, "no active session");
    }
    if (method === "POST" && path === SESSIONS) {
      if (this.startsToRefuse > 0) {
        this.startsToRefuse -= 1;
        return refused(500, "database is locked");
      }
      if (this.holdsStarts) {
        const held = answeredLater(() => this.open(backendTime()));
        this.heldStarts.push(held.send);
        return held.response;
      }
      return this.open(backendTime());
    }
    if (method === "POST" && PLAYS_PATH.test(path)) {
      if (this.playsToRefuse > 0) {
        this.playsToRefuse -= 1;
        return refused(500, "database is locked");
      }
      this.log((body as { track_id: number }).track_id);
      return this.current;
    }
    if (method === "POST" && END_PATH.test(path)) {
      if (this.refusesEnds) return refused(500, "database is locked");
      const ended = { ...this.current!, ended_at: backendTime() };
      this.current = null;
      return ended;
    }
    return undefined;
  }
}

const socketsToLive: WebSocket[] = [];

function alsoReachSocketsTheAppOpens(): void {
  const HarnessSocket = globalThis.WebSocket;
  globalThis.WebSocket = class extends HarnessSocket {
    constructor(url: string | URL) {
      super(url);
      socketsToLive.push(this);
    }
  };
}

export function boothWith(
  log: SessionLog,
  inLive: () => Cell[],
  answersFirst: Answer = () => undefined,
): BoothBackend {
  const backend = new BoothBackend((request) => {
    const answeredFirst = answersFirst(request);
    if (answeredFirst !== undefined) return answeredFirst;
    const { path, body } = request;
    if (path.endsWith("/ableton/decks")) return { columns: LIVE_TRACK_OF, cells: inLive() };
    if (path.endsWith("/recommend/by-column")) {
      return { column: (body as { column: string }).column, combo_size: 0, recs: [] };
    }
    return log.answer(request);
  });
  alsoReachSocketsTheAppOpens();
  return backend;
}

export function decksThisBrowserLoaded(): Record<string, { track_id: number }> {
  const saved = window.localStorage.getItem("dance.companion.state.v2");
  return saved ? JSON.parse(saved).loadedDecks : {};
}

export function sessionsStarted(backend: BoothBackend): number {
  return backend.sent("POST", SESSIONS).length;
}

export function playsLogged(backend: BoothBackend): number[] {
  return backend.requests
    .filter((request) => request.method === "POST" && PLAYS_PATH.test(request.path))
    .map((request) => (request.body as { track_id: number }).track_id);
}

export function journeysAsked(backend: BoothBackend): Array<number[] | undefined> {
  return backend
    .sent("POST", "/api/v1/recommend/by-column")
    .map((request) => (request.body as { trailing_track_ids?: number[] }).trailing_track_ids);
}

async function theAppShowsIt(): Promise<void> {
  await act(async () => {
    await new Promise((shown) => requestAnimationFrame(shown));
  });
}

export async function secondsPass(seconds: number): Promise<void> {
  act(() => {
    vi.advanceTimersByTime(seconds * 1000);
  });
  await theAppShowsIt();
}

export async function liveNowPlays(
  backend: BoothBackend,
  playing: Record<string, number>,
): Promise<void> {
  backend.liveNowPlays(playing);
  await theAppShowsIt();
}

export async function liveStops(): Promise<void> {
  const stopped = JSON.stringify({ ...EMPTY_ABLETON_STATE, tempo: 128, is_playing: false });
  act(() => {
    for (const socket of socketsToLive) {
      if (socket.url.endsWith("/ws")) socket.onmessage?.({ data: stopped } as MessageEvent);
    }
  });
  await theAppShowsIt();
}
