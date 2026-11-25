import { onlineManager } from "@tanstack/react-query";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { store } from "../src/store";
import type { AbletonState } from "../src/types";
import {
  BoothBackend,
  LIVE_TRACK_OF,
  renderApp,
  songOn,
  type Cell,
  type SeenRequest,
} from "./boothHarness";

const API = "/api/v1";
const SERVER_DOWN_BANNER =
  "The Dance server is not answering. This screen shows the last known state.";
const SERVER_DOWN_TOOLTIP =
  "The Dance server is not answering, so this screen cannot reach Ableton Live. Start the server again.";
const UNCONFIRMED_RECORDING =
  "Could not confirm an armed Recorder track — Live may capture nothing. Check Live is reachable, then retry.";
const RECORD_NOT_SENT = "[Errno 65] No route to host";
const RECORD_DID_NOT_CHANGE =
  "Live could not be reached, so recording did not change. Check that Live is open, then try again.";

const NIGHT_DRIVE = { track_id: 7, title: "Night Drive" };
const CLUB_MIX = { track_id: 8, title: "Night Drive (Club Mix)" };
const HALF_LIGHT = { track_id: 9, title: "Half Light" };
const LOW_TIDE = { track_id: 10, title: "Low Tide" };
const SLOW_BURN = { track_id: 11, title: "Slow Burn" };
const SLOW_BURN_DUB = { track_id: 12, title: "Slow Burn (Dub)" };

const SPOTIFY_KEYS = {
  key: "spotify",
  label: "Spotify keys",
  status: "optional",
  detail: "Add DANCE_SPOTIFY_CLIENT_ID to search Spotify.",
  required: false,
};
const FFMPEG = {
  key: "ffmpeg",
  label: "ffmpeg",
  status: "missing",
  detail: "Required to read downloads. brew install ffmpeg",
  required: true,
};
const ALL_READY = { ok: true, all_green: true, checks: [] };

type Server = "up" | "refusing connections" | "down behind the dev proxy";
type AppSocket = { url: string; onmessage: ((event: MessageEvent) => void) | null };

const LIVE_AT_REST: AbletonState = {
  tempo: 120,
  is_playing: false,
  beat: 0,
  playing_clips: {},
  track_volumes: {},
  track_meters: {},
  playing_positions: {},
  crossfader: 0,
  soloed_kinds: [],
  deck_map_revision: 1,
};

// The app keeps one socket for the whole file, so the list outlives each test.
const socketsOpenedHere: AppSocket[] = [];

let backend: BoothBackend;
let server: Server;
let liveAnswers: boolean;
let inLive: Cell[];
let setup: unknown;
let session: unknown;
let recordAnswer: unknown;
let liveState: AbletonState;

function theDanceServer({ method, path }: SeenRequest): unknown {
  if (server === "refusing connections") throw new TypeError("Failed to fetch");
  if (server === "down behind the dev proxy") return new Response("", { status: 500 });
  if (path === `${API}/ableton/health`) return { live_reachable: liveAnswers };
  if (path === `${API}/ableton/decks`) return { columns: LIVE_TRACK_OF, cells: inLive };
  if (path === `${API}/health/deps`) return setup;
  if (path === `${API}/sessions/current` && session) return session;
  if (method === "POST" && path === `${API}/ableton/record/on`) return recordAnswer;
  if (method === "POST" && path === `${API}/ableton/record/off`) {
    return { ok: true, recording: false, armed_track: null, warnings: [] };
  }
  return undefined;
}

function rememberTheSocketsTheAppOpens(): void {
  const HarnessSocket = globalThis.WebSocket as unknown as new (url: string) => AppSocket;
  globalThis.WebSocket = class extends HarnessSocket {
    constructor(url: string) {
      super(url);
      socketsOpenedHere.push(this);
    }
  } as unknown as typeof WebSocket;
}

function liveSends(change: Partial<AbletonState>): void {
  liveState = { ...liveState, ...change };
  act(() => {
    for (const socket of socketsOpenedHere) {
      if (socket.url.endsWith("/ws")) {
        socket.onmessage?.({ data: JSON.stringify(liveState) } as MessageEvent);
      }
    }
  });
}

function theServerNeverAnswersAHealthCheck(): void {
  const answered = globalThis.fetch;
  globalThis.fetch = ((input: RequestInfo | URL, init: RequestInit = {}) => {
    if (!String(input).startsWith(`${API}/ableton/health`)) return answered(input, init);
    return new Promise<Response>((_resolve, reject) => {
      init.signal?.addEventListener("abort", () =>
        reject(new DOMException("The request was given up on.", "AbortError")),
      );
    });
  }) as typeof fetch;
}

function clockIsOurs(): void {
  vi.useFakeTimers({
    toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "Date"],
  });
}

async function timePasses(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function secondsPassOneAtATime(seconds: number): Promise<void> {
  for (let second = 0; second < seconds; second += 1) {
    await timePasses(1_000);
  }
}

function header(): HTMLElement {
  return screen.getByRole("banner");
}

function noticeUnderTheHeader(text: string): HTMLElement {
  const notice = screen.getByText(text);
  expect(notice).toHaveAttribute("role", "alert");
  expect(header()).not.toContainElement(notice);
  expect(
    header().compareDocumentPosition(notice) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
  return notice;
}

function at(bpm: number, cells: Cell[]): Cell[] {
  return cells.map((cell) => ({ ...cell, bpm }));
}

function deckAPlaysRow(row: number): Record<string, number> {
  return Object.fromEntries(
    ["drums_a", "bass_a", "vocals_a", "other_a"].map((kind) => [LIVE_TRACK_OF[kind], row]),
  );
}

function tempoSent(): unknown[] {
  return backend.sent("POST", `${API}/ableton/tempo`).map((sent) => sent.body);
}

beforeEach(() => {
  server = "up";
  liveAnswers = true;
  inLive = [];
  setup = ALL_READY;
  session = null;
  recordAnswer = { ok: true, recording: true, armed_track: 10, warnings: [] };
  liveState = LIVE_AT_REST;
  store.setView("booth");
  store.setBoothPage(0);
  backend = new BoothBackend(theDanceServer);
  rememberTheSocketsTheAppOpens();
});

afterEach(() => {
  onlineManager.setOnline(true);
  vi.useRealTimers();
  backend.restore();
});

describe("the health light", () => {
  it("says Live, Live not answering or Server down, whichever is true", async () => {
    clockIsOurs();
    renderApp();
    await timePasses(100);
    expect(within(header()).getByText("Live")).toBeInTheDocument();
    expect(screen.queryByText(SERVER_DOWN_BANNER)).not.toBeInTheDocument();

    liveAnswers = false;
    await secondsPassOneAtATime(3);
    expect(within(header()).getByText("Live")).toBeInTheDocument();
    await secondsPassOneAtATime(5);
    expect(within(header()).getByText("Live not answering")).toBeInTheDocument();
    expect(screen.queryByText(SERVER_DOWN_BANNER)).not.toBeInTheDocument();

    server = "refusing connections";
    await secondsPassOneAtATime(8);
    expect(within(header()).getByText("Server down")).toBeInTheDocument();
    expect(screen.getByText(SERVER_DOWN_BANNER)).toHaveAttribute("role", "alert");

    server = "up";
    liveAnswers = true;
    await secondsPassOneAtATime(3);
    expect(within(header()).getByText("Live")).toBeInTheDocument();
    expect(screen.queryByText(SERVER_DOWN_BANNER)).not.toBeInTheDocument();
  });

  it("says Server down once the Dance server stops while Live was answering", async () => {
    clockIsOurs();
    renderApp();
    await timePasses(100);
    expect(within(header()).getByText("Live")).toBeInTheDocument();

    server = "refusing connections";
    await secondsPassOneAtATime(10);

    expect(within(header()).getByText("Server down")).toBeInTheDocument();
    expect(screen.getByText(SERVER_DOWN_BANNER)).toHaveAttribute("role", "alert");
  });

  it("says Server down, not Live offline, when it opens with the Dance server down", async () => {
    server = "down behind the dev proxy";
    renderApp();

    const light = await within(header()).findByText("Server down");
    expect(light.closest("[title]")).toHaveAttribute("title", SERVER_DOWN_TOOLTIP);
    expect(within(header()).queryByText("Live offline")).not.toBeInTheDocument();
    expect(screen.getByText(SERVER_DOWN_BANNER)).toHaveAttribute("role", "alert");
  });

  it("keeps asking after the computer loses its network, since the server is on this computer", async () => {
    clockIsOurs();
    renderApp();
    await timePasses(100);
    expect(within(header()).getByText("Live")).toBeInTheDocument();

    act(() => {
      window.dispatchEvent(new Event("offline"));
    });
    liveAnswers = false;
    await secondsPassOneAtATime(8);

    expect(within(header()).getByText("Live not answering")).toBeInTheDocument();
  });

  it("says Server down when the Dance server takes too long to answer", async () => {
    clockIsOurs();
    theServerNeverAnswersAHealthCheck();
    renderApp();
    await secondsPassOneAtATime(3);
    expect(within(header()).getByText("Checking…")).toBeInTheDocument();
    expect(screen.queryByText(SERVER_DOWN_BANNER)).not.toBeInTheDocument();

    await secondsPassOneAtATime(2);
    expect(within(header()).getByText("Server down")).toBeInTheDocument();
    expect(screen.getByText(SERVER_DOWN_BANNER)).toBeInTheDocument();
  });
});

describe("the tabs", () => {
  it("keeps Booth, Set, Pipeline and Course in view in the Booth", async () => {
    renderApp();
    const tabs = within(header()).getByRole("tablist");

    expect(within(tabs).getByRole("tab", { name: "Booth" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(within(tabs).getByRole("tab", { name: "Set" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(within(tabs).getByRole("tab", { name: "Pipeline" })).toHaveAttribute(
      "aria-selected",
      "false",
    );
    expect(within(tabs).getByRole("link", { name: "Course" })).toHaveAttribute(
      "href",
      "/course/index.html",
    );
    expect(within(header()).queryByTitle("Set, Pipeline, Course")).not.toBeInTheDocument();

    fireEvent.click(within(tabs).getByRole("tab", { name: "Set" }));
    expect(await screen.findByText("No active set yet")).toBeInTheDocument();
    expect(within(tabs).getByRole("tab", { name: "Set" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    fireEvent.click(within(tabs).getByRole("tab", { name: "Booth" }));
    expect(await screen.findByTestId("scene-grid")).toBeInTheDocument();
    expect(within(tabs).getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Booth",
      "Set",
      "Pipeline",
    ]);
  });
});

describe("the tempo buttons", () => {
  it("sit side by side and are offered only where the tempo stays in range", async () => {
    renderApp();
    liveSends({ tempo: 128 });
    const double = within(header()).getByRole("button", { name: "×2" });
    const halve = within(header()).getByRole("button", { name: "÷2" });

    await within(header()).findByText("128.0");
    expect(double).toBeDisabled();
    expect(halve).toBeEnabled();
    fireEvent.click(double);
    fireEvent.click(halve);
    await waitFor(() => expect(tempoSent()).toEqual([{ bpm: 64 }]));

    liveSends({ tempo: 120 });
    await within(header()).findByText("120.0");
    expect(double).toBeEnabled();
    fireEvent.click(double);
    await waitFor(() => expect(tempoSent()).toEqual([{ bpm: 64 }, { bpm: 240 }]));

    liveSends({ tempo: 80 });
    await within(header()).findByText("80.0");
    expect(halve).toBeEnabled();

    liveSends({ tempo: 79 });
    await within(header()).findByText("79.0");
    expect(halve).toBeDisabled();
    expect(double).toBeEnabled();
    fireEvent.click(halve);
    expect(tempoSent()).toHaveLength(2);

    const row = double.parentElement;
    expect(halve.parentElement).toBe(row);
    expect(row).toHaveClass("flex", "gap-1");
    expect(row).not.toHaveClass("flex-col");
    expect(double).toHaveClass("min-h-6");
    expect(halve).toHaveClass("min-h-6");
  });
});

describe("the transport button", () => {
  it("is named for what a click does, and lit while Live runs", async () => {
    renderApp();
    liveSends({ is_playing: false });
    const play = await within(header()).findByRole("button", { name: "Play" });
    expect(play).toHaveAttribute("title", "Play transport");
    expect(play).not.toHaveClass("bg-emerald-500");

    liveSends({ is_playing: true });
    await waitFor(() =>
      expect(within(header()).getByRole("button", { name: "Play" })).toHaveAttribute(
        "title",
        "Transport is running",
      ),
    );
    expect(within(header()).getByRole("button", { name: "Play" })).toHaveClass("bg-emerald-500");
    expect(within(header()).queryByRole("button", { name: "Pause" })).not.toBeInTheDocument();

    fireEvent.click(within(header()).getByRole("button", { name: "Play" }));
    await waitFor(() => expect(backend.sent("POST", `${API}/ableton/play`)).toHaveLength(1));
    expect(backend.sent("POST", `${API}/ableton/stop`)).toHaveLength(0);
  });
});

describe("the status marks", () => {
  it("say Setup beside the dot, in amber for something optional and red for something required", async () => {
    setup = { ok: true, all_green: false, checks: [SPOTIFY_KEYS] };
    renderApp();

    const optional = await within(header()).findByRole("button", {
      name: "Setup: something optional is missing",
    });
    expect(within(optional).getByText("Setup")).toHaveClass("text-amber-300");

    setup = { ok: false, all_green: false, checks: [FFMPEG, SPOTIFY_KEYS] };
    fireEvent.click(optional);
    fireEvent.click(screen.getByRole("button", { name: "↻ refresh" }));
    const required = await within(header()).findByRole("button", {
      name: "Setup: something required is missing",
    });
    expect(within(required).getByText("Setup")).toHaveClass("text-rose-300");

    setup = ALL_READY;
    fireEvent.click(screen.getByRole("button", { name: "↻ refresh" }));
    const ready = await within(header()).findByRole("button", { name: "Setup is complete" });
    expect(ready).toHaveTextContent("");
  });

  it("say No session beside the ring when no session is running", async () => {
    renderApp();

    expect(await within(header()).findByText("No session")).toBeInTheDocument();
    expect(within(header()).getByText("○")).toBeInTheDocument();
  });

  it("say nothing about a session that is running beyond its plays", async () => {
    session = {
      id: 3,
      name: "Friday night",
      notes: null,
      started_at: new Date().toISOString(),
      ended_at: null,
      plays: [],
    };
    renderApp();

    expect(await within(header()).findByText("0 plays")).toBeInTheDocument();
    expect(within(header()).queryByText("No session")).not.toBeInTheDocument();
  });
});

describe("the tempo of each deck", () => {
  it("reads A and B for the songs playing, or else the songs on the page in front", async () => {
    inLive = [
      ...at(120, songOn(0, "a", NIGHT_DRIVE)),
      ...at(127, songOn(0, "b", CLUB_MIX)),
      ...at(128, songOn(1, "a", HALF_LIGHT)),
      ...at(126, songOn(1, "b", LOW_TIDE)),
      ...at(124, songOn(2, "a", SLOW_BURN)),
      ...at(124.3, songOn(2, "b", SLOW_BURN_DUB)),
    ];
    renderApp();
    liveSends({});

    expect(await within(header()).findByText("A 120 · B 127")).toHaveClass("text-rose-200");
    expect(within(header()).queryByText(/Δ/)).not.toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: "Page 2" }));
    expect(await within(header()).findByText("A 128 · B 126")).toHaveClass("text-amber-200");

    liveSends({ is_playing: true, playing_clips: deckAPlaysRow(2) });
    expect(await within(header()).findByText("A 120 · B 126")).toHaveClass("text-rose-200");

    liveSends({ is_playing: false, playing_clips: {} });
    fireEvent.click(screen.getByRole("button", { name: "Page 3" }));
    await waitFor(() =>
      expect(within(header()).queryByText(/^A \d/)).not.toBeInTheDocument(),
    );
  });
});

describe("the record button", () => {
  it("says so in a row under the header when Live may be recording nothing", async () => {
    recordAnswer = {
      ok: true,
      recording: true,
      armed_track: null,
      warnings: [UNCONFIRMED_RECORDING],
    };
    renderApp();

    fireEvent.click(within(header()).getByRole("button", { name: "Start recording" }));
    const stop = await within(header()).findByRole("button", { name: /^Stop recording/ });
    expect(stop).toHaveTextContent("00:00");
    await screen.findByText(UNCONFIRMED_RECORDING);
    noticeUnderTheHeader(UNCONFIRMED_RECORDING);

    fireEvent.click(stop);
    await within(header()).findByRole("button", { name: "Start recording" });
    await waitFor(() => expect(screen.queryByText(UNCONFIRMED_RECORDING)).not.toBeInTheDocument());

    recordAnswer = { ok: false, recording: false, warning: RECORD_NOT_SENT };
    fireEvent.click(within(header()).getByRole("button", { name: "Start recording" }));
    await screen.findByText(RECORD_DID_NOT_CHANGE);
    expect(noticeUnderTheHeader(RECORD_DID_NOT_CHANGE)).toHaveAttribute("title", RECORD_NOT_SENT);
    expect(screen.queryByText(RECORD_NOT_SENT)).not.toBeInTheDocument();
    expect(within(header()).getByRole("button", { name: "Start recording" })).toBeInTheDocument();

    recordAnswer = { ok: true, recording: true, armed_track: 10, warnings: [] };
    fireEvent.click(within(header()).getByRole("button", { name: "Start recording" }));
    await within(header()).findByRole("button", { name: /^Stop recording/ });
    await waitFor(() => expect(screen.queryByText(RECORD_DID_NOT_CHANGE)).not.toBeInTheDocument());
    expect(screen.queryByText(UNCONFIRMED_RECORDING)).not.toBeInTheDocument();
  });
});
