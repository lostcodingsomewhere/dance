import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SetEditor } from "../src/views/SetEditor";

const TRACK_ID = 7;
const DURATION_SECONDS = 200;
const WAVEFORM_WIDTH_PX = 1000;
const ROLES = ["drums", "bass", "vocals", "other", "mix"];
const APP_PICKS = [
  { slot: 1, position_ms: 0, section_label: "intro" },
  { slot: 2, position_ms: 30_000, section_label: "buildup" },
  { slot: 3, position_ms: 60_000, section_label: "drop" },
  { slot: 4, position_ms: 100_000, section_label: "breakdown" },
  { slot: 5, position_ms: 140_000, section_label: "drop" },
];

interface FakePad {
  slot: number;
  role: string;
  stem_file_id: number | null;
  position_ms: number;
  length_ms: number;
  length_bars: number;
  section_label: string;
  source: string;
  silent: boolean;
}

interface SeenRequest {
  method: string;
  path: string;
  body: unknown;
}

function appPick(role: string, slot: number): FakePad {
  return {
    ...APP_PICKS[slot - 1],
    role,
    stem_file_id: role === "mix" ? null : 10 + ROLES.indexOf(role),
    length_ms: 15_000,
    length_bars: 8,
    source: "auto",
    silent: false,
  };
}

let pads: FakePad[];
let requests: SeenRequest[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function padsAfter(role: string, slot: number, change: (pad: FakePad) => FakePad): FakePad[] {
  return pads.map((pad) =>
    pad.slot === slot && (role === "all" || pad.role === role) ? change(pad) : pad,
  );
}

function fakeBackend(url: string, init: RequestInit = {}): Response {
  const method = init.method ?? "GET";
  const path = url.split("?")[0];
  const body = typeof init.body === "string" ? JSON.parse(init.body) : undefined;
  requests.push({ method, path, body });

  if (path === "/api/v1/sets/active") {
    return json({
      id: 1,
      name: "Friday",
      notes: null,
      is_active: true,
      created_at: "2026-11-01T00:00:00",
      updated_at: "2026-11-01T00:00:00",
      tracks: [],
    });
  }
  if (path === "/api/v1/sets/1/plan") {
    return json({
      set_id: 1,
      queues: {
        drums: [],
        bass: [],
        vocals: [],
        other: [],
        song: [
          {
            track_id: TRACK_ID,
            title: "Night Drive",
            artist: "Test Artist",
            key_camelot: "8A",
            bpm: 128,
            floor_energy: 6,
          },
        ],
      },
    });
  }
  if (path === "/api/v1/sets/1/plan-recs") {
    return json({ column: "mix", combo_size: 0, recs: [] });
  }
  if (path === `/api/v1/tracks/${TRACK_ID}/waveform`) {
    return json({ peaks: [0.2, 0.8, 0.5, 0.3], duration_seconds: DURATION_SECONDS, num_peaks: 4 });
  }
  if (path === `/api/v1/tracks/${TRACK_ID}/pads`) return json(pads);

  const onePad = path.match(new RegExp(`^/api/v1/tracks/${TRACK_ID}/pads/(\\w+)/(\\d)$`));
  if (onePad && method === "PUT") {
    const [, role, slot] = onePad;
    const position_ms = (body as { position_ms: number }).position_ms;
    if (position_ms >= DURATION_SECONDS * 1000) {
      return json({ detail: "position is outside the song" }, 422);
    }
    pads = padsAfter(role, Number(slot), (pad) => ({ ...pad, position_ms, source: "manual" }));
    return json(pads.find((pad) => pad.slot === Number(slot)));
  }
  if (onePad && method === "DELETE") {
    const [, role, slot] = onePad;
    pads = padsAfter(role, Number(slot), (pad) => appPick(pad.role, pad.slot));
    return json(pads);
  }
  return json({ detail: `no fake for ${method} ${path}` }, 404);
}

function renderSetView() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <SetEditor />
    </QueryClientProvider>,
  );
}

function pointerEvent(type: string, clientX: number): MouseEvent {
  return Object.assign(new MouseEvent(type, { bubbles: true, clientX }), { pointerId: 1 });
}

function clickWaveformAt(waveform: Element, clientX: number) {
  fireEvent(waveform, pointerEvent("pointerdown", clientX));
  fireEvent(waveform, pointerEvent("pointerup", clientX));
}

function pressEnterOn(button: Element) {
  fireEvent.keyDown(button, { key: "Enter" });
  fireEvent.click(button);
}

function requestsWith(method: string): SeenRequest[] {
  return requests.filter((request) => request.method === method);
}

const originalFetch = globalThis.fetch;

beforeEach(() => {
  pads = ROLES.flatMap((role) => APP_PICKS.map((pick) => appPick(role, pick.slot)));
  requests = [];
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) =>
    fakeBackend(typeof input === "string" ? input : input.toString(), init),
  ) as unknown as typeof fetch;
  // jsdom has no layout and no pointer capture, both of which the waveform's click-to-seek needs.
  Object.assign(Element.prototype, {
    setPointerCapture: () => {},
    releasePointerCapture: () => {},
    hasPointerCapture: () => false,
  });
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: WAVEFORM_WIDTH_PX,
    bottom: 72,
    width: WAVEFORM_WIDTH_PX,
    height: 72,
    toJSON: () => ({}),
  });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("cue pads in the Set view", () => {
  it("moves a queued song's pad by hand, resets it, and shows a rejected move", async () => {
    renderSetView();

    fireEvent.mouseDown(await screen.findByText("Night Drive"));
    pressEnterOn(screen.getByRole("button", { name: "Pads" }));
    const editor = await screen.findByRole("dialog", { name: /Night Drive/ });
    expect(await within(editor).findAllByTestId("pad-flag")).toHaveLength(5);

    pressEnterOn(within(editor).getByRole("button", { name: /^Pad 3, drop, 1:00/ }));
    const reset = within(editor).getByRole("button", { name: /^Reset/ });
    expect(reset).toBeDisabled();

    const waveform = within(editor).getAllByTestId("waveform-bar")[0].closest("div") as Element;
    clickWaveformAt(waveform, 400);

    await waitFor(() =>
      expect(requestsWith("PUT")).toEqual([
        {
          method: "PUT",
          path: `/api/v1/tracks/${TRACK_ID}/pads/all/3`,
          body: { position_ms: 80_000 },
        },
      ]),
    );
    await waitFor(() => expect(reset).toBeEnabled());
    expect(within(editor).getByRole("button", { name: /^Pad 3, drop, 1:20/ })).toBeInTheDocument();

    fireEvent.click(reset);

    await waitFor(() =>
      expect(requestsWith("DELETE")).toEqual([
        { method: "DELETE", path: `/api/v1/tracks/${TRACK_ID}/pads/all/3`, body: undefined },
      ]),
    );
    await waitFor(() => expect(reset).toBeDisabled());
    expect(within(editor).getByRole("button", { name: /^Pad 3, drop, 1:00/ })).toBeInTheDocument();

    clickWaveformAt(waveform, WAVEFORM_WIDTH_PX);

    expect(await within(editor).findByRole("alert")).toHaveTextContent(
      "position is outside the song",
    );
    expect(within(editor).getByRole("button", { name: /^Pad 3, drop, 1:00/ })).toBeInTheDocument();
    expect(requestsWith("POST")).toEqual([]);
  });
});
