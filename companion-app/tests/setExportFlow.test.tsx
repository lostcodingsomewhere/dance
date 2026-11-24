import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SetEditor } from "../src/views/SetEditor";

const SET_ID = 1;
const EXPORT_PATH = `/api/v1/sets/${SET_ID}/als`;
const OPEN_PATH = "/api/v1/files/open-in-live";
const REVEAL_PATH = "/api/v1/files/reveal";
const OUT_PATH = "/Users/x/Music/Dance/Sets/Friday night.als";
const LIVE_APP = "Ableton Live 12 Standard";
const EMPTY_PLAN_MESSAGE = "the plan is empty, queue a pick first";

const EXPORTED = {
  ok: true,
  out_path: OUT_PATH,
  size_bytes: 20412,
  page_count: 3,
  scene_count: 15,
  clip_count: 80,
  skipped: [
    { track_id: 12, role: "other", reason: "track is 'separated', need 'complete'" },
    { track_id: 15, role: "song", reason: "no stems on disk" },
  ],
};

interface SeenRequest {
  method: string;
  path: string;
  body: unknown;
}

interface Reply {
  status: number;
  body: unknown;
}

let requests: SeenRequest[];
let exportReply: Reply;
let exportHeld: Promise<void>;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function planItem(track_id: number, title: string | null) {
  return { track_id, title, artist: "Test Artist", key_camelot: "8A", bpm: 128, floor_energy: 6 };
}

function holdExportAnswer(): () => void {
  let release = () => {};
  exportHeld = new Promise((resolve) => {
    release = resolve;
  });
  return release;
}

async function fakeBackend(url: string, init: RequestInit = {}): Promise<Response> {
  const method = init.method ?? "GET";
  const path = url.split("?")[0];
  const body = typeof init.body === "string" ? JSON.parse(init.body) : undefined;
  requests.push({ method, path, body });

  if (path === "/api/v1/sets/active") {
    return json({
      id: SET_ID,
      name: "Friday night",
      notes: null,
      is_active: true,
      created_at: "2026-11-01T00:00:00",
      updated_at: "2026-11-01T00:00:00",
      tracks: [],
    });
  }
  if (path === `/api/v1/sets/${SET_ID}/plan`) {
    return json({
      set_id: SET_ID,
      queues: {
        drums: [],
        bass: [],
        vocals: [],
        other: [planItem(12, "Half Light")],
        song: [planItem(7, "Night Drive"), planItem(15, null)],
      },
    });
  }
  if (path === `/api/v1/sets/${SET_ID}/plan-recs`) {
    return json({ column: "mix", combo_size: 0, recs: [] });
  }
  if (path === EXPORT_PATH && method === "POST") {
    await exportHeld;
    return json(exportReply.body, exportReply.status);
  }
  if (path === OPEN_PATH && method === "POST") return json({ ok: true, app: LIVE_APP });
  if (path === REVEAL_PATH && method === "POST") return json({ ok: true });
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

function requestsTo(path: string): SeenRequest[] {
  return requests.filter((request) => request.path === path);
}

function exportResult(): Promise<HTMLElement> {
  return screen.findByRole("region", { name: "Export result" });
}

const originalFetch = globalThis.fetch;

beforeEach(() => {
  requests = [];
  exportReply = { status: 200, body: EXPORTED };
  exportHeld = Promise.resolve();
  globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
    fakeBackend(typeof input === "string" ? input : input.toString(), init),
  ) as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("exporting a set to Live from the Set view", () => {
  it("writes the plan as one Live Set, opens it in Live, and shows a refused export", async () => {
    renderSetView();
    await screen.findByText("Night Drive");

    const answerExport = holdExportAnswer();
    fireEvent.click(screen.getByRole("button", { name: "Export to Live" }));
    expect(await screen.findByRole("button", { name: "Exporting…" })).toBeDisabled();
    answerExport();

    const result = await exportResult();
    expect(requestsTo(EXPORT_PATH)).toEqual([{ method: "POST", path: EXPORT_PATH, body: {} }]);
    expect(screen.getByRole("button", { name: "Export to Live" })).toBeEnabled();
    expect(within(result).getByText("Friday night.als")).toHaveAttribute("title", OUT_PATH);
    expect(within(result).getByText("3 pages · 80 clips")).toBeInTheDocument();

    const skippedWithTitle = within(result).getByText(
      "Half Light (melody): track is 'separated', need 'complete'",
    );
    expect(skippedWithTitle).not.toBeVisible();
    fireEvent.click(within(result).getByText("2 picks left out"));
    expect(skippedWithTitle).toBeVisible();
    expect(within(result).getByText("track 15 (song): no stems on disk")).toBeVisible();

    fireEvent.click(within(result).getByRole("button", { name: "Open in Live" }));
    expect(await within(result).findByText(`Opened in ${LIVE_APP}`)).toBeInTheDocument();
    expect(requestsTo(OPEN_PATH)).toEqual([
      { method: "POST", path: OPEN_PATH, body: { path: OUT_PATH } },
    ]);

    fireEvent.click(within(result).getByRole("button", { name: "Show in Finder" }));
    await waitFor(() =>
      expect(requestsTo(REVEAL_PATH)).toEqual([
        { method: "POST", path: REVEAL_PATH, body: { path: OUT_PATH } },
      ]),
    );

    exportReply = { status: 400, body: { detail: EMPTY_PLAN_MESSAGE } };
    fireEvent.click(screen.getByRole("button", { name: "Export to Live" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(EMPTY_PLAN_MESSAGE);
    expect(screen.queryByText("Friday night.als")).not.toBeInTheDocument();
    expect(screen.queryByText(`Opened in ${LIVE_APP}`)).not.toBeInTheDocument();
    expect(requestsTo(EXPORT_PATH)).toHaveLength(2);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss export result" }));
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Export result" })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: "Export to Live" })).toHaveFocus();
  });
});
