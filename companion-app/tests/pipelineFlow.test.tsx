import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { PipelineRecentTrack } from "../src/types";
import { PipelineOps } from "../src/views/PipelineOps";
import { BoothBackend, type Answer, type SeenRequest } from "./boothHarness";

const API = "/api/v1/pipeline";
const OUT_OF_MEMORY_FIRST_SENTENCE =
  "separate: MPS backend out of memory (MPS allocated: 9.05 GB, other allocations: 8.99 GB, max allowed: 18.13 GB).";
const OUT_OF_MEMORY =
  `${OUT_OF_MEMORY_FIRST_SENTENCE} Tried to allocate 262.14 MB on private pool. ` +
  "Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).";
const FILE_IS_GONE = "Track file not found: /music/half_light.mp3";
const RUN_IN_PROGRESS =
  "Another pipeline run is already in progress (pid 4242, `dance process`, started 2026-11-24T10:00:00+00:00). " +
  "Only one dispatcher may run at a time, so wait for it to finish (lock file: /music/data/dispatcher.lock).";
const NO_FFMPEG = "ffmpeg not found. Install it, then try again.";
const NOT_YOUR_PLAYLIST =
  "Spotify refused this playlist. It lists a playlist's songs only for the account that owns it.";
const ADDRESS_NOT_ALLOWED = "This address may not use the Dance server.";
const NOTHING_TO_DOWNLOAD = "No rows to download. Every row was a duplicate.";
const CSV = "Track Name,Artist Name(s)\nShallow Water,Elderbrook\n";
const CSV_FIELD = /Paste Exportify CSV here/;
const NOTHING_NEW_ON_DISK = { new: 0, updated: 0, unchanged: 1, errors: 0 };

function track(
  fields: Partial<PipelineRecentTrack> & Pick<PipelineRecentTrack, "id" | "title">,
): PipelineRecentTrack {
  return {
    artist: "Test Artist",
    state: "complete",
    updated_at: "2026-11-20T10:00:00Z",
    error_message: null,
    ...fields,
  };
}

const FAILED = track({ id: 7, title: "Night Drive", state: "error", error_message: OUT_OF_MEMORY });
const FILE_MISSING = track({ id: 8, title: "Half Light", state: "error", error_message: FILE_IS_GONE });
const FINISHED = track({ id: 9, title: "Slow Burn" });

function statusOf(tracks: PipelineRecentTrack[]) {
  const counts: Record<string, number> = {};
  for (const each of tracks) counts[each.state] = (counts[each.state] ?? 0) + 1;
  return {
    counts,
    total: tracks.length,
    in_progress: false,
    errors: counts.error ?? 0,
    complete: counts.complete ?? 0,
    weighted_progress: 50,
  };
}

function aLibraryOf(tracks: PipelineRecentTrack[], routes: Record<string, Answer> = {}): Answer {
  return (request: SeenRequest): unknown => {
    const route = routes[`${request.method} ${request.path}`];
    if (route) return route(request);
    if (request.path === `${API}/status`) return statusOf(tracks);
    if (request.path === `${API}/watch`) return { enabled: false, interval_seconds: 60 };
    if (request.path === `${API}/recent`) {
      const state = new URLSearchParams(request.query).get("state");
      return state ? tracks.filter((each) => each.state === state) : tracks;
    }
    return undefined;
  };
}

function refusal(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function renderPipelineView() {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <PipelineOps />
    </QueryClientProvider>,
  );
}

function rowOf(title: string): Promise<HTMLElement> {
  return screen.findByRole("row", { name: new RegExp(title) });
}

function csvDialog(): HTMLElement {
  return screen.getByRole("dialog", { name: "Ingest from CSV" });
}

function clickOutside(dialog: HTMLElement): void {
  const backdrop = dialog.parentElement as HTMLElement;
  fireEvent.mouseDown(backdrop);
  fireEvent.mouseUp(backdrop);
  fireEvent.click(backdrop);
}

function holdAnswersTo(path: string): () => Promise<void> {
  const answering = globalThis.fetch;
  const delivered: Promise<unknown>[] = [];
  let letThrough = () => {};
  const held = new Promise<void>((resolve) => {
    letThrough = resolve;
  });
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const answer = answering(input, init);
    if (!String(input).startsWith(path)) return answer;
    const late = held.then(() => answer);
    delivered.push(late.catch(() => undefined));
    return late;
  }) as typeof fetch;
  return async () => {
    letThrough();
    await Promise.all(delivered);
    await new Promise((resolve) => setTimeout(resolve, 20));
  };
}

let backend: BoothBackend;

afterEach(() => {
  backend.restore();
});

describe("failed tracks in the Pipeline view", () => {
  it("sends a failed track back for another try and then shows it waiting", async () => {
    const tracks = [FAILED, FINISHED];
    backend = new BoothBackend(
      aLibraryOf(tracks, {
        [`POST ${API}/tracks/7/retry`]: () => {
          tracks[0] = { ...FAILED, state: "pending", error_message: null };
          return tracks[0];
        },
      }),
    );
    renderPipelineView();

    const failed = await rowOf("Night Drive");
    expect(
      within(await rowOf("Slow Burn")).queryByRole("button", { name: "Retry" }),
    ).not.toBeInTheDocument();
    fireEvent.click(within(failed).getByRole("button", { name: "Retry" }));

    await waitFor(() =>
      expect(within(failed).getByRole("button", { name: "pending" })).toBeInTheDocument(),
    );
    expect(backend.sent("POST", `${API}/tracks/7/retry`)).toHaveLength(1);
    expect(within(failed).queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    expect(within(failed).queryByText(/out of memory/)).not.toBeInTheDocument();
  });

  it("keeps a retried track waiting when a list refresh was already on its way", async () => {
    const tracks = [FAILED];
    backend = new BoothBackend(
      aLibraryOf(tracks, {
        [`POST ${API}/scan`]: () => NOTHING_NEW_ON_DISK,
        [`POST ${API}/tracks/7/retry`]: () => {
          tracks[0] = { ...FAILED, state: "pending", error_message: null };
          return tracks[0];
        },
      }),
    );
    renderPipelineView();
    const failed = await rowOf("Night Drive");

    const deliverTheList = holdAnswersTo(`${API}/recent`);
    fireEvent.click(screen.getByRole("button", { name: "Scan" }));
    await waitFor(() => expect(backend.sent("GET", `${API}/recent`)).toHaveLength(2));
    fireEvent.click(within(failed).getByRole("button", { name: "Retry" }));
    await waitFor(() =>
      expect(within(failed).getByRole("button", { name: "pending" })).toBeInTheDocument(),
    );
    await act(deliverTheList);

    expect(within(failed).getByRole("button", { name: "pending" })).toBeInTheDocument();
    expect(within(failed).queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("says why a retry did not happen and leaves the track as it was", async () => {
    backend = new BoothBackend(
      aLibraryOf([FAILED], {
        [`POST ${API}/tracks/7/retry`]: () => refusal(409, RUN_IN_PROGRESS),
      }),
    );
    renderPipelineView();

    const failed = await rowOf("Night Drive");
    fireEvent.click(within(failed).getByRole("button", { name: "Retry" }));

    const notice = await screen.findByRole("alert");
    expect(notice).toHaveTextContent(RUN_IN_PROGRESS);
    expect(notice).not.toHaveTextContent("API 409");
    expect(within(failed).getByRole("button", { name: "error" })).toBeInTheDocument();
    expect(within(failed).getByRole("button", { name: "Retry" })).toBeEnabled();
  });
});
