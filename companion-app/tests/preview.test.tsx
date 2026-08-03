import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * A refused preview must not look like a working one.
 *
 * `/ableton/preview` answers **HTTP 200 with `ok: false`** when the bridge
 * declines to fire — no confirmed Cue track, Live unreachable, file missing.
 * That is deliberate: firing into a guessed track index is how an audition
 * ends up blasting out of the master instead of the headphones, so the bridge
 * refuses rather than guesses.
 *
 * The hook used to read only the payload's cue_track_idx/slot, so a refusal
 * left the card lit and a playhead running over complete silence. Holding
 * headphones, the only available conclusion is "my audio interface is
 * broken" — the failure is invisible exactly when a first-time user is least
 * equipped to diagnose it.
 *
 * `ok: true` is a weaker claim than it looks, too: the bridge fires
 * best-effort and reports what it could NOT confirm (clip existence, whether
 * Live ever said is_playing). Those warnings are the difference between an
 * audition and silence, so they surface as well — but the preview state
 * stays up, because the clip may genuinely be playing.
 */

const abletonPreview = vi.fn();
vi.mock("../src/api", () => ({
  abletonPreview: (...args: unknown[]) => abletonPreview(...args),
  abletonPreviewStop: () => Promise.resolve({ ok: true }),
}));

const { useStartPreview } = await import("../src/hooks/usePreview");
const { store, useAppState } = await import("../src/store");

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({
    defaultOptions: { mutations: { retry: false }, queries: { retry: false } },
  });
  return React.createElement(QueryClientProvider, { client: qc }, children);
}

/** Drive the real hook and observe the real store through its subscription. */
function renderPreview() {
  return renderHook(() => ({ start: useStartPreview(), app: useAppState() }), {
    wrapper,
  });
}

beforeEach(() => {
  abletonPreview.mockReset();
  store.setPreviewing(null);
  store.clearLoadWarnings();
});

describe("useStartPreview", () => {
  it("clears the optimistic state and surfaces why when the bridge refuses", async () => {
    abletonPreview.mockResolvedValue({
      ok: false,
      cue_track_idx: null,
      slot: 0,
      warnings: ["Could not establish a dedicated 'Cue' track in Live"],
    });

    const { result } = renderPreview();
    await act(async () => {
      result.current.start.mutate({ trackId: 7, column: "drums" });
    });
    await waitFor(() => expect(result.current.start.isPending).toBe(false));
    // No lit card, no playhead — nothing claims to be auditioning.
    expect(result.current.app.previewing).toBeNull();
    expect(result.current.app.loadWarnings?.warnings).toEqual([
      "Could not establish a dedicated 'Cue' track in Live",
    ]);
  });

  it("still explains itself when a refusal carries no warnings", async () => {
    abletonPreview.mockResolvedValue({ ok: false });

    const { result } = renderPreview();
    await act(async () => {
      result.current.start.mutate({ trackId: 7, column: "drums" });
    });
    await waitFor(() => expect(result.current.start.isPending).toBe(false));
    expect(result.current.app.previewing).toBeNull();
    expect(result.current.app.loadWarnings?.warnings?.length).toBeGreaterThan(0);
  });

  it("keeps the preview but shows warnings on a degraded ok:true", async () => {
    abletonPreview.mockResolvedValue({
      ok: true,
      cue_track_idx: 10,
      slot: 0,
      warnings: ["Preview clip did not report playing within 8s"],
    });

    const { result } = renderPreview();
    await act(async () => {
      result.current.start.mutate({ trackId: 7, column: "drums" });
    });
    await waitFor(() => expect(result.current.start.isPending).toBe(false));
    // The clip may well be playing, so the cue strip stays up...
    expect(result.current.app.previewing).toMatchObject({ trackId: 7, cueTrackIdx: 10 });
    // ...but the user is told what could not be confirmed.
    expect(result.current.app.loadWarnings?.warnings).toEqual([
      "Preview clip did not report playing within 8s",
    ]);
  });

  it("leaves a clean success completely quiet", async () => {
    abletonPreview.mockResolvedValue({
      ok: true,
      cue_track_idx: 10,
      slot: 0,
      warnings: [],
    });

    const { result } = renderPreview();
    await act(async () => {
      result.current.start.mutate({ trackId: 7, column: "drums" });
    });
    await waitFor(() => expect(result.current.start.isPending).toBe(false));
    expect(result.current.app.previewing).toMatchObject({ trackId: 7, cueTrackIdx: 10 });
    expect(result.current.app.loadWarnings).toBeNull();
  });
});
