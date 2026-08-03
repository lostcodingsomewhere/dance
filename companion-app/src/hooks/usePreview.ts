import { useMutation } from "@tanstack/react-query";
import * as api from "../api";
import { store, useAppStore } from "../store";

/**
 * Audition a candidate clip through Ableton's Cue track (Scarlett outs 3/4 →
 * headphones). The hook keeps local UI state in the app store so multiple
 * banner cards can highlight the previewing one. Single-preview at a time:
 * starting a new preview implicitly replaces the previous on the backend,
 * and the local state mirrors that.
 */
export function useStartPreview() {
  return useMutation({
    mutationFn: (vars: { trackId: number; column: string }) =>
      api.abletonPreview({ track_id: vars.trackId, column: vars.column }),
    onMutate: (vars) => {
      store.setPreviewing({ trackId: vars.trackId, column: vars.column });
    },
    onSuccess: (result, vars) => {
      // A REFUSED preview still returns HTTP 200 — the backend declines to
      // fire (no confirmed Cue track, Live unreachable, missing file) and
      // says so via ok:false + warnings, precisely so it can't blast an
      // audition through the master. Treat that as a failure here: leaving
      // the optimistic state up would show a lit card and a running
      // playhead over total silence, and the obvious conclusion for anyone
      // holding headphones is that their audio interface is broken.
      if (result.ok === false) {
        store.setPreviewing(null);
        store.setLoadWarnings(
          `Preview refused · ${vars.column}`,
          result.warnings?.length
            ? result.warnings
            : ["Ableton refused the preview (no reason given)."],
        );
        return;
      }
      // Backfill cue track/slot so CueStrip's waveform can seek into the
      // running preview clip via /ableton/transport/seek.
      store.setPreviewing({
        trackId: vars.trackId,
        column: vars.column,
        cueTrackIdx: result.cue_track_idx ?? null,
        slot: result.slot ?? null,
      });
      // ok:true is not the same as "you can hear it". The bridge fires
      // best-effort and reports what it could not confirm — clip existence,
      // or that Live ever said is_playing. Those warnings ARE the difference
      // between a working audition and silence, so show them; the preview
      // state stays up because the clip may well be playing.
      if (result.warnings?.length) {
        store.setLoadWarnings(`Preview · ${vars.column}`, result.warnings);
      }
    },
    onError: () => {
      // If the request fails, drop the optimistic preview state so the
      // button doesn't get stuck.
      store.setPreviewing(null);
    },
  });
}

export function useStopPreview() {
  return useMutation({
    mutationFn: () => api.abletonPreviewStop(),
    onSettled: () => {
      store.setPreviewing(null);
    },
  });
}

/** Read the currently-previewing candidate (null when nothing is auditioning). */
export function usePreviewState(): {
  trackId: number;
  column: string;
  cueTrackIdx?: number | null;
  slot?: number | null;
} | null {
  return useAppStore((s) => s.previewing);
}
