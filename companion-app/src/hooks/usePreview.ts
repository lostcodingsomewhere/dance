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
export function usePreviewState(): { trackId: number; column: string } | null {
  return useAppStore((s) => s.previewing);
}
