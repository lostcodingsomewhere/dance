import { useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";

/**
 * Transport mutations — direct OSC passthrough to Ableton via the bridge.
 * Fire-and-forget: AbletonState updates arrive on the WebSocket, not as a
 * mutation response. We still invalidate ["ableton", "state"] so any
 * derived query (e.g. dashboards) refetches eagerly if it needs to.
 */
export function useFireScene() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sceneIndex: number) => api.abletonFireScene(sceneIndex),
    meta: { attempt: (sceneIndex: number) => `play scene ${sceneIndex + 1}` },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useFireCell() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, slot }: { track: number; slot: number }) =>
      api.abletonFireCell(track, slot),
    meta: { attempt: "play that clip" },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopTrack() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (trackIndex: number) => api.abletonStopTrack(trackIndex),
    meta: { attempt: "stop that column" },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopAllClips() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.abletonStopAllClips(),
    meta: { attempt: "stop all clips" },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopCell() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, slot }: { track: number; slot: number }) =>
      api.abletonStopCell(track, slot),
    meta: { attempt: "stop that clip" },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopScene() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sceneIndex: number) => api.abletonStopScene(sceneIndex),
    meta: { attempt: (sceneIndex: number) => `stop scene ${sceneIndex + 1}` },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useSoloTrack() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, soloed }: { track: number; soloed: boolean }) =>
      api.abletonSoloTrack(track, soloed),
    meta: {
      attempt: ({ soloed }: { soloed: boolean }) =>
        soloed ? "solo that column" : "unsolo that column",
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

/**
 * Delete a single cell (one stem in one scene). Invalidates ``ableton``
 * + ``decks`` so the SceneGrid re-renders without the removed cell.
 */
export function useDeleteCell() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, slot }: { track: number; slot: number }) =>
      api.abletonDeleteCell(track, slot),
    meta: { attempt: "remove that clip" },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ableton", "state"] });
      qc.invalidateQueries({ queryKey: ["decks"] });
    },
  });
}

export function useSeekClip() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      track,
      slot,
      positionBeats,
    }: {
      track: number;
      slot: number;
      positionBeats: number;
    }) => api.abletonSeekClip(track, slot, positionBeats),
    meta: { attempt: "jump to that spot" },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}
