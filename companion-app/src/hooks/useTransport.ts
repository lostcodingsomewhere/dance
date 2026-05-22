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
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useFireCell() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, slot }: { track: number; slot: number }) =>
      api.abletonFireCell(track, slot),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopTrack() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (trackIndex: number) => api.abletonStopTrack(trackIndex),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopAllClips() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.abletonStopAllClips(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopCell() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, slot }: { track: number; slot: number }) =>
      api.abletonStopCell(track, slot),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useStopScene() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sceneIndex: number) => api.abletonStopScene(sceneIndex),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}

export function useSoloTrack() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ track, soloed }: { track: number; soloed: boolean }) =>
      api.abletonSoloTrack(track, soloed),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
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
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ableton", "state"] }),
  });
}
