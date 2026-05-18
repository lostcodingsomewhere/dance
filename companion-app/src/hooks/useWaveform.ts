import { useQuery } from "@tanstack/react-query";
import * as api from "../api";

/**
 * Fetch a track's full-mix waveform peaks. Cached aggressively because the
 * backend writes a sidecar JSON on disk — fetching is cheap after the first
 * decode, and the peaks don't change for a given file.
 */
export function useTrackWaveform(
  trackId: number | null | undefined,
  numPeaks = 200,
) {
  return useQuery({
    queryKey: ["waveform", "track", trackId, numPeaks],
    queryFn: () => api.getTrackWaveform(trackId as number, numPeaks),
    enabled: trackId != null,
    staleTime: Infinity,
    gcTime: 30 * 60_000,
  });
}

/**
 * Fetch a stem's waveform peaks. Used by ComboStrip's per-role cards (one
 * waveform per playing stem) so cells reveal what's actually playing.
 */
export function useStemWaveform(
  stemFileId: number | null | undefined,
  numPeaks = 200,
) {
  return useQuery({
    queryKey: ["waveform", "stem", stemFileId, numPeaks],
    queryFn: () => api.getStemWaveform(stemFileId as number, numPeaks),
    enabled: stemFileId != null,
    staleTime: Infinity,
    gcTime: 30 * 60_000,
  });
}
