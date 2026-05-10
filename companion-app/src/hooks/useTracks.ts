import { useQuery } from "@tanstack/react-query";
import * as api from "../api";
import type { TrackFilters } from "../types";

export function useTracks(filters: TrackFilters = {}) {
  return useQuery({
    queryKey: ["tracks", filters],
    queryFn: () => api.getTracks(filters),
    staleTime: 30_000,
  });
}

export function useTrack(id: number | null | undefined) {
  return useQuery({
    queryKey: ["track", id],
    queryFn: () => api.getTrack(id as number),
    enabled: id != null,
    staleTime: 60_000,
  });
}

export function useStems(trackId: number | null | undefined) {
  return useQuery({
    queryKey: ["stems", trackId],
    queryFn: () => api.getStems(trackId as number),
    enabled: trackId != null,
    staleTime: 60_000,
  });
}
