import { useQuery } from "@tanstack/react-query";
import * as api from "../api";

export function useRegions(trackId: number | null | undefined) {
  return useQuery({
    queryKey: ["regions", trackId],
    queryFn: () => api.getRegions(trackId as number),
    enabled: trackId != null,
    staleTime: 60_000,
  });
}
