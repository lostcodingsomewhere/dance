import { useQuery } from "@tanstack/react-query";
import * as api from "../api";

/** Returns recommendations from the given seeds. Disabled when no seeds. */
export function useRecommend(seeds: number[], k = 12) {
  return useQuery({
    queryKey: ["recommend", seeds.slice().sort(), k],
    queryFn: () => api.recommend({ seeds, k }),
    enabled: seeds.length > 0,
    staleTime: 15_000,
  });
}
