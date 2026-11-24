import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import type { PadTarget } from "../types";

const padsKey = (trackId: number | null | undefined) => ["pads", trackId] as const;

export function usePads(trackId: number | null | undefined) {
  return useQuery({
    queryKey: padsKey(trackId),
    queryFn: () => api.getPads(trackId as number),
    enabled: trackId != null,
    staleTime: 60_000,
  });
}

export function useMovePad(trackId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (move: { target: PadTarget; slot: number; positionMs: number }) =>
      api.movePad(trackId, move.target, move.slot, move.positionMs),
    onSuccess: () => qc.invalidateQueries({ queryKey: padsKey(trackId) }),
  });
}

export function useResetPad(trackId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (pad: { target: PadTarget; slot: number }) =>
      api.resetPad(trackId, pad.target, pad.slot),
    onSuccess: () => qc.invalidateQueries({ queryKey: padsKey(trackId) }),
  });
}
