/**
 * React-query hooks for Sets (the persistent curated track plans) and their
 * tail-recs. The active set drives the Set Rail; tail-recs are the
 * "what comes next" section at the bottom of the rail.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import type { DanceSet } from "../types";

const KEYS = {
  list: () => ["sets"] as const,
  active: () => ["sets", "active"] as const,
  set: (id: number) => ["sets", id] as const,
  tailRecs: (id: number, opts: { k?: number; excludeSessionPlays?: boolean }) =>
    ["sets", id, "tail-recs", opts.k ?? 10, !!opts.excludeSessionPlays] as const,
};

export function useSets() {
  return useQuery({
    queryKey: KEYS.list(),
    queryFn: api.listSets,
    staleTime: 5_000,
  });
}

export function useActiveSet() {
  return useQuery({
    queryKey: KEYS.active(),
    queryFn: api.getActiveSet,
    // Refresh frequently — the rail wants near-realtime when other tabs
    // add tracks. Cheap query, 5 s polling.
    refetchInterval: 5_000,
    staleTime: 1_000,
  });
}

export function useSet(id: number | null | undefined) {
  return useQuery({
    queryKey: id ? KEYS.set(id) : ["sets", "none"],
    queryFn: () => (id ? api.getSet(id) : Promise.resolve(null as DanceSet | null)),
    enabled: !!id,
  });
}

export function useTailRecs(
  setId: number | null | undefined,
  opts: { k?: number; excludeSessionPlays?: boolean } = {},
) {
  return useQuery({
    queryKey: setId
      ? KEYS.tailRecs(setId, opts)
      : ["sets", "none", "tail-recs"],
    queryFn: () =>
      setId
        ? api.getTailRecs(setId, opts)
        : Promise.resolve({ set_id: 0, set_track_count: 0, recs: [] }),
    enabled: !!setId,
    staleTime: 10_000,
  });
}

// Mutations -----------------------------------------------------------------

function invalidateSet(qc: ReturnType<typeof useQueryClient>, setId: number) {
  qc.invalidateQueries({ queryKey: KEYS.set(setId) });
  qc.invalidateQueries({ queryKey: KEYS.active() });
  qc.invalidateQueries({ queryKey: KEYS.list() });
  qc.invalidateQueries({ queryKey: ["sets", setId, "tail-recs"] });
}

export function useCreateSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { name: string; notes?: string }) =>
      api.createSet(args.name, args.notes),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: KEYS.list() });
      qc.setQueryData(KEYS.set(created.id), created);
    },
  });
}

export function useActivateSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.activateSet(id),
    onSuccess: (updated) => {
      qc.setQueryData(KEYS.active(), updated);
      qc.setQueryData(KEYS.set(updated.id), updated);
      qc.invalidateQueries({ queryKey: KEYS.list() });
    },
  });
}

export function useDeleteSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteSet(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: KEYS.list() });
      qc.invalidateQueries({ queryKey: KEYS.active() });
    },
  });
}

export function useUpdateSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: {
      id: number;
      patch: { name?: string; notes?: string | null };
    }) => api.updateSet(args.id, args.patch),
    onSuccess: (updated) => {
      qc.setQueryData(KEYS.set(updated.id), updated);
      qc.setQueryData(KEYS.active(), (prev: DanceSet | null | undefined) =>
        prev && prev.id === updated.id ? updated : prev,
      );
      qc.invalidateQueries({ queryKey: KEYS.list() });
    },
  });
}

export function useAddTrackToSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: {
      setId: number;
      trackId: number;
      position?: number;
      note?: string;
    }) =>
      api.addTrackToSet(args.setId, args.trackId, {
        position: args.position,
        note: args.note,
      }),
    onSuccess: (updated) => {
      invalidateSet(qc, updated.id);
      qc.setQueryData(KEYS.set(updated.id), updated);
    },
  });
}

export function useMoveTrackInSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { setId: number; trackId: number; position: number }) =>
      api.moveTrackInSet(args.setId, args.trackId, args.position),
    onSuccess: (updated) => {
      invalidateSet(qc, updated.id);
      qc.setQueryData(KEYS.set(updated.id), updated);
    },
  });
}

export function useRemoveTrackFromSet() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { setId: number; trackId: number }) =>
      api.removeTrackFromSet(args.setId, args.trackId),
    onSuccess: (updated) => {
      invalidateSet(qc, updated.id);
      qc.setQueryData(KEYS.set(updated.id), updated);
    },
  });
}
