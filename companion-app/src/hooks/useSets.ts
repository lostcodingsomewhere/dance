/**
 * React-query hooks for Sets (the persistent curated track plans). A Set is
 * its plan — per-role queues live in the rec grid; see ``useSetPlan``. These
 * hooks cover the set lifecycle (list / active / create / activate / update /
 * delete); plan reads + edits are in ``useSetPlan``.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import type { DanceSet } from "../types";

const KEYS = {
  list: () => ["sets"] as const,
  active: () => ["sets", "active"] as const,
  set: (id: number) => ["sets", id] as const,
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
    // Refresh frequently — other tabs / ⌘K add to the plan. Cheap query.
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

// Mutations -----------------------------------------------------------------

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
