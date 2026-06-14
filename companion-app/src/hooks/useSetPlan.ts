/**
 * The plan = per-role queues. One GET + a PUT-the-whole-thing model (edits are
 * local: append / remove / reorder, then PUT). Plus per-role plan-recs.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import type { SetPlan } from "../types";

const KEY = (setId: number) => ["sets", setId, "plan"] as const;

export function useSetPlan(setId: number | null | undefined) {
  return useQuery({
    queryKey: setId ? KEY(setId) : ["sets", "none", "plan"],
    queryFn: () => (setId ? api.getPlan(setId) : null),
    enabled: !!setId,
    staleTime: 5_000,
  });
}

/** Mutations that rewrite the plan from the current queues + a local edit.
 *  ``setId`` may be null (e.g. Booth with no active set) — mutations no-op. */
export function usePlanMutations(setId: number | null) {
  const qc = useQueryClient();

  const put = useMutation({
    mutationFn: (queues: Record<string, number[]>) => api.putPlan(setId as number, queues),
    onSuccess: (plan) => {
      if (setId == null) return;
      qc.setQueryData(KEY(setId), plan);
      qc.invalidateQueries({ queryKey: ["sets", setId] });
    },
  });

  // Derive the current {role: [ids]} from cached SetPlan, apply an edit, PUT.
  const currentQueues = (): Record<string, number[]> => {
    const plan = qc.getQueryData<SetPlan>(KEY(setId ?? -1));
    const out: Record<string, number[]> = {};
    for (const [role, items] of Object.entries(plan?.queues ?? {})) {
      out[role] = items.map((i) => i.track_id);
    }
    return out;
  };

  const addToRole = (role: string, trackId: number) => {
    if (setId == null) return;
    const q = currentQueues();
    const list = q[role] ?? [];
    if (!list.includes(trackId)) q[role] = [...list, trackId];
    put.mutate(q);
  };

  const removeFromRole = (role: string, index: number) => {
    const q = currentQueues();
    q[role] = (q[role] ?? []).filter((_, i) => i !== index);
    put.mutate(q);
  };

  const moveInRole = (role: string, from: number, to: number) => {
    const q = currentQueues();
    const list = [...(q[role] ?? [])];
    if (to < 0 || to >= list.length) return;
    const [it] = list.splice(from, 1);
    list.splice(to, 0, it);
    q[role] = list;
    put.mutate(q);
  };

  return { put, addToRole, removeFromRole, moveInRole, isPending: put.isPending };
}

/** Append a track to a role queue server-side — safe without the plan loaded
 *  (e.g. from ⌘K). Updates the plan cache on success. */
export function useAppendToPlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { setId: number; role: string; trackId: number }) =>
      api.appendToPlan(v.setId, v.role, v.trackId),
    onSuccess: (plan, v) => {
      qc.setQueryData(KEY(v.setId), plan);
      qc.invalidateQueries({ queryKey: ["sets", v.setId] });
    },
  });
}

export function usePlanRecs(setId: number | null, role: string, k = 8) {
  return useQuery({
    queryKey: ["sets", setId, "plan-recs", role, k],
    queryFn: () => api.getPlanRecs(setId!, role, k),
    enabled: !!setId,
    staleTime: 10_000,
  });
}
