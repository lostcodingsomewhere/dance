/**
 * The plan = per-role queues. One GET + a PUT-the-whole-thing model (edits are
 * local: append / remove / reorder, then PUT). Plus per-role plan-recs.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import type { ColumnRecsResponse, SetPlan } from "../types";

const KEY = (setId: number) => ["sets", setId, "plan"] as const;
const SAVE_KEY = (setId: number | null) => ["sets", setId, "plan", "save"] as const;

type OnSaved = (plan: SetPlan) => void;

export interface PlanWritten {
  plan: SetPlan;
  picksLeftOut: number;
}

// PUT /plan refuses a whole plan that names a deleted track, and its 404 lists those ids.
const DELETED_TRACKS = /unknown track ids in plan: \[([^\]]*)\]/;

function deletedTracksNamedBy(error: unknown): Set<number> {
  if (!(error instanceof api.ApiError) || error.status !== 404) return new Set();
  const listed = api.serverMessage(error).match(DELETED_TRACKS)?.[1] ?? "";
  return new Set((listed.match(/\d+/g) ?? []).map(Number));
}

function picksIn(queues: Record<string, number[]>): number {
  return Object.values(queues).reduce((picks, ids) => picks + ids.length, 0);
}

export async function putPlanWithoutDeletedTracks(
  setId: number,
  queues: Record<string, number[]>,
): Promise<PlanWritten> {
  try {
    return { plan: await api.putPlan(setId, queues), picksLeftOut: 0 };
  } catch (error) {
    const deleted = deletedTracksNamedBy(error);
    const kept = Object.fromEntries(
      Object.entries(queues).map(([role, ids]) => [role, ids.filter((id) => !deleted.has(id))]),
    );
    const picksLeftOut = picksIn(queues) - picksIn(kept);
    if (picksLeftOut === 0) throw error;
    return { plan: await api.putPlan(setId, kept), picksLeftOut };
  }
}

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

  const append = useAppendToPlan();
  const put = useMutation({
    mutationKey: SAVE_KEY(setId),
    mutationFn: async (queues: Record<string, number[]>) =>
      (await putPlanWithoutDeletedTracks(setId as number, queues)).plan,
    onSuccess: (plan) => {
      if (setId == null) return;
      qc.setQueryData(KEY(setId), plan);
      qc.invalidateQueries({ queryKey: ["sets", setId] });
    },
  });

  // Derive the current {role: [ids]} from cached SetPlan, apply an edit, PUT.
  //
  // Returns null when the plan has not loaded yet. That case MUST NOT be
  // treated as "an empty plan": PUT /sets/{id}/plan seeds every role empty
  // and overwrites only the roles it is sent, so PUTting a single-role
  // object built from a cold cache silently ERASES the other four columns.
  const currentQueues = (): Record<string, number[]> | null => {
    const plan = qc.getQueryData<SetPlan>(KEY(setId ?? -1));
    if (plan == null) return null;
    const out: Record<string, number[]> = {};
    for (const [role, items] of Object.entries(plan.queues ?? {})) {
      out[role] = items.map((i) => i.track_id);
    }
    return out;
  };

  // Adds go through the SERVER-SIDE append, not read-modify-PUT.
  //
  // The PUT path rebuilds the whole plan from a react-query cache that only
  // refreshes on the previous PUT's response, so two ＋ clicks inside one
  // round-trip both read the same snapshot and the second overwrites the
  // first — the earlier pick vanishes with no error. The append endpoint
  // merges server-side against the row itself, which has neither problem,
  // and it validates the track id.
  const originalOfVersion = (role: string, trackId: number): number | null => {
    const recLists = qc.getQueriesData<ColumnRecsResponse>({
      queryKey: ["sets", setId, "plan-recs", role],
    });
    for (const [, list] of recLists) {
      const rec = list?.recs.find((r) => r.track_id === trackId);
      if (rec?.version_of != null) return rec.version_of;
    }
    return null;
  };

  const addToRole = (role: string, trackId: number) => {
    if (setId == null) return;
    append.mutate({ setId, role, trackId, afterTrackId: originalOfVersion(role, trackId) });
  };

  // A save built from the cache while another is in flight would undo that one.
  const save = (queues: Record<string, number[]>, onSaved?: OnSaved) => {
    if (qc.isMutating({ mutationKey: SAVE_KEY(setId) }) > 0) return;
    put.mutate(queues, { onSuccess: onSaved });
  };

  const removeFromRole = (role: string, index: number, trackId: number, onSaved?: OnSaved) => {
    const q = currentQueues();
    // No server-side remove endpoint exists, so this stays read-modify-PUT —
    // but refuse to run against an unloaded plan rather than wiping it.
    if (q == null) return;
    const list = q[role] ?? [];
    if (list[index] !== trackId) return;
    q[role] = list.filter((_, i) => i !== index);
    save(q, onSaved);
  };

  const restoreToRole = (role: string, index: number, trackId: number, onSaved?: OnSaved) => {
    const q = currentQueues();
    if (q == null) return;
    const list = [...(q[role] ?? [])];
    list.splice(Math.min(index, list.length), 0, trackId);
    q[role] = list;
    save(q, onSaved);
  };

  const moveInRole = (role: string, from: number, to: number, onSaved?: OnSaved) => {
    const q = currentQueues();
    if (q == null) return;
    const list = [...(q[role] ?? [])];
    if (to < 0 || to >= list.length) return;
    const [it] = list.splice(from, 1);
    list.splice(to, 0, it);
    q[role] = list;
    save(q, onSaved);
  };

  return {
    put,
    addToRole,
    removeFromRole,
    restoreToRole,
    moveInRole,
    isPending: put.isPending,
  };
}

/** Append a track to a role queue server-side — safe without the plan loaded
 *  (e.g. from ⌘K). Updates the plan cache on success. */
export function useAppendToPlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: {
      setId: number;
      role: string;
      trackId: number;
      afterTrackId?: number | null;
    }) => api.appendToPlan(v.setId, v.role, v.trackId, v.afterTrackId ?? null),
    onSuccess: (plan, v) => {
      qc.setQueryData(KEY(v.setId), plan);
      qc.invalidateQueries({ queryKey: ["sets", v.setId] });
    },
  });
}

export function usePlanRecs(
  setId: number | null,
  role: string,
  opts: { k?: number; enabled?: boolean } = {},
) {
  const k = opts.k ?? 8;
  return useQuery({
    queryKey: ["sets", setId, "plan-recs", role, k],
    queryFn: () => api.getPlanRecs(setId!, role, k),
    // ``enabled`` lets the Booth ask for plan-scored recs only in the cold
    // case (nothing playing), instead of fetching them on every render.
    enabled: !!setId && (opts.enabled ?? true),
    staleTime: 10_000,
  });
}
