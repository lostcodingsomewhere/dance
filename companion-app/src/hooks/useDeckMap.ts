import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import { store, useAppStore } from "../store";

/**
 * Poll the bridge's view of "what songs are staged on which scenes in Live".
 * This is the source of truth — the FE's localStorage `loadedDecks` is a
 * fallback for when the backend isn't reachable.
 *
 * Auto-reconciles: if backend's scene set differs from what's in localStorage
 * by more than just "backend hasn't received our latest Load yet", we trust
 * the backend and drop the stale local entries. This keeps the "out-of-sync"
 * warning from persisting forever after a reset.
 */
export function useDeckMap() {
  const localDecks = useAppStore((s) => s.loadedDecks);
  const query = useQuery({
    queryKey: ["ableton", "decks"],
    queryFn: api.abletonDeckMap,
    refetchInterval: 2000,
    staleTime: 1000,
  });

  useEffect(() => {
    const scenes = query.data?.scenes;
    if (!scenes) return;
    const backendScenes = new Set(scenes.map((s) => s.scene_index));
    const localScenes = Object.keys(localDecks).map(Number);
    // Drop any local entry whose scene the backend doesn't know about. Don't
    // ADD missing locals — the user might have just loaded and the backend
    // poll hasn't completed yet. Pruning only is safe in both directions.
    let dirty = false;
    for (const sceneIdx of localScenes) {
      if (!backendScenes.has(sceneIdx)) {
        store.unloadDeck(sceneIdx);
        dirty = true;
      }
    }
    // dirty just used as a tracer for future logging hooks.
    void dirty;
  }, [query.data, localDecks]);

  return query;
}

/**
 * Clear the bridge's scene→track map AND the FE's mirror. Use when starting
 * a fresh session in Live (after manually cleaning up old Deck tracks).
 * Does not delete anything in Live — the user is in charge of their session.
 */
export function useResetDecks() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.abletonResetDecks,
    onSuccess: () => {
      store.clearDecks();
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
    },
  });
}

/**
 * Hard cleanup: delete every "Deck *" track in Live via OSC, then reset
 * bridge + FE state. Use when Live's session has orphan deck tracks the
 * bridge doesn't know about (typically after a backend restart).
 */
export function useCleanDecks() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.abletonCleanDecks,
    onSuccess: () => {
      store.clearDecks();
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
    },
  });
}
