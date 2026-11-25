import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api";
import { sideOf } from "../lib/roles";
import { store } from "../store";
import type { DeckCell, LoadedDeck } from "../types";
import { useAbletonState } from "./useAbletonState";

function liveHolds(cells: DeckCell[], deck: LoadedDeck): boolean {
  return cells.some(
    (cell) =>
      cell.scene_index === deck.scene_index &&
      (deck.side == null || sideOf(cell.kind) === deck.side),
  );
}

/**
 * Poll the bridge's view of "what cells are loaded in Live". This is the
 * source of truth — the FE's localStorage `loadedDecks` is a hint that
 * gets pruned when the backend disagrees (post-Phase-7 we read cells,
 * not scenes; a scene "exists" in the auto-reconcile sense if any of
 * its cells are populated).
 *
 * Freshness comes from the WebSocket contract's ``deck_map_revision``: it
 * bumps whenever the loaded deck-cell map changes (load / clear /
 * anchor-fill), and we invalidate the deck-map query the instant it does,
 * so the grid reflects edits immediately instead of up to a poll-tick
 * late. The HTTP poll is kept slow (10s) purely as a fallback for the
 * case where the WS frame is missed or the revision doesn't move.
 */
export function useDeckMap() {
  const qc = useQueryClient();
  const deckMapRevision = useAbletonState().deck_map_revision;
  const query = useQuery({
    queryKey: ["ableton", "decks"],
    queryFn: api.abletonDeckMap,
    refetchInterval: 10000,
    staleTime: 1000,
  });

  // Refetch the instant the bridge signals the deck-cell map changed.
  // The slow poll above only matters if a revision bump is ever missed.
  useEffect(() => {
    qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
  }, [deckMapRevision, qc]);

  // A load made after this map was fetched is not in it yet, so only a new map prunes.
  useEffect(() => {
    const cells = query.data?.cells;
    if (!cells) return;
    for (const deck of Object.values(store.peekLoadedDecks())) {
      if (!liveHolds(cells, deck)) store.unloadDeck(deck.scene_index, deck.side);
    }
  }, [query.data]);

  return query;
}

/**
 * Clear the bridge's deck-column + cell maps AND the FE's mirror. Does
 * not delete anything in Live — the user is in charge of their session.
 */
export function useResetDecks() {
  const qc = useQueryClient();
  return useMutation({
    meta: { attempt: "reset the decks" },
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
 * bridge doesn't know about (typically after a backend restart) and you
 * want to start from a clean slate.
 */
export function useCleanDecks() {
  const qc = useQueryClient();
  return useMutation({
    meta: { attempt: "clean up the decks" },
    mutationFn: api.abletonCleanDecks,
    onSuccess: () => {
      store.clearDecks();
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
    },
  });
}

/**
 * Non-destructive resync: scan Live's deck-column clip slots and adopt
 * whatever clips are there into the bridge's cell map. Lets the FE
 * render the correct grid even when the in-memory state diverged from
 * Live (e.g. backend restart, manually-placed clips).
 */
export function useResyncDecks() {
  const qc = useQueryClient();
  return useMutation({
    meta: { attempt: "resync with Live" },
    mutationFn: api.abletonResyncDecks,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
    },
  });
}
