// Tiny ad-hoc app store using useSyncExternalStore. Holds: current view,
// session id, and the "loaded decks" map (scene_index → track_id) that
// connects our Push-to-Live calls to the playing_clips state coming back
// from Ableton over the websocket. That linkage is what makes auto-logging
// of plays possible without a backend change.

import { useSyncExternalStore } from "react";
import type { LoadedDeck, ViewName } from "./types";

interface AppState {
  currentSessionId: number | null;
  currentView: ViewName;
  /** Keyed by scene_index. One deck per scene. */
  loadedDecks: Record<number, LoadedDeck>;
  commandBarOpen: boolean;
  /**
   * Currently auditioning candidate in the Cue track (headphones-only via
   * Scarlett outs 3/4). Cleared on stopPreview or when the user commits a
   * track to a real deck row. Local state only — backend tracks the actual
   * Cue clip via the bridge.
   */
  previewing: {
    trackId: number;
    column: string;
    cueTrackIdx?: number | null;
    slot?: number | null;
  } | null;
  /**
   * Visual order of the 5 stem columns in the Booth (drums/bass/vocals/
   * other/mix). Defaults to the canonical order from ``lib/roles``, but
   * the user can drag column headers around to reorder. Persisted to
   * localStorage so the layout sticks across reloads. Reordering is
   * purely cosmetic — the underlying Ableton track indices don't move;
   * we just iterate this list when rendering SceneGrid / ComboStrip /
   * BoothColumnHeaders / rec banners.
   */
  stemColumnOrder: string[];
}

const STORAGE_KEY = "dance.companion.state.v2";

// Canonical default order — mirrors ``lib/roles.STEM_COLUMNS``. Re-declared
// here to avoid a circular import (the store can't reach into ``lib`` if
// ``lib`` ever imports the store). Keep the two in sync.
const DEFAULT_STEM_COLUMN_ORDER = ["drums", "bass", "vocals", "other", "mix"];

function readPersisted(): Partial<AppState> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<AppState>;
    // ``loadedDecks`` survives reloads so the SceneGrid mirror keeps its
    // mapping of which dance-track is in which scene. ``stemColumnOrder`` also
    // persists so the user's column layout sticks. The rest (open, previewing)
    // is intentionally ephemeral.
    const persistedOrder = parsed.stemColumnOrder;
    // Validate that the persisted order has all 5 expected columns and
    // nothing extra (defends against storage rot if we ever rename a stem).
    const isValidOrder =
      Array.isArray(persistedOrder) &&
      persistedOrder.length === DEFAULT_STEM_COLUMN_ORDER.length &&
      DEFAULT_STEM_COLUMN_ORDER.every((c) => persistedOrder.includes(c));
    return {
      loadedDecks: parsed.loadedDecks ?? {},
      stemColumnOrder: isValidOrder
        ? persistedOrder
        : DEFAULT_STEM_COLUMN_ORDER,
    };
  } catch {
    return {};
  }
}

function persist(s: AppState): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        loadedDecks: s.loadedDecks,
        stemColumnOrder: s.stemColumnOrder,
      }),
    );
  } catch {
    // localStorage full or unavailable — ignore.
  }
}

const initial: AppState = {
  currentSessionId: null,
  currentView: "booth",
  loadedDecks: {},
  commandBarOpen: false,
  previewing: null,
  stemColumnOrder: DEFAULT_STEM_COLUMN_ORDER,
  ...readPersisted(),
};

let state: AppState = initial;
const listeners = new Set<() => void>();

function emit(): void {
  persist(state);
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

function getSnapshot(): AppState {
  return state;
}

export function useAppStore<T>(selector: (s: AppState) => T): T {
  return useSyncExternalStore(
    subscribe,
    () => selector(state),
    () => selector(initial),
  );
}

export function useAppState(): AppState {
  return useSyncExternalStore(subscribe, getSnapshot, () => initial);
}

/**
 * Compute the next free scene index from a loadedDecks map. Songs stack
 * vertically — first load goes to scene 0, then 1, etc. — so we just pick
 * one above the max in use.
 */
export function nextSceneIndex(
  loadedDecks: Record<number, LoadedDeck>,
): number {
  const used = Object.keys(loadedDecks).map(Number);
  if (used.length === 0) return 0;
  return Math.max(...used) + 1;
}

export const store = {
  setView(view: ViewName): void {
    if (state.currentView === view) return;
    state = { ...state, currentView: view };
    emit();
  },
  /** Same as nextSceneIndex(state.loadedDecks) but for non-React callsites. */
  peekNextScene(): number {
    return nextSceneIndex(state.loadedDecks);
  },
  setSessionId(id: number | null): void {
    state = { ...state, currentSessionId: id };
    emit();
  },
  registerDeck(deck: LoadedDeck): void {
    state = {
      ...state,
      loadedDecks: { ...state.loadedDecks, [deck.scene_index]: deck },
    };
    emit();
  },
  unloadDeck(sceneIndex: number): void {
    if (state.loadedDecks[sceneIndex] == null) return;
    const next = { ...state.loadedDecks };
    delete next[sceneIndex];
    state = { ...state, loadedDecks: next };
    emit();
  },
  clearDecks(): void {
    state = { ...state, loadedDecks: {} };
    emit();
  },
  openCommandBar(): void {
    if (state.commandBarOpen) return;
    state = { ...state, commandBarOpen: true };
    emit();
  },
  closeCommandBar(): void {
    if (!state.commandBarOpen) return;
    state = { ...state, commandBarOpen: false };
    emit();
  },
  /** Replace the column order — typically called when a drag-and-drop
   *  on the Booth's column headers settles. Validates the new array has
   *  exactly the 5 canonical columns; falls back to the default if not.
   *  Cheap defense against bad payloads (e.g. a stale browser tab). */
  setStemColumnOrder(order: string[]): void {
    const sameLen = order.length === DEFAULT_STEM_COLUMN_ORDER.length;
    const sameSet =
      sameLen && DEFAULT_STEM_COLUMN_ORDER.every((c) => order.includes(c));
    const next = sameSet ? [...order] : DEFAULT_STEM_COLUMN_ORDER;
    if (
      state.stemColumnOrder.length === next.length &&
      state.stemColumnOrder.every((v, i) => v === next[i])
    )
      return;
    state = { ...state, stemColumnOrder: next };
    emit();
  },
  resetStemColumnOrder(): void {
    state = { ...state, stemColumnOrder: DEFAULT_STEM_COLUMN_ORDER };
    emit();
  },
  setPreviewing(
    p:
      | {
          trackId: number;
          column: string;
          cueTrackIdx?: number | null;
          slot?: number | null;
        }
      | null,
  ): void {
    state = { ...state, previewing: p };
    emit();
  },
};
