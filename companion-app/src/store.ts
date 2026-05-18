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
  /** Ordered set of track ids the user is staging for a future set. */
  stack: number[];
  commandBarOpen: boolean;
  /**
   * Currently auditioning candidate in the Cue track (headphones-only via
   * Scarlett outs 3/4). Cleared on stopPreview or when the user commits a
   * track to a real deck row. Local state only — backend tracks the actual
   * Cue clip via the bridge.
   */
  previewing: { trackId: number; column: string } | null;
}

const STORAGE_KEY = "dance.companion.state.v2";

function readPersisted(): Partial<AppState> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<AppState>;
    return {
      loadedDecks: parsed.loadedDecks ?? {},
      stack: parsed.stack ?? [],
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
      JSON.stringify({ loadedDecks: s.loadedDecks, stack: s.stack }),
    );
  } catch {
    // localStorage full or unavailable — ignore.
  }
}

const initial: AppState = {
  currentSessionId: null,
  currentView: "booth",
  loadedDecks: {},
  stack: [],
  commandBarOpen: false,
  previewing: null,
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
  addToStack(id: number): void {
    if (state.stack.includes(id)) return;
    state = { ...state, stack: [...state.stack, id] };
    emit();
  },
  removeFromStack(id: number): void {
    state = { ...state, stack: state.stack.filter((x) => x !== id) };
    emit();
  },
  moveInStack(from: number, to: number): void {
    if (from === to || from < 0 || to < 0) return;
    const arr = state.stack.slice();
    if (from >= arr.length || to >= arr.length) return;
    const [item] = arr.splice(from, 1);
    arr.splice(to, 0, item);
    state = { ...state, stack: arr };
    emit();
  },
  clearStack(): void {
    state = { ...state, stack: [] };
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
  setPreviewing(p: { trackId: number; column: string } | null): void {
    state = { ...state, previewing: p };
    emit();
  },
};
