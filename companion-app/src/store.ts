// Tiny ad-hoc app store using useSyncExternalStore. Holds: current view,
// session id, and the "loaded decks" map (scene_index → track_id) that
// connects our Push-to-Live calls to the playing_clips state coming back
// from Ableton over the websocket. That linkage is what makes auto-logging
// of plays possible without a backend change.

import { useSyncExternalStore } from "react";
import type { ColumnRec, LoadedDeck, ViewName } from "./types";

interface AppState {
  currentSessionId: number | null;
  currentView: ViewName;
  /** Keyed by scene_index. One deck per scene. */
  loadedDecks: Record<number, LoadedDeck>;
  commandBarOpen: boolean;
  /** Whether the Set Rail drawer is open in Booth. Auto-collapses 3s after
   *  a clip fire so the SceneGrid stays sovereign during a mix. */
  setRailOpen: boolean;
  /**
   * Currently auditioning candidate in the Cue track (headphones-only via
   * Scarlett outs 3/4). Cleared on stopPreview or when the user commits a
   * track to a real deck row. Local state only — backend tracks the actual
   * Cue clip via the bridge.
   */
  previewing: { trackId: number; column: string } | null;
  /**
   * Soft queue of tracks the user wants surfaced as whole-song candidates
   * in the Mix-column rec banner. Sourced from rail taps and the legacy
   * pinToSong gesture. Runtime-only — the persisted equivalent is the
   * active Set; pins are ephemeral selections within it.
   */
  pinnedSongRecs: ColumnRec[];
}

const STORAGE_KEY = "dance.companion.state.v2";

function readPersisted(): Partial<AppState> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<AppState>;
    // Only loadedDecks persists — legacy ``stack`` and ``pinnedSongRecs``
    // keys may still live in storage from older versions; the migration
    // prompt reads ``stack`` directly to import it as a Set.
    return {
      loadedDecks: parsed.loadedDecks ?? {},
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
  setRailOpen: false,
  previewing: null,
  pinnedSongRecs: [],
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
  openSetRail(): void {
    if (state.setRailOpen) return;
    state = { ...state, setRailOpen: true };
    emit();
  },
  closeSetRail(): void {
    if (!state.setRailOpen) return;
    state = { ...state, setRailOpen: false };
    emit();
  },
  toggleSetRail(): void {
    state = { ...state, setRailOpen: !state.setRailOpen };
    emit();
  },
  setPreviewing(p: { trackId: number; column: string } | null): void {
    state = { ...state, previewing: p };
    emit();
  },
  /**
   * Pin a rec to the top of the SONG column's rec list. Used when the
   * user is browsing stem recs (e.g. vocals candidates) and wants to
   * remember "this whole track is worth a look in song mode." Stored
   * with stem_file_id forced to null so the pinned card renders as a
   * song candidate, not a stem one.
   */
  pinToSong(rec: ColumnRec): void {
    if (state.pinnedSongRecs.some((r) => r.track_id === rec.track_id)) return;
    const songified: ColumnRec = { ...rec, stem_file_id: null };
    state = {
      ...state,
      pinnedSongRecs: [songified, ...state.pinnedSongRecs],
    };
    emit();
  },
  unpinFromSong(trackId: number): void {
    if (!state.pinnedSongRecs.some((r) => r.track_id === trackId)) return;
    state = {
      ...state,
      pinnedSongRecs: state.pinnedSongRecs.filter((r) => r.track_id !== trackId),
    };
    emit();
  },
};
