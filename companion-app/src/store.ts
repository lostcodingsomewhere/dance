// Tiny ad-hoc app store using useSyncExternalStore. No external state lib.
// Holds: seeds, pinned seeds (a subset that the user explicitly pinned),
// currentSessionId, and currentView.

import { useSyncExternalStore } from "react";
import type { ViewName } from "./types";

interface AppState {
  pinnedSeeds: number[]; // tracks the user manually pinned as "next"
  currentSessionId: number | null;
  currentView: ViewName;
}

const initial: AppState = {
  pinnedSeeds: [],
  currentSessionId: null,
  currentView: "now",
};

let state: AppState = initial;
const listeners = new Set<() => void>();

function emit(): void {
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

export const store = {
  setView(view: ViewName): void {
    if (state.currentView === view) return;
    state = { ...state, currentView: view };
    emit();
  },
  setSessionId(id: number | null): void {
    state = { ...state, currentSessionId: id };
    emit();
  },
  pin(id: number): void {
    if (state.pinnedSeeds.includes(id)) return;
    state = { ...state, pinnedSeeds: [...state.pinnedSeeds, id] };
    emit();
  },
  unpin(id: number): void {
    state = {
      ...state,
      pinnedSeeds: state.pinnedSeeds.filter((s) => s !== id),
    };
    emit();
  },
  clearPins(): void {
    state = { ...state, pinnedSeeds: [] };
    emit();
  },
};
