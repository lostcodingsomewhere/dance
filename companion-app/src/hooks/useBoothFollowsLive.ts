import { useEffect, useRef } from "react";

import { pageCount, pageOfScene } from "../lib/pages";
import { sideOf } from "../lib/roles";
import { store, useAppStore } from "../store";
import type { DeckMap } from "../types";
import { useAbletonState } from "./useAbletonState";
import { useDeckMap } from "./useDeckMap";

interface ClipStart {
  kind: string;
  scene: number;
}

function clipsThatStarted(
  columns: NonNullable<DeckMap["columns"]>,
  playing: Record<string, number>,
  before: Record<string, number>,
): ClipStart[] {
  return Object.entries(columns)
    .filter(([, liveTrack]) => playing[liveTrack] != null && playing[liveTrack] !== before[liveTrack])
    .map(([kind, liveTrack]) => ({ kind, scene: playing[liveTrack] }));
}

function hasCellOnSide(deckMap: DeckMap, side: "a" | "b", scene: number): boolean {
  return deckMap.cells.some((cell) => cell.scene_index === scene && sideOf(cell.kind) === side);
}

function forgetArmsTheSetHasNoRowFor(deckMap: DeckMap, mapReadAt: number): void {
  const { armed, armedAt } = store.peekArms();
  for (const side of ["a", "b"] as const) {
    const scene = armed[side];
    const mapIsNewerThanTheArm = mapReadAt > armedAt[side];
    if (scene != null && mapIsNewerThanTheArm && !hasCellOnSide(deckMap, side, scene)) {
      store.clearArm(side);
    }
  }
}

function keepThePageInsideTheSet(deckMap: DeckMap, mapReadAt: number): void {
  const { armedAt } = store.peekArms();
  // A load turns to its page before the map that holds the page is read again.
  if (mapReadAt <= Math.max(armedAt.a, armedAt.b)) return;
  const lastPage = Math.max(0, pageCount(deckMap.cells) - 1);
  if (store.peekBoothPage() > lastPage) store.setBoothPage(lastPage);
}

function followTheStarts(starts: ClipStart[]): void {
  const latest = starts.reduce((highest, start) => (start.scene > highest.scene ? start : highest));
  store.setBoothPage(pageOfScene(latest.scene));
  for (const start of starts) {
    const side = sideOf(start.kind);
    if (side) store.clearArm(side);
  }
}

export function useBoothFollowsLive(): void {
  const playing = useAbletonState().playing_clips;
  const deckMap = useDeckMap();
  const lastSeen = useRef<Record<string, number> | null>(null);
  const armed = useAppStore((s) => s.armed);

  useEffect(() => {
    if (!deckMap.data) return;
    forgetArmsTheSetHasNoRowFor(deckMap.data, deckMap.dataUpdatedAt);
    keepThePageInsideTheSet(deckMap.data, deckMap.dataUpdatedAt);
  }, [deckMap.data, deckMap.dataUpdatedAt, armed]);

  useEffect(() => {
    const columns = deckMap.data?.columns;
    if (!columns || playing == null) return;
    const starts = clipsThatStarted(columns, playing, lastSeen.current ?? {});
    lastSeen.current = playing;
    if (starts.length > 0) followTheStarts(starts);
  }, [deckMap.data?.columns, playing]);
}
