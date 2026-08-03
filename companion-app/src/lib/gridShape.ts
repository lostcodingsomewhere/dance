import type { GridShape, Zone } from "./gridNav";
import type { PlanRole } from "../types";

/**
 * What the plan grid currently holds, published by the bands that render it.
 *
 * Deliberately NOT React state. Each band fetches its own list through its own
 * query hook, so the shape is assembled from five independent sources; routing
 * that through the app store would re-render every column whenever any one of
 * them settled. Keyboard navigation only needs the shape at the instant a key
 * is pressed, so a plain module-level record read on demand is both cheaper
 * and more honest about the access pattern.
 *
 * Subscribers exist for one narrow job: re-validating the focus after a list
 * changes underneath it (see ``reconcileFocus``).
 */
const shape: GridShape = {};
const subscribers = new Set<() => void>();

function sameIds(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

/** Record one zone's track ids. No-ops when nothing actually changed, so a
 *  re-render with identical data doesn't churn focus. */
export function publishZone(role: PlanRole, zone: Zone, ids: number[]): void {
  const current = shape[role] ?? { plan: [], recs: [] };
  if (sameIds(current[zone], ids)) return;
  shape[role] = { ...current, [zone]: [...ids] };
  for (const fn of subscribers) fn();
}

/** The live shape. Treat as read-only. */
export function readShape(): GridShape {
  return shape;
}

export function onShapeChange(fn: () => void): () => void {
  subscribers.add(fn);
  return () => {
    subscribers.delete(fn);
  };
}

/** Drop everything — a column that unmounts (mode switch, set change) must not
 *  leave stale ids behind for the cursor to walk into. */
export function resetShape(): void {
  for (const key of Object.keys(shape)) {
    delete shape[key as PlanRole];
  }
  for (const fn of subscribers) fn();
}
