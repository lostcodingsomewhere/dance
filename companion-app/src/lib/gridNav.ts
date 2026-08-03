import { PLAN_ROLES, type PlanRole } from "../types";

/**
 * Keyboard navigation over the plan grid — pure, so it can be reasoned about
 * and tested without React or a live backend.
 *
 * WHY this exists. Planning is an audition loop: look → hear → decide → next.
 * Driven by mouse, one iteration costs two trips to 28px buttons (▶ then ＋),
 * so triaging one screenful — 5 roles × 4 recs — is ~40 precise mouse moves.
 * That turns listening into data entry, which is the single biggest reason
 * building a set feels like a chore.
 *
 * The motions are deliberately IDENTICAL in the Set page and the Booth. Only
 * the commit verb differs (queue to plan vs load to a deck). That's what makes
 * planning double as rehearsal: the hands learn one vocabulary and it
 * transfers to the performance, rather than the two surfaces teaching
 * conflicting habits.
 */

export type Zone = "plan" | "recs";

export interface GridFocus {
  role: PlanRole;
  zone: Zone;
  index: number;
}

/** What each column currently holds — track ids per zone, in display order. */
export type GridShape = Partial<Record<PlanRole, { plan: number[]; recs: number[] }>>;

/** Ordered zones within a column: your picks sit above the recommendations. */
const ZONES: Zone[] = ["plan", "recs"];

function lengthOf(shape: GridShape, role: PlanRole, zone: Zone): number {
  return shape[role]?.[zone].length ?? 0;
}

function columnIsEmpty(shape: GridShape, role: PlanRole): boolean {
  return ZONES.every((z) => lengthOf(shape, role, z) === 0);
}

/** The track id under a focus, or null when the focus points at nothing. */
export function focusedTrackId(
  shape: GridShape,
  focus: GridFocus | null,
): number | null {
  if (!focus) return null;
  return shape[focus.role]?.[focus.zone][focus.index] ?? null;
}

/** Is this exact cell the focused one? Drives the focus ring. */
export function isFocused(
  focus: GridFocus | null,
  role: PlanRole,
  zone: Zone,
  index: number,
): boolean {
  return (
    focus != null &&
    focus.role === role &&
    focus.zone === zone &&
    focus.index === index
  );
}

/**
 * Where focus lands when the user first reaches for the keyboard.
 *
 * Recs before plan: the plan is what you already decided, the recs are what
 * you came to judge. Falls through to the first column holding anything, so
 * the first keypress never lands on an empty cell.
 */
export function firstFocus(
  shape: GridShape,
  roles: readonly PlanRole[] = PLAN_ROLES,
): GridFocus | null {
  for (const role of roles) {
    for (const zone of ["recs", "plan"] as Zone[]) {
      if (lengthOf(shape, role, zone) > 0) return { role, zone, index: 0 };
    }
  }
  return null;
}

/**
 * Move within a column. A column reads as one vertical list — plan picks on
 * top, recs beneath — so ↓ off the end of the plan continues into the recs
 * rather than dead-ending at the seam.
 */
function moveVertical(
  shape: GridShape,
  focus: GridFocus,
  step: 1 | -1,
): GridFocus {
  const flat: Array<{ zone: Zone; index: number }> = [];
  for (const zone of ZONES) {
    for (let i = 0; i < lengthOf(shape, focus.role, zone); i++) {
      flat.push({ zone, index: i });
    }
  }
  if (flat.length === 0) return focus;
  const at = flat.findIndex(
    (c) => c.zone === focus.zone && c.index === focus.index,
  );
  // An unknown position (the list shrank under us) snaps to an end rather
  // than refusing to move.
  const next = at === -1 ? (step === 1 ? 0 : flat.length - 1) : at + step;
  const clamped = Math.max(0, Math.min(flat.length - 1, next));
  return { role: focus.role, ...flat[clamped] };
}

/**
 * Move between columns, preserving the zone.
 *
 * Preserving the ZONE (rather than a flattened offset) is what keeps this
 * predictable: browsing recs and stepping sideways must land in recs, never
 * drop into the neighbour's plan just because that column queued more picks.
 * Empty columns are skipped entirely — there is no reason to strand focus
 * somewhere with nothing to audition.
 */
function moveHorizontal(
  shape: GridShape,
  focus: GridFocus,
  step: 1 | -1,
  roles: readonly PlanRole[],
): GridFocus {
  const at = roles.indexOf(focus.role);
  if (at === -1) return focus;
  for (let i = at + step; i >= 0 && i < roles.length; i += step) {
    const role = roles[i];
    if (columnIsEmpty(shape, role)) continue;
    // Same zone when it has anything, otherwise the column's other zone.
    const zone: Zone =
      lengthOf(shape, role, focus.zone) > 0
        ? focus.zone
        : ZONES.find((z) => lengthOf(shape, role, z) > 0)!;
    const index = Math.min(focus.index, lengthOf(shape, role, zone) - 1);
    return { role, zone, index };
  }
  return focus; // already at the last column with content
}

export type NavKey = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight";

/**
 * Apply one arrow key. Returns the new focus, or the same object when the
 * move is a no-op (already at an edge) so callers can skip a re-render.
 */
export function moveFocus(
  shape: GridShape,
  focus: GridFocus | null,
  key: NavKey,
  roles: readonly PlanRole[] = PLAN_ROLES,
): GridFocus | null {
  if (focus == null) return firstFocus(shape, roles);
  // Focus pointing at a column that no longer holds anything (a rec was
  // queued away, a plan pick removed) restarts rather than trapping the user.
  if (columnIsEmpty(shape, focus.role)) return firstFocus(shape, roles);
  switch (key) {
    case "ArrowDown":
      return moveVertical(shape, focus, 1);
    case "ArrowUp":
      return moveVertical(shape, focus, -1);
    case "ArrowRight":
      return moveHorizontal(shape, focus, 1, roles);
    case "ArrowLeft":
      return moveHorizontal(shape, focus, -1, roles);
  }
}

/**
 * Keep a focus valid after the underlying lists change.
 *
 * Committing a rec is the common case: it leaves the recs list one shorter,
 * and the natural expectation is that the NEXT candidate is now under the
 * cursor — same position, new occupant — so triage keeps flowing without
 * touching the arrow keys. Only when the list ran out does focus move.
 */
export function reconcileFocus(
  shape: GridShape,
  focus: GridFocus | null,
  roles: readonly PlanRole[] = PLAN_ROLES,
): GridFocus | null {
  if (focus == null) return null;
  const len = lengthOf(shape, focus.role, focus.zone);
  if (focus.index < len) return focus; // still points at something
  if (len > 0) return { ...focus, index: len - 1 }; // list shrank past us
  const other = ZONES.find((z) => lengthOf(shape, focus.role, z) > 0);
  if (other) {
    return { role: focus.role, zone: other, index: 0 };
  }
  return firstFocus(shape, roles); // whole column emptied out
}
