import { useCallback, useEffect, useMemo } from "react";

import { useGridKeys } from "../hooks/useGridKeys";
import { useLoadToDeck } from "../hooks/useLoadToDeck";
import { useStartPreview, useStopPreview, usePreviewState } from "../hooks/usePreview";
import { usePlanMutations } from "../hooks/useSetPlan";
import type { GridFocus } from "../lib/gridNav";
import { store } from "../store";
import { PLAN_ROLES, type PlanRole } from "../types";
import { ColumnHeader, PlanBand, RecsBand } from "./RoleColumn";

/** The recommender calls the whole-track role "mix"; the plan calls it "song". */
function visRole(role: PlanRole): string {
  return role === "song" ? "mix" : role;
}

/**
 * The one grid the app is built around: five role columns, each with your
 * queued plan picks on top and recs below. Used in the Booth (mode="live" —
 * recs tail off what's playing) and the Set page (mode="plan" — recs scored
 * against the rest of the plan). ``setId`` may be null in the Booth when no
 * set is active (then it's just the live recs, no plan zone).
 *
 * ## The audition loop
 *
 * Planning is look → hear → decide → next, and by mouse each iteration costs
 * two trips to 28px buttons, so triaging one screenful is ~40 precise moves.
 * The keyboard collapses that to: ↑↓←→ move, space hears, enter commits.
 *
 * The motions are IDENTICAL in both modes on purpose — only ``enter`` changes
 * meaning (queue a pick vs load a deck). That is what makes planning double as
 * rehearsal for the booth rather than a separate skill to learn.
 */
export function RoleColumnsGrid({
  setId,
  mode,
}: {
  setId: number | null;
  mode: "plan" | "live";
}) {
  const { addToRole } = usePlanMutations(setId);
  const load = useLoadToDeck();
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();

  // Switching mode or set re-populates every column; a cursor left pointing
  // into the old lists would walk into tracks that are no longer there.
  //
  // Only the FOCUS is cleared here, never the published shape. React runs
  // child effects before parent effects, so a resetShape() at this level
  // lands after the bands have already published and silently wipes the very
  // lists it was meant to refresh — leaving the grid unnavigable. Each band
  // owns its own entry instead, republishing when its data changes and
  // clearing it on unmount.
  useEffect(() => {
    store.setGridFocus(null);
  }, [setId, mode]);

  const onAudition = useCallback(
    (trackId: number, focus: GridFocus) => {
      const column = visRole(focus.role);
      // Space toggles, exactly like the card's ▶ — pressing it again on the
      // track you're already hearing stops rather than restarting it.
      if (previewing?.trackId === trackId && previewing?.column === column) {
        stopPreview.mutate();
      } else {
        startPreview.mutate({ trackId, column });
      }
    },
    [previewing, startPreview, stopPreview],
  );

  const onCommit = useCallback(
    (trackId: number, focus: GridFocus, shift: boolean) => {
      if (mode === "live") {
        // enter → Deck A, shift+enter → Deck B. Two decks, two keystrokes.
        load.mutate({ trackId, role: focus.role, title: null, side: shift ? "b" : "a" });
        return;
      }
      if (setId == null) return;
      // Committing a rec leaves the list one shorter and ``reconcileFocus``
      // holds the cursor in place, so the next candidate lands under it and
      // triage continues without touching an arrow key.
      addToRole(focus.role, trackId);
    },
    [mode, setId, addToRole, load],
  );

  const onEscape = useCallback(() => {
    if (previewing) stopPreview.mutate();
    else store.setGridFocus(null);
  }, [previewing, stopPreview]);

  const actions = useMemo(
    () => ({ onAudition, onCommit, onEscape }),
    [onAudition, onCommit, onEscape],
  );
  useGridKeys(actions);

  // Rendered as horizontal BANDS rather than five independent columns:
  //   [headers] · [PLAN band] · divider · [RECS band]
  // The PLAN band is a 5-col grid with ``items-stretch``, so every plan zone
  // (filled card or empty "nothing queued") is the height of the tallest —
  // an uneven plan no longer pushes some columns' Recs below the others. The
  // RECS band is a separate grid with ``items-start``, so a column with fewer
  // recs keeps its own height (no dead space). The divider gives a clear seam
  // between what you've ADDED (plan) and what's RECOMMENDED (recs).
  const cols = "grid grid-cols-5 gap-x-2 min-w-0";
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className={cols}>
        {PLAN_ROLES.map((role) => (
          <ColumnHeader key={role} role={role} />
        ))}
      </div>

      {setId != null && (
        <>
          <div className={`${cols} items-stretch`}>
            {PLAN_ROLES.map((role) => (
              <PlanBand key={role} setId={setId} role={role} mode={mode} />
            ))}
          </div>
          {/* seam between added (plan) and recommended (recs) */}
          <div className="mt-1 border-t border-neutral-800" />
        </>
      )}

      <div className={`${cols} items-start`}>
        {PLAN_ROLES.map((role) => (
          <RecsBand key={role} setId={setId} role={role} mode={mode} />
        ))}
      </div>

      <KeyHints mode={mode} />
    </div>
  );
}

/**
 * The keybindings, always visible.
 *
 * A shortcut nobody knows about is a shortcut nobody uses, and this app is
 * being learned from scratch by the person who built it. Cheap to show,
 * and it doubles as the reminder that the booth uses the same motions.
 */
function KeyHints({ mode }: { mode: "plan" | "live" }) {
  const commit = mode === "live" ? "load Deck A · ⇧⏎ Deck B" : "add to plan";
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-neutral-900 pt-1.5 text-[10px] text-neutral-600">
      <Hint keys="↑↓←→" label="move" />
      <Hint keys="space" label="hear it (headphones)" />
      <Hint keys="⏎" label={commit} />
      <Hint keys="esc" label="stop" />
      <span className="text-neutral-700">
        same keys in {mode === "live" ? "Set" : "Booth"}
      </span>
    </div>
  );
}

function Hint({ keys, label }: { keys: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <kbd className="rounded border border-neutral-700 bg-neutral-900 px-1 py-px font-mono text-[9px] text-neutral-400">
        {keys}
      </kbd>
      <span>{label}</span>
    </span>
  );
}
