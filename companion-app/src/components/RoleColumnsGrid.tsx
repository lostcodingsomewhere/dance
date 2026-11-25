import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useGridKeys } from "../hooks/useGridKeys";
import { useLoadToDeck, type LoadToDeckVars } from "../hooks/useLoadToDeck";
import { useStartPreview, useStopPreview, usePreviewState } from "../hooks/usePreview";
import { usePlanMutations } from "../hooks/useSetPlan";
import type { GridFocus } from "../lib/gridNav";
import { store } from "../store";
import {
  PLAN_ROLES,
  type ColumnRecsResponse,
  type PlanItem,
  type PlanRole,
  type SetPlan,
} from "../types";
import { Modal } from "./Modal";
import { PadEditor } from "./PadEditor";
import {
  ColumnHeader,
  PlanBand,
  PlanRow,
  RecsBand,
  RecsRow,
  usePlanBandShown,
  whyNotReady,
} from "./RoleColumn";
import { VersionFinder } from "./VersionFinder";

/** The recommender calls the whole-track role "mix"; the plan calls it "song". */
function visRole(role: PlanRole): string {
  return role === "song" ? "mix" : role;
}

function titleOfPick(
  queryClient: QueryClient,
  setId: number | null,
  role: PlanRole,
  trackId: number,
): string | null {
  const queued = queryClient
    .getQueryData<SetPlan>(["sets", setId, "plan"])
    ?.queues?.[role]?.find((item) => item.track_id === trackId);
  if (queued?.title) return queued.title;
  const recLists = [
    ...queryClient.getQueriesData<ColumnRecsResponse>({
      queryKey: ["recommend", "by-column", visRole(role)],
    }),
    ...queryClient.getQueriesData<ColumnRecsResponse>({
      queryKey: ["sets", setId, "plan-recs", role],
    }),
  ];
  for (const [, list] of recLists) {
    const title = list?.recs?.find((rec) => rec.track_id === trackId)?.track_title;
    if (title) return title;
  }
  return null;
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
  const queryClient = useQueryClient();
  const keyboardLoadInFlight = useRef(false);
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const [padsPick, setPadsPick] = useState<PlanItem | null>(null);
  const closePads = useCallback(() => setPadsPick(null), []);
  const [versionsPick, setVersionsPick] = useState<PlanItem | null>(null);
  const closeVersions = useCallback(() => setVersionsPick(null), []);
  const planBandShown = usePlanBandShown(setId, mode);

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
        const queued =
          focus.zone === "plan"
            ? queryClient
                .getQueryData<SetPlan>(["sets", setId, "plan"])
                ?.queues?.[focus.role]?.find((item) => item.track_id === trackId)
            : undefined;
        const notReady = queued ? whyNotReady(queued) : null;
        if (queued && notReady) {
          store.reportFailure(`load ${queued.title ?? `Track #${trackId}`}`, [notReady]);
          return;
        }
        if (keyboardLoadInFlight.current) return;
        keyboardLoadInFlight.current = true;
        // enter → Deck A, shift+enter → Deck B. Two decks, two keystrokes.
        load.mutate(
          {
            trackId,
            role: focus.role,
            title: titleOfPick(queryClient, setId, focus.role, trackId),
            side: shift ? "b" : "a",
          },
          {
            onSettled: () => {
              keyboardLoadInFlight.current = false;
            },
          },
        );
        return;
      }
      if (setId == null) return;
      // Committing a rec leaves the list one shorter and ``reconcileFocus``
      // holds the cursor in place, so the next candidate lands under it and
      // triage continues without touching an arrow key.
      addToRole(focus.role, trackId);
    },
    [mode, setId, addToRole, load, queryClient],
  );

  const onEscape = useCallback(() => {
    if (previewing) stopPreview.mutate();
    else store.setGridFocus(null);
  }, [previewing, stopPreview]);

  const actions = useMemo(
    () => ({ onAudition, onCommit, onEscape }),
    [onAudition, onCommit, onEscape],
  );
  useGridKeys(actions, padsPick == null && versionsPick == null);

  // Rendered as horizontal BANDS rather than five independent columns:
  //   [headers] · [PLAN band] · divider · [RECS band]
  // In the Set view the PLAN band is a 5-col grid with ``items-stretch``, so
  // every plan zone (filled card or empty "nothing queued") is the height of
  // the tallest — an uneven plan no longer pushes some columns' Recs below the
  // others. In the Booth it is ``items-start``: each plan zone is capped at a
  // few one-line rows and scrolls inside itself, and the band is left out
  // while nothing is planned. The RECS band is a separate grid with
  // ``items-start``, so a column with fewer recs keeps its own height (no dead
  // space). The divider gives a clear seam between what you've ADDED (plan)
  // and what's RECOMMENDED (recs).
  const cols = "grid grid-cols-5 gap-x-2 min-w-0";
  return (
    <div data-plan-grid className="flex flex-col gap-1 min-w-0">
      <div className={cols}>
        {PLAN_ROLES.map((role) => (
          <ColumnHeader key={role} role={role} mode={mode} />
        ))}
      </div>

      {setId != null && planBandShown && (
        <>
          <PlanRow
            setId={setId}
            className={`${cols} ${mode === "live" ? "items-start" : "items-stretch"}`}
          >
            {PLAN_ROLES.map((role) => (
              <PlanBand
                key={role}
                setId={setId}
                role={role}
                mode={mode}
                onOpenPads={mode === "plan" ? setPadsPick : undefined}
                onFindVersions={mode === "plan" && role === "song" ? setVersionsPick : undefined}
              />
            ))}
          </PlanRow>
          {/* seam between added (plan) and recommended (recs) */}
          <div data-plan-seam className="mt-1 border-t border-neutral-800" />
        </>
      )}

      <RecsRow setId={setId} mode={mode} className={`${cols} items-start`}>
        {PLAN_ROLES.map((role) => (
          <RecsBand key={role} setId={setId} role={role} mode={mode} />
        ))}
      </RecsRow>

      <KeyHints mode={mode} loading={load.isPending ? load.variables : undefined} />

      <Modal
        open={padsPick != null}
        onClose={closePads}
        title={padsPick ? `Pads · ${padsPick.title ?? `Track #${padsPick.track_id}`}` : undefined}
        widthClass="max-w-4xl"
      >
        {padsPick && <PadEditor trackId={padsPick.track_id} />}
      </Modal>

      <Modal
        open={versionsPick != null}
        onClose={closeVersions}
        title={
          versionsPick
            ? `Versions · ${versionsPick.title ?? `Track #${versionsPick.track_id}`}`
            : undefined
        }
      >
        {versionsPick && setId != null && (
          <VersionFinder trackId={versionsPick.track_id} setId={setId} />
        )}
      </Modal>
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
function KeyHints({ mode, loading }: { mode: "plan" | "live"; loading?: LoadToDeckVars }) {
  const live = mode === "live";
  const commit = live ? "load Deck A · ⇧⏎ Deck B" : "add to plan";
  return (
    <div
      className={`mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-neutral-900 pt-1.5 ${
        live ? "text-xs text-neutral-400" : "text-[10px] text-neutral-600"
      }`}
    >
      <Hint keys="↑↓←→" label="move" live={live} />
      <Hint keys="space" label="hear it (headphones)" live={live} />
      <Hint keys="⏎" label={commit} live={live} />
      <Hint keys="esc" label="stop" live={live} />
      <span className={live ? "" : "text-neutral-700"}>
        same keys in {live ? "Set" : "Booth"}
      </span>
      {loading && (
        <span role="status" className="text-amber-200">
          {`loading ${loading.title ?? `Track #${loading.trackId}`}${loading.side ? ` onto Deck ${loading.side.toUpperCase()}` : ""}…`}
        </span>
      )}
    </div>
  );
}

function Hint({ keys, label, live }: { keys: string; label: string; live: boolean }) {
  return (
    <span className="inline-flex items-center gap-1">
      <kbd
        className={`rounded border border-neutral-700 bg-neutral-900 px-1 py-px font-mono text-neutral-400 ${
          live ? "text-xs" : "text-[9px]"
        }`}
      >
        {keys}
      </kbd>
      <span>{label}</span>
    </span>
  );
}
