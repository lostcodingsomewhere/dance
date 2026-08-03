import type React from "react";
import { useEffect, useRef } from "react";

import { useColumnRecs } from "../hooks/useColumnRecs";
import { useLoadToDeck } from "../hooks/useLoadToDeck";
import { useSetPlan, usePlanMutations, usePlanRecs } from "../hooks/useSetPlan";
import { useStartPreview, useStopPreview, usePreviewState } from "../hooks/usePreview";
import { publishZone } from "../lib/gridShape";
import { isFocused, type Zone } from "../lib/gridNav";
import { ROLE_STYLES, roleLabel, type StemRole } from "../lib/roles";
import { store, useAppStore } from "../store";
import type { ColumnRec, PlanItem, PlanRole } from "../types";
import { ScoreBreakdown } from "./ScoreBreakdown";

type Mode = "plan" | "live";

/**
 * Keep the keyboard cursor on screen as it walks a long column.
 *
 * Fires only when focus ARRIVES, not on every render, and instantly rather
 * than smoothly — held arrow keys outrun a smooth scroll, leaving the cursor
 * chasing a viewport that's still animating to where it was two cards ago.
 * Optional-called because not every environment implements it (jsdom).
 */
function useScrollIntoViewWhenFocused(focused: boolean | undefined) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focused) ref.current?.scrollIntoView?.({ block: "nearest" });
  }, [focused]);
  return ref;
}

/** Publish a zone's track ids so keyboard navigation knows the grid's shape.
 *  In an effect (not during render) because it notifies subscribers. */
function usePublishZone(
  role: PlanRole,
  zone: Zone,
  ids: number[],
  /** Skip publishing while the list is in flight. "Unknown" is NOT "empty":
   *  reporting [] mid-refetch makes the cursor think the zone vanished and
   *  relocate — so committing a rec would bounce focus up into the plan and
   *  you'd have to arrow back down for every single pick. */
  loading = false,
): void {
  const key = ids.join(",");
  useEffect(() => {
    if (loading) return;
    publishZone(role, zone, ids);
    // ``key`` is the identity of the list; ``ids`` is a fresh array each
    // render and would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [role, zone, key, loading]);

  // Clear on unmount ONLY — a zone that stops rendering (the plan band with
  // no active set) must not leave ids behind for the cursor to walk into.
  //
  // Deliberately a separate effect keyed on [role, zone]. Folding this into
  // the publish effect above makes its cleanup run on every LIST CHANGE, not
  // just unmount, so each new set of recs was preceded by a transient "this
  // zone is empty" — which relocated the cursor into the plan on every
  // commit, the exact stutter the loop exists to avoid.
  useEffect(() => {
    return () => publishZone(role, zone, []);
  }, [role, zone]);
}

function visRole(role: PlanRole): StemRole {
  return role === "song" ? "mix" : role;
}

function loadKinds(role: PlanRole): string[] | undefined {
  return role === "song" ? undefined : [role];
}

/**
 * One stem card — used for BOTH a queued plan pick and a rec. Always shows
 * ▶ preview. The deck-load buttons (⤒A/⤒B) only appear in **live** (Booth)
 * mode — planning is about queuing, not firing decks. A queued pick gets an
 * emerald left edge (``accent``) so it reads as "yours" vs a suggestion.
 */
function StemRow({
  trackId,
  title,
  artist,
  keyCamelot,
  bpm,
  energy,
  role,
  mode,
  leading,
  breakdown,
  action,
  tooltip,
  accent,
  focused,
  onFocus,
}: {
  trackId: number;
  title: string | null;
  artist: string | null;
  keyCamelot: string | null;
  bpm: number | null;
  energy: number | null;
  role: PlanRole;
  mode: Mode;
  leading: string;
  breakdown?: Record<string, number>;
  action: React.ReactNode;
  tooltip?: string;
  accent?: boolean;
  /** Under the keyboard cursor — draws the focus ring and scrolls into view. */
  focused?: boolean;
  /** Clicking a card moves the keyboard cursor here, so mouse and keyboard
   *  compose instead of fighting over which one owns "the current card". */
  onFocus?: () => void;
}) {
  const styles = ROLE_STYLES[visRole(role)];
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const previewColumn = visRole(role);
  const isPreviewing =
    previewing?.trackId === trackId && previewing?.column === previewColumn;
  const load = useLoadToDeck();
  const cardRef = useScrollIntoViewWhenFocused(focused);
  const togglePreview = () => {
    if (isPreviewing) stopPreview.mutate();
    else startPreview.mutate({ trackId, column: previewColumn });
  };
  return (
    <div
      ref={cardRef}
      onMouseDown={onFocus}
      className={`rounded-md border px-2 py-1.5 text-xs ${
        accent ? "border-l-2 border-l-emerald-500/70 " : ""
      }${
        focused ? "ring-2 ring-amber-300/80 ring-offset-1 ring-offset-neutral-950 " : ""
      }${
        isPreviewing
          ? "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.25)]"
          : `${styles.border} ${styles.bg}`
      }`}
      title={tooltip}
    >
      <div className="flex items-center gap-1.5">
        <span className="font-mono text-[10px] text-neutral-500 tabular-nums shrink-0">
          {leading}
        </span>
        <span className="truncate flex-1 min-w-0 text-neutral-50 text-sm font-medium leading-tight">
          {title ?? `Track #${trackId}`}
        </span>
        {/* Preview + add/remove sit top-right — compact and consistent in both
            plan & live modes. The prominent deck-load (⤒A/⤒B) is a separate
            row below in live mode only. */}
        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={togglePreview}
            disabled={startPreview.isPending || stopPreview.isPending}
            className={`shrink-0 w-7 h-7 text-xs rounded transition-colors disabled:opacity-50 inline-flex items-center justify-center ${
              isPreviewing
                ? "bg-cyan-500/30 hover:bg-cyan-500/40 text-cyan-200 border border-cyan-400/40"
                : "bg-neutral-800 hover:bg-neutral-700 text-neutral-300"
            }`}
            title={isPreviewing ? "Stop preview" : "Preview in headphones (cue)"}
          >
            {isPreviewing ? "⏹" : "▶"}
          </button>
          {action}
        </div>
      </div>
      <div className="text-[11px] text-neutral-400 truncate leading-tight">
        {artist ?? "—"}
      </div>
      <div className="text-[10px] text-neutral-500 truncate font-mono leading-tight">
        {bpm != null && <span>{bpm.toFixed(0)}</span>}
        {keyCamelot && <span className="ml-1.5">· {keyCamelot}</span>}
        {energy != null && <span className="ml-1.5 text-neutral-600">· E{energy}</span>}
      </div>
      {breakdown && Object.keys(breakdown).length > 0 && (
        <ScoreBreakdown breakdown={breakdown} size="sm" className="mt-1" />
      )}
      {mode === "live" && (
        <div className="mt-1 grid grid-cols-2 gap-1">
          <button
            type="button"
            disabled={load.isPending}
            onClick={() => load.mutate({ trackId, role, title, side: "a" })}
            className="h-7 text-xs font-bold rounded bg-violet-700/70 hover:bg-violet-600 text-white transition-colors disabled:opacity-50 inline-flex items-center justify-center gap-1"
            title="Load → Deck A"
          >
            {load.isPending ? "…" : <><span>⤒</span><span>A</span></>}
          </button>
          <button
            type="button"
            disabled={load.isPending}
            onClick={() => load.mutate({ trackId, role, title, side: "b" })}
            className="h-7 text-xs font-bold rounded bg-violet-900/70 hover:bg-violet-800 text-white transition-colors disabled:opacity-50 inline-flex items-center justify-center gap-1"
            title="Load → Deck B"
          >
            {load.isPending ? "…" : <><span>⤒</span><span>B</span></>}
          </button>
        </div>
      )}
    </div>
  );
}

/** A queued pick in the plan (top zone) — emerald-edged, × to remove. */
function QueueCard({
  item,
  role,
  mode,
  index,
  focused,
  onFocus,
  onRemove,
}: {
  item: PlanItem;
  role: PlanRole;
  mode: Mode;
  index: number;
  focused?: boolean;
  onFocus?: () => void;
  onRemove: () => void;
}) {
  return (
    <StemRow
      trackId={item.track_id}
      title={item.title}
      artist={item.artist}
      keyCamelot={item.key_camelot}
      bpm={item.bpm}
      energy={item.floor_energy}
      role={role}
      mode={mode}
      accent
      focused={focused}
      onFocus={onFocus}
      leading={`${index + 1}`}
      action={
        <button
          type="button"
          onClick={onRemove}
          className="shrink-0 w-7 h-7 text-sm rounded bg-neutral-800/70 hover:bg-rose-500/30 text-neutral-400 hover:text-rose-100 border border-neutral-700/60 transition-colors inline-flex items-center justify-center"
          title="Remove from plan"
        >
          ×
        </button>
      }
    />
  );
}

/** A rec candidate (bottom zone) — ＋ to add to the plan. */
function PlanRecCard({
  rec,
  role,
  mode,
  focused,
  onFocus,
  onAdd,
}: {
  rec: ColumnRec;
  role: PlanRole;
  mode: Mode;
  focused?: boolean;
  onFocus?: () => void;
  onAdd?: () => void;
}) {
  return (
    <StemRow
      trackId={rec.track_id}
      title={rec.track_title}
      artist={rec.track_artist}
      keyCamelot={rec.key_camelot}
      bpm={rec.bpm}
      energy={rec.floor_energy}
      role={role}
      mode={mode}
      focused={focused}
      onFocus={onFocus}
      leading={`${Math.round(rec.score * 100)}`}
      breakdown={rec.score_breakdown}
      tooltip={rec.reasons.join(" · ")}
      action={
        onAdd ? (
          <button
            type="button"
            onClick={onAdd}
            className="shrink-0 w-7 h-7 text-sm rounded bg-emerald-700/70 hover:bg-emerald-600 text-white transition-colors inline-flex items-center justify-center"
            title="Add to plan"
          >
            ＋
          </button>
        ) : (
          <span className="w-7" aria-hidden />
        )
      }
    />
  );
}

/** Small uppercase zone label so PLAN vs RECS is unmistakable. */
function ZoneLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-0.5 pt-1 text-[9px] font-bold uppercase tracking-wider text-neutral-600">
      {children}
    </div>
  );
}

/** Recs scored against the plan (Set page). */
function PlanRecsZone({ setId, role }: { setId: number; role: PlanRole }) {
  const recs = usePlanRecs(setId, role);
  const { addToRole } = usePlanMutations(setId);
  return (
    <RecsList
      loading={recs.isLoading}
      recs={recs.data?.recs ?? []}
      role={role}
      mode="plan"
      onAdd={(tid) => addToRole(role, tid)}
    />
  );
}

/**
 * Recs in the Booth.
 *
 * Two regimes, and which one applies is decided by whether there is anything
 * to score against — not by a mode flag:
 *
 * - **Something playing** (or a tempo / trailing plays): re-score against the
 *   live combo. This is the choose-your-adventure stream from docs/vision.md.
 * - **Nothing playing**: score against the *plan* instead. Without this the
 *   scorer has zero computable features, so all five columns return the same
 *   arbitrary tracks at score 0.00 — which is precisely what the Booth showed
 *   on open, before you'd fired anything.
 *
 * With no active set and nothing playing there is genuinely nothing to say,
 * so we say that instead of faking it.
 */
function LiveRecsZone({ setId, role }: { setId: number | null; role: PlanRole }) {
  // ``visRole`` maps the plan role "song" onto the recommender's "mix" feed.
  // Passing the raw role asked the backend for a column it doesn't have, so
  // the SONG column silently showed "no candidates" — the one column a
  // whole-track A/B set is built from.
  const recs = useColumnRecs(visRole(role), { k: 4 });
  const { addToRole } = usePlanMutations(setId);
  const cold = !recs.hasContext;
  const planRecs = usePlanRecs(setId, role, { k: 4, enabled: cold && setId != null });

  if (cold && setId != null) {
    return (
      <RecsList
        loading={planRecs.isLoading}
        recs={planRecs.data?.recs ?? []}
        role={role}
        mode="live"
        onAdd={(tid) => addToRole(role, tid)}
        note="from your plan"
      />
    );
  }
  if (cold) {
    return (
      <div className="px-1 text-[10px] italic leading-snug text-neutral-600">
        nothing playing — load a deck or start a set
      </div>
    );
  }
  return (
    <RecsList
      loading={recs.isLoading}
      recs={recs.data?.recs ?? []}
      role={role}
      mode="live"
      onAdd={setId != null ? (tid) => addToRole(role, tid) : undefined}
    />
  );
}

function RecsList({
  loading,
  recs,
  role,
  mode,
  onAdd,
  note,
}: {
  loading: boolean;
  recs: ColumnRec[];
  role: PlanRole;
  mode: Mode;
  onAdd?: (trackId: number) => void;
  /** Where these came from, when it isn't the obvious one. Keeps the DJ from
   *  wondering why the Booth is showing plan-scored picks. */
  note?: string;
}) {
  const focus = useAppStore((s) => s.gridFocus);
  usePublishZone(role, "recs", recs.map((r) => r.track_id), loading);
  return (
    <div className="flex flex-col gap-1">
      {note && !loading && recs.length > 0 && (
        <div className="px-1 text-[9px] uppercase tracking-wider text-neutral-600">
          {note}
        </div>
      )}
      {loading && <div className="text-[10px] text-neutral-600 px-1">…</div>}
      {!loading && recs.length === 0 && (
        <div className="text-[10px] text-neutral-600 italic px-1">no candidates</div>
      )}
      {recs.map((rec, i) => (
        <PlanRecCard
          key={rec.stem_file_id ?? rec.track_id}
          rec={rec}
          role={role}
          mode={mode}
          focused={isFocused(focus, role, "recs", i)}
          onFocus={() => store.setGridFocus({ role, zone: "recs", index: i })}
          onAdd={onAdd ? () => onAdd(rec.track_id) : undefined}
        />
      ))}
    </div>
  );
}

/**
 * A role column is rendered as three horizontal BANDS across the grid (see
 * RoleColumnsGrid): the header, the PLAN band (your queued picks), and the RECS
 * band (recommendations). Splitting them into bands lets the plan zones share
 * one stretched row — so an empty "nothing queued" lines up with a filled card
 * next to it — while each recs list keeps its own height (a column with fewer
 * recs doesn't leave dead space). It also gives a natural seam between
 * "added" (plan) and "recommended" (recs).
 */

/** The role's colored title cell. */
export function ColumnHeader({ role }: { role: PlanRole }) {
  const styles = ROLE_STYLES[visRole(role)];
  return (
    <div className={`rounded px-2 py-1 text-[10px] font-semibold uppercase text-center border ${styles.header}`}>
      {roleLabel(visRole(role))}
    </div>
  );
}

/** PLAN band for one role — your queued picks (emerald edge), or the empty
 * placeholder. ``flex-1`` lets the zone fill the band's stretched height so
 * every column's plan zone is the same height. */
export function PlanBand({
  setId,
  role,
  mode,
}: {
  setId: number;
  role: PlanRole;
  mode: Mode;
}) {
  const plan = useSetPlan(setId);
  const queue: PlanItem[] = plan.data?.queues?.[role] ?? [];
  return (
    <div className="flex h-full flex-col gap-1 min-w-0">
      <ZoneLabel>◆ Plan {queue.length > 0 && `· ${queue.length}`}</ZoneLabel>
      <PlanZone setId={setId} role={role} mode={mode} queue={queue} className="flex-1" />
    </div>
  );
}

/** RECS band for one role — live (booth) or plan-scored (set) recommendations. */
export function RecsBand({
  setId,
  role,
  mode,
}: {
  setId: number | null;
  role: PlanRole;
  mode: Mode;
}) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <ZoneLabel>Recs</ZoneLabel>
      {mode === "live" ? (
        <LiveRecsZone setId={setId} role={role} />
      ) : setId != null ? (
        <PlanRecsZone setId={setId} role={role} />
      ) : null}
    </div>
  );
}

function PlanZone({
  setId,
  role,
  mode,
  queue,
  className,
}: {
  setId: number;
  role: PlanRole;
  mode: Mode;
  queue: PlanItem[];
  /** Passed ``flex-1`` so the zone grows to fill a stretched plan band —
   * that's what makes the empty placeholder match the tallest queued column. */
  className?: string;
}) {
  const { removeFromRole } = usePlanMutations(setId);
  const focus = useAppStore((s) => s.gridFocus);
  usePublishZone(role, "plan", queue.map((q) => q.track_id));
  if (queue.length === 0) {
    return (
      <div
        className={`flex min-h-[3rem] items-center justify-center rounded-md border border-dashed border-neutral-800 px-2 text-[10px] text-neutral-700 ${className ?? ""}`}
      >
        nothing queued
      </div>
    );
  }
  return (
    <div className={`flex flex-col gap-1 ${className ?? ""}`}>
      {queue.map((item, i) => (
        <QueueCard
          key={`${item.track_id}-${i}`}
          item={item}
          role={role}
          mode={mode}
          index={i}
          focused={isFocused(focus, role, "plan", i)}
          onFocus={() => store.setGridFocus({ role, zone: "plan", index: i })}
          onRemove={() => removeFromRole(role, i)}
        />
      ))}
    </div>
  );
}
