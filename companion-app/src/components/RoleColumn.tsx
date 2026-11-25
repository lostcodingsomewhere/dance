import type React from "react";
import { useEffect, useRef, useState } from "react";

import { serverMessage } from "../api";
import { useColumnRecs } from "../hooks/useColumnRecs";
import { useLoadToDeck } from "../hooks/useLoadToDeck";
import { useSetPlan, usePlanMutations, usePlanRecs } from "../hooks/useSetPlan";
import { useStartPreview, useStopPreview, usePreviewState } from "../hooks/usePreview";
import { formatDuration } from "../lib/format";
import { publishZone } from "../lib/gridShape";
import { isFocused, type Zone } from "../lib/gridNav";
import { ROLE_STYLES, roleLabel, type DeckSide, type StemRole } from "../lib/roles";
import { store, useAppStore } from "../store";
import { PLAN_ROLES, type ColumnRec, type PlanItem, type PlanRole, type SetPlan } from "../types";
import { ScoreBreakdown } from "./ScoreBreakdown";

type Mode = "plan" | "live";

const UNDO_SECONDS = 8;

interface RemovedPick {
  item: PlanItem;
  index: number;
  planAfterwards: string;
}

function fingerprint(plan: SetPlan | null | undefined): string {
  return JSON.stringify([
    plan?.set_id,
    ...PLAN_ROLES.map((role) => (plan?.queues?.[role] ?? []).map((item) => item.track_id)),
  ]);
}

export function whyNotReady(item: PlanItem): string | null {
  if (item.state === undefined || item.state === "complete") return null;
  if (item.state === null) return "no longer in your library";
  if (item.download_failed) return "download failed";
  if (item.state === "error") return "failed, see Pipeline";
  if (item.state === "pending") return "on its way";
  return "processing";
}

function titleOf(item: PlanItem): string {
  return item.title ?? `Track #${item.track_id}`;
}

function whyNotSaved(error: Error): string {
  const reason = serverMessage(error);
  const serverGaveAReason = reason !== error.message;
  return serverGaveAReason ? `The server said: ${reason}` : "Could not reach the server.";
}

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

function versionNote(rec: ColumnRec): string | undefined {
  if (rec.version_of == null) return undefined;
  const original = rec.version_of_title ?? "a song you picked";
  const length = rec.duration_seconds != null ? ` · ${formatDuration(rec.duration_seconds)}` : "";
  return `${rec.version_label ?? "another cut"} of ${original}${length}`;
}

const LOAD_BUTTON_TONE: Record<DeckSide, string> = {
  a: "bg-violet-700/70 hover:bg-violet-600",
  b: "bg-violet-900/70 hover:bg-violet-800",
};

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
  metaAction,
  tooltip,
  accent,
  versionNote,
  besideOriginalOn,
  notReady,
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
  metaAction?: React.ReactNode;
  tooltip?: string;
  accent?: boolean;
  versionNote?: string;
  besideOriginalOn?: DeckSide;
  notReady?: string | null;
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
      role="group"
      aria-label={title ?? `Track #${trackId}`}
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
            disabled={notReady != null || startPreview.isPending || stopPreview.isPending}
            className={`shrink-0 w-7 h-7 text-xs rounded transition-colors disabled:opacity-50 inline-flex items-center justify-center ${
              isPreviewing
                ? "bg-cyan-500/30 hover:bg-cyan-500/40 text-cyan-200 border border-cyan-400/40"
                : "bg-neutral-800 hover:bg-neutral-700 text-neutral-300"
            }`}
            title={
              notReady != null
                ? "Not ready to preview"
                : isPreviewing
                  ? "Stop preview"
                  : "Preview in headphones (cue)"
            }
          >
            {isPreviewing ? "⏹" : "▶"}
          </button>
          {action}
        </div>
      </div>
      <div className="text-[11px] text-neutral-400 truncate leading-tight">
        {artist ?? "—"}
      </div>
      {versionNote && (
        <div className="flex items-center gap-1 text-[10px] leading-tight text-fuchsia-200/90">
          <span className="shrink-0 rounded bg-fuchsia-500/25 px-1 font-semibold uppercase tracking-wide">
            version
          </span>
          <span className="truncate">{versionNote}</span>
        </div>
      )}
      <div className="flex items-center gap-1.5 text-[10px] text-neutral-500 font-mono leading-tight">
        <span className="truncate flex-1 min-w-0">
          {notReady != null && (
            <span className="mr-1.5 font-sans text-[11px] text-amber-300">{notReady}</span>
          )}
          {bpm != null && <span>{bpm.toFixed(0)}</span>}
          {keyCamelot && <span className="ml-1.5">· {keyCamelot}</span>}
          {energy != null && <span className="ml-1.5 text-neutral-600">· E{energy}</span>}
        </span>
        {metaAction}
      </div>
      {breakdown && Object.keys(breakdown).length > 0 && (
        <ScoreBreakdown breakdown={breakdown} size="sm" className="mt-1" />
      )}
      {mode === "live" && notReady == null && (
        <div className="mt-1 grid grid-cols-2 gap-1">
          {(["a", "b"] as const).map((side) => (
            <button
              key={side}
              type="button"
              disabled={load.isPending}
              onClick={() => load.mutate({ trackId, role, title, side })}
              className={`h-7 text-xs font-bold rounded text-white transition-colors disabled:opacity-50 inline-flex items-center justify-center gap-1 ${
                LOAD_BUTTON_TONE[side]
              } ${side === besideOriginalOn ? "ring-2 ring-fuchsia-300/80" : ""}`}
              title={
                side === besideOriginalOn
                  ? `Load → Deck ${side.toUpperCase()}, the deck beside its original`
                  : `Load → Deck ${side.toUpperCase()}`
              }
            >
              {load.isPending ? "…" : <><span>⤒</span><span>{side.toUpperCase()}</span></>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const FOCUS_RING = "ring-2 ring-amber-300/80 ring-offset-1 ring-offset-neutral-950";
const PREVIEWING_TONE =
  "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_12px_rgba(34,211,238,0.25)]";
const LIVE_BUTTON =
  "inline-flex h-7 w-11 shrink-0 items-center justify-center rounded text-sm transition-colors disabled:opacity-50";
const LIVE_RECS_MIN_HEIGHT = "min-h-[356px]";
// 3 px of padding on every side keeps a row's focus ring inside the scroll box.
const LIVE_PLAN_BOX =
  "-m-[3px] flex max-h-[122px] flex-col gap-1 overflow-y-auto p-[3px] scroll-py-[3px] [@media(max-height:1000px)]:max-h-[82px]";

function usePreviewToggle(trackId: number, role: PlanRole) {
  const startPreview = useStartPreview();
  const stopPreview = useStopPreview();
  const previewing = usePreviewState();
  const column = visRole(role);
  const isPreviewing = previewing?.trackId === trackId && previewing?.column === column;
  return {
    isPreviewing,
    busy: startPreview.isPending || stopPreview.isPending,
    toggle: () => {
      if (isPreviewing) stopPreview.mutate();
      else startPreview.mutate({ trackId, column });
    },
  };
}

function LivePreviewButton({ preview }: { preview: ReturnType<typeof usePreviewToggle> }) {
  return (
    <button
      type="button"
      onClick={preview.toggle}
      disabled={preview.busy}
      className={`${LIVE_BUTTON} ${
        preview.isPreviewing
          ? "border border-cyan-400/40 bg-cyan-500/30 text-cyan-200 hover:bg-cyan-500/40"
          : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
      }`}
      title={preview.isPreviewing ? "Stop preview" : "Preview in headphones (cue)"}
    >
      {preview.isPreviewing ? "⏹" : "▶"}
    </button>
  );
}

function LiveDeckButtons({
  trackId,
  role,
  title,
  besideOriginalOn,
}: {
  trackId: number;
  role: PlanRole;
  title: string | null;
  besideOriginalOn?: DeckSide;
}) {
  const load = useLoadToDeck();
  return (
    <div className="flex shrink-0 items-center gap-1">
      {(["a", "b"] as const).map((side) => (
        <button
          key={side}
          type="button"
          disabled={load.isPending}
          onClick={() => load.mutate({ trackId, role, title, side })}
          className={`${LIVE_BUTTON} gap-0.5 font-bold text-white ${LOAD_BUTTON_TONE[side]} ${
            side === besideOriginalOn ? "ring-2 ring-fuchsia-300/80" : ""
          }`}
          title={
            side === besideOriginalOn
              ? `Load → Deck ${side.toUpperCase()}, the deck beside its original`
              : `Load → Deck ${side.toUpperCase()}`
          }
        >
          {load.isPending ? "…" : <><span>⤒</span><span>{side.toUpperCase()}</span></>}
        </button>
      ))}
    </div>
  );
}

function LivePlanRow({
  item,
  role,
  index,
  focused,
  onFocus,
}: {
  item: PlanItem;
  role: PlanRole;
  index: number;
  focused?: boolean;
  onFocus?: () => void;
}) {
  const styles = ROLE_STYLES[visRole(role)];
  const preview = usePreviewToggle(item.track_id, role);
  const rowRef = useScrollIntoViewWhenFocused(focused);
  const notReady = whyNotReady(item);
  return (
    <div
      ref={rowRef}
      role="group"
      aria-label={titleOf(item)}
      onMouseDown={onFocus}
      className={`flex h-9 shrink-0 items-center gap-2 rounded-md border border-l-2 border-l-emerald-500/70 px-2 ${
        focused ? `${FOCUS_RING} ` : ""
      }${preview.isPreviewing ? PREVIEWING_TONE : `${styles.border} ${styles.bg}`}`}
    >
      <span className="w-5 shrink-0 text-right font-mono text-xs tabular-nums text-neutral-400">
        {index + 1}
      </span>
      <span className="min-w-0 flex-1 truncate text-base font-semibold leading-tight text-neutral-50">
        {titleOf(item)}
      </span>
      {notReady != null ? (
        <span className="shrink-0 text-xs text-amber-300">{notReady}</span>
      ) : (
        <>
          <LivePreviewButton preview={preview} />
          <LiveDeckButtons trackId={item.track_id} role={role} title={item.title} />
        </>
      )}
    </div>
  );
}

function LiveRecCard({
  rec,
  role,
  focused,
  onFocus,
  onAdd,
  besideOriginalOn,
}: {
  rec: ColumnRec;
  role: PlanRole;
  focused?: boolean;
  onFocus?: () => void;
  onAdd?: () => void;
  besideOriginalOn?: DeckSide;
}) {
  const styles = ROLE_STYLES[visRole(role)];
  const preview = usePreviewToggle(rec.track_id, role);
  const cardRef = useScrollIntoViewWhenFocused(focused);
  const title = rec.track_title ?? `Track #${rec.track_id}`;
  const version = versionNote(rec);
  const facts = [
    rec.bpm != null ? rec.bpm.toFixed(0) : null,
    rec.key_camelot,
    rec.floor_energy != null ? `E${rec.floor_energy}` : null,
    rec.track_artist,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div
      ref={cardRef}
      role="group"
      aria-label={title}
      onMouseDown={onFocus}
      title={rec.reasons.join(" · ")}
      className={`flex shrink-0 flex-col gap-0.5 overflow-hidden rounded-md border px-2 py-1 ${
        version ? "h-[108px]" : "h-[86px]"
      } ${focused ? `${FOCUS_RING} ` : ""}${
        preview.isPreviewing ? PREVIEWING_TONE : `${styles.border} ${styles.bg}`
      }`}
    >
      <div className="flex h-[22px] shrink-0 items-center gap-2">
        <span className="shrink-0 font-mono text-xs tabular-nums text-neutral-400">
          {Math.round(rec.score * 100)}
        </span>
        <span className="min-w-0 flex-1 truncate text-base font-semibold leading-tight text-neutral-50">
          {title}
        </span>
      </div>
      {version && (
        <div className="flex h-5 shrink-0 items-center gap-1 text-xs text-fuchsia-200/90">
          <span className="shrink-0 rounded bg-fuchsia-500/25 px-1 font-semibold uppercase tracking-wide">
            version
          </span>
          <span className="min-w-0 truncate">{version}</span>
        </div>
      )}
      <div className="flex h-7 shrink-0 items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[13px] tabular-nums text-neutral-400">
          {facts}
        </span>
        <div className="flex shrink-0 items-center gap-1">
          <LivePreviewButton preview={preview} />
          {onAdd && (
            <button
              type="button"
              onClick={onAdd}
              className={`${LIVE_BUTTON} bg-emerald-700/70 text-white hover:bg-emerald-600`}
              title="Add to plan"
            >
              ＋
            </button>
          )}
        </div>
        <LiveDeckButtons
          trackId={rec.track_id}
          role={role}
          title={rec.track_title}
          besideOriginalOn={besideOriginalOn}
        />
      </div>
      {rec.score_breakdown && <ScoreBreakdown breakdown={rec.score_breakdown} size="line" />}
    </div>
  );
}

// The grid's window key handler would take Enter/Space as commit/audition and cancel the click.
function keepActivationKeysFromGrid(e: React.KeyboardEvent) {
  if (e.key === "Enter" || e.key === " ") e.stopPropagation();
}

function MoveButton({
  direction,
  title,
  onMove,
}: {
  direction: "up" | "down";
  title: string;
  onMove?: () => void;
}) {
  if (!onMove) return <span aria-hidden className="w-5 shrink-0" />;
  return (
    <button
      type="button"
      onClick={onMove}
      onKeyDown={keepActivationKeysFromGrid}
      aria-label={`Move ${title} ${direction}`}
      title={`Move ${direction} in the plan`}
      className="-my-px w-5 shrink-0 rounded border border-neutral-700/60 bg-neutral-800/70 font-sans text-[10px] leading-tight text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 transition-colors"
    >
      {direction === "up" ? "↑" : "↓"}
    </button>
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
  onMoveUp,
  onMoveDown,
  onOpenPads,
  onFindVersions,
}: {
  item: PlanItem;
  role: PlanRole;
  mode: Mode;
  index: number;
  focused?: boolean;
  onFocus?: () => void;
  onRemove: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onOpenPads?: () => void;
  onFindVersions?: () => void;
}) {
  const notReady = whyNotReady(item);
  const metaButton =
    "-my-px shrink-0 rounded border border-neutral-700/60 bg-neutral-800/70 px-1.5 font-sans text-[10px] leading-tight text-neutral-300 hover:bg-neutral-700 hover:text-neutral-100 transition-colors";
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
      notReady={notReady}
      focused={focused}
      onFocus={onFocus}
      leading={`${index + 1}`}
      metaAction={
        <>
          {(onMoveUp || onMoveDown) && (
            <>
              <MoveButton direction="up" title={titleOf(item)} onMove={onMoveUp} />
              <MoveButton direction="down" title={titleOf(item)} onMove={onMoveDown} />
            </>
          )}
          {onFindVersions && (
            <button
              type="button"
              onClick={onFindVersions}
              onKeyDown={keepActivationKeysFromGrid}
              className={metaButton}
              title="Find other versions of this song on Spotify"
            >
              Versions
            </button>
          )}
          {onOpenPads && notReady == null && (
            <button
              type="button"
              onClick={onOpenPads}
              onKeyDown={keepActivationKeysFromGrid}
              className={metaButton}
              title="Show and move this song's cue pads"
            >
              Pads
            </button>
          )}
        </>
      }
      action={
        <button
          type="button"
          onClick={(event) => {
            // The second click of a double-click lands on the pick that moved up into this place.
            if (event.detail > 1) return;
            onRemove();
          }}
          className="ml-3 shrink-0 w-7 h-7 text-sm rounded bg-neutral-800/70 hover:bg-rose-500/30 text-neutral-400 hover:text-rose-100 border border-neutral-700/60 transition-colors inline-flex items-center justify-center"
          title="Remove from plan"
        >
          ×
        </button>
      }
    />
  );
}

function RemovedPickLine({ title, onUndo }: { title: string; onUndo: () => void }) {
  return (
    <div className="flex min-h-[2rem] items-center gap-2 rounded-md border border-dashed border-neutral-700 pl-2 pr-10 text-xs text-neutral-300">
      <span role="status" className="min-w-0 truncate">
        Removed {title}
      </span>
      <button
        type="button"
        onClick={onUndo}
        onKeyDown={keepActivationKeysFromGrid}
        className="shrink-0 rounded border border-neutral-600 bg-neutral-800 px-2 py-0.5 text-xs text-neutral-100 hover:bg-neutral-700 transition-colors"
      >
        Undo
      </button>
    </div>
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
  besideOriginalOn,
}: {
  rec: ColumnRec;
  role: PlanRole;
  mode: Mode;
  focused?: boolean;
  onFocus?: () => void;
  onAdd?: () => void;
  besideOriginalOn?: DeckSide;
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
      versionNote={versionNote(rec)}
      besideOriginalOn={besideOriginalOn}
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
function ZoneLabel({ live, children }: { live?: boolean; children: React.ReactNode }) {
  return (
    <div
      className={`px-0.5 pt-1 font-bold uppercase tracking-wider ${
        live ? "text-xs text-neutral-400" : "text-[9px] text-neutral-600"
      }`}
    >
      {children}
    </div>
  );
}

function ServerNotReached({ consequence, onRetry }: { consequence?: string; onRetry: () => void }) {
  return (
    <div
      role="alert"
      className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-dashed border-rose-500/40 px-2 py-1.5 text-xs leading-snug text-rose-200"
    >
      <span>{["Could not reach the server.", consequence].filter(Boolean).join(" ")}</span>
      <button
        type="button"
        onClick={onRetry}
        onKeyDown={keepActivationKeysFromGrid}
        className="shrink-0 rounded border border-rose-400/40 bg-neutral-900 px-2 py-0.5 text-xs text-neutral-100 hover:bg-neutral-800 transition-colors"
      >
        Retry
      </button>
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
      failed={recs.isError && recs.data == null}
      onRetry={() => recs.refetch()}
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
 * - **Something playing** (or trailing plays, or a tempo with no planned pick
 *   to score against): re-score against the live combo. This is the
 *   choose-your-adventure stream from docs/vision.md.
 * - **Nothing playing**: score against the *plan* instead. Without this the
 *   scorer has zero computable features, so all five columns return the same
 *   arbitrary tracks at score 0.00 — which is precisely what the Booth showed
 *   on open, before you'd fired anything.
 *
 * With no active set and nothing playing there is genuinely nothing to say,
 * so we say that instead of faking it.
 */
function LiveRecsZone({ setId, role }: { setId: number | null; role: PlanRole }) {
  const plan = useSetPlan(setId);
  // ``visRole`` maps the plan role "song" onto the recommender's "mix" feed.
  // Passing the raw role asked the backend for a column it doesn't have, so
  // the SONG column silently showed "no candidates" — the one column a
  // whole-track A/B set is built from.
  const recs = useColumnRecs(visRole(role), { k: 4, planHasPicks: hasPicks(plan.data) });
  const { addToRole } = usePlanMutations(setId);
  const cold = !recs.hasContext;
  const planRecs = usePlanRecs(setId, role, { k: 4, enabled: cold && setId != null });
  const deckBesideTheOriginal = (rec: ColumnRec): DeckSide | undefined => {
    const original = rec.version_of == null ? undefined : recs.deckOfSong.get(rec.version_of);
    if (original == null) return undefined;
    return original === "a" ? "b" : "a";
  };

  if (cold && setId != null) {
    return (
      <RecsList
        loading={planRecs.isLoading}
        failed={planRecs.isError && planRecs.data == null}
        onRetry={() => planRecs.refetch()}
        recs={planRecs.data?.recs ?? []}
        role={role}
        mode="live"
        onAdd={(tid) => addToRole(role, tid)}
        note="from your plan"
        deckBesideTheOriginal={deckBesideTheOriginal}
      />
    );
  }
  if (cold) {
    return (
      <>
        <ZoneLabel live>Recs</ZoneLabel>
        <div className={`px-1 text-xs italic leading-snug text-neutral-500 ${LIVE_RECS_MIN_HEIGHT}`}>
          nothing playing — load a deck or start a set
        </div>
      </>
    );
  }
  return (
    <RecsList
      loading={recs.isLoading}
      failed={recs.isError && recs.data == null}
      onRetry={() => recs.refetch()}
      recs={recs.data?.recs ?? []}
      role={role}
      mode="live"
      onAdd={setId != null ? (tid) => addToRole(role, tid) : undefined}
      deckBesideTheOriginal={deckBesideTheOriginal}
    />
  );
}

function RecsList({
  loading,
  failed,
  onRetry,
  recs,
  role,
  mode,
  onAdd,
  note,
  deckBesideTheOriginal,
}: {
  loading: boolean;
  failed: boolean;
  onRetry: () => void;
  recs: ColumnRec[];
  role: PlanRole;
  mode: Mode;
  onAdd?: (trackId: number) => void;
  deckBesideTheOriginal?: (rec: ColumnRec) => DeckSide | undefined;
  /** Where these came from, when it isn't the obvious one. Keeps the DJ from
   *  wondering why the Booth is showing plan-scored picks. */
  note?: string;
}) {
  const focus = useAppStore((s) => s.gridFocus);
  usePublishZone(role, "recs", recs.map((r) => r.track_id), loading);
  const live = mode === "live";
  const showNote = note != null && !loading && recs.length > 0;
  const quiet = live ? "px-1 text-xs text-neutral-500" : "text-[10px] text-neutral-600 px-1";
  return (
    <>
      {live && (
        <ZoneLabel live>
          Recs{showNote && <span className="ml-2 font-normal">{note}</span>}
        </ZoneLabel>
      )}
      <div className={`flex flex-col gap-1 ${live ? LIVE_RECS_MIN_HEIGHT : ""}`}>
        {!live && showNote && (
          <div className="px-1 text-[9px] uppercase tracking-wider text-neutral-600">
            {note}
          </div>
        )}
        {loading && <div className={quiet}>…</div>}
        {!loading && failed && <ServerNotReached onRetry={onRetry} />}
        {!loading && !failed && recs.length === 0 && (
          <div className={`${quiet} italic`}>no candidates</div>
        )}
        {recs.map((rec, i) => {
          const card = {
            rec,
            role,
            focused: isFocused(focus, role, "recs", i),
            onFocus: () => store.setGridFocus({ role, zone: "recs", index: i }),
            onAdd: onAdd ? () => onAdd(rec.track_id) : undefined,
            besideOriginalOn: deckBesideTheOriginal?.(rec),
          };
          const key = rec.stem_file_id ?? rec.track_id;
          return live ? (
            <LiveRecCard key={key} {...card} />
          ) : (
            <PlanRecCard key={key} {...card} mode={mode} />
          );
        })}
      </div>
    </>
  );
}

/**
 * A role column is rendered as three horizontal BANDS across the grid (see
 * RoleColumnsGrid): the header, the PLAN band (your queued picks), and the RECS
 * band (recommendations). In the Set view, splitting them into bands lets the
 * plan zones share one stretched row — so an empty "nothing queued" lines up
 * with a filled card next to it — while each recs list keeps its own height (a
 * column with fewer recs doesn't leave dead space). In the Booth each plan zone
 * is a box of one-line rows that scrolls inside itself, so the recs below stay
 * on screen however long the plan is. The bands also give a natural seam
 * between "added" (plan) and "recommended" (recs).
 */

/** The role's colored title cell. */
export function ColumnHeader({
  role,
  mode,
  setId,
}: {
  role: PlanRole;
  mode?: Mode;
  setId?: number | null;
}) {
  const styles = ROLE_STYLES[visRole(role)];
  const plan = useSetPlan(mode === "live" ? setId : null);
  const planned = plan.data?.queues?.[role]?.length ?? 0;
  return (
    <div
      className={`rounded px-2 py-1 font-semibold uppercase text-center border ${
        mode === "live" ? "text-xs" : "text-[10px]"
      } ${styles.header}`}
    >
      {roleLabel(visRole(role))}
      {mode === "live" && planned > 0 && (
        <span className="ml-1.5 font-normal normal-case opacity-80">{planned} planned</span>
      )}
    </div>
  );
}

/** PLAN band for one role — your queued picks (emerald edge), or the empty
 * placeholder. In the Set view ``flex-1`` lets the zone fill the band's
 * stretched height so every column's plan zone is the same height. */
export function PlanBand({
  setId,
  role,
  mode,
  onOpenPads,
  onFindVersions,
}: {
  setId: number;
  role: PlanRole;
  mode: Mode;
  onOpenPads?: (item: PlanItem) => void;
  onFindVersions?: (item: PlanItem) => void;
}) {
  const plan = useSetPlan(setId);
  const queue: PlanItem[] = plan.data?.queues?.[role] ?? [];
  const live = mode === "live";
  return (
    <div
      role="group"
      aria-label={`${roleLabel(visRole(role))} plan`}
      className={`flex flex-col gap-1 min-w-0 ${live ? "" : "h-full"}`}
    >
      {!live && <ZoneLabel>◆ Plan {queue.length > 0 && `· ${queue.length}`}</ZoneLabel>}
      {live ? (
        <LivePlanZone setId={setId} role={role} />
      ) : (
        <PlanZone
          setId={setId}
          role={role}
          mode={mode}
          onOpenPads={onOpenPads}
          onFindVersions={onFindVersions}
          className="flex-1"
        />
      )}
    </div>
  );
}

function hasPicks(plan: SetPlan | null | undefined): boolean {
  return Object.values(plan?.queues ?? {}).some((queue) => queue.length > 0);
}

export function usePlanBandShown(setId: number | null, mode: Mode): boolean {
  const plan = useSetPlan(setId);
  if (setId == null) return false;
  if (mode === "plan") return true;
  const couldNotLoad = plan.isError && plan.data == null;
  return couldNotLoad || hasPicks(plan.data);
}

export function PlanRow({
  setId,
  className,
  children,
}: {
  setId: number;
  className: string;
  children: React.ReactNode;
}) {
  const plan = useSetPlan(setId);
  if (plan.isError && plan.data == null) {
    return (
      <ServerNotReached consequence="Your plan is not shown." onRetry={() => plan.refetch()} />
    );
  }
  return <div className={className}>{children}</div>;
}

export function RecsRow({
  setId,
  mode,
  className,
  children,
}: {
  setId: number | null;
  mode: Mode;
  className: string;
  children: React.ReactNode;
}) {
  const plan = useSetPlan(setId);
  if (mode === "plan" && plan.isLoading) return null;
  if (mode === "plan" && plan.data != null && !hasPicks(plan.data)) {
    return (
      <p className="px-2 py-6 text-center text-sm leading-relaxed text-neutral-300">
        Nothing planned yet. Press ⌘K and add a first song. Recommendations follow from what you
        plan.
      </p>
    );
  }
  return <div className={className}>{children}</div>;
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
    <div
      role="group"
      aria-label={`${roleLabel(visRole(role))} recs`}
      className="flex flex-col gap-1 min-w-0"
    >
      {mode === "live" ? (
        <LiveRecsZone setId={setId} role={role} />
      ) : (
        <>
          <ZoneLabel>Recs</ZoneLabel>
          {setId != null ? <PlanRecsZone setId={setId} role={role} /> : null}
        </>
      )}
    </div>
  );
}

function LivePlanZone({ setId, role }: { setId: number; role: PlanRole }) {
  const focus = useAppStore((s) => s.gridFocus);
  const plan = useSetPlan(setId);
  const queue: PlanItem[] = plan.data?.queues?.[role] ?? [];
  usePublishZone(role, "plan", queue.map((q) => q.track_id));
  if (queue.length === 0) {
    return (
      <div className="flex h-9 items-center px-2 text-xs text-neutral-500">nothing queued</div>
    );
  }
  return (
    <div data-plan-scroll-box className={LIVE_PLAN_BOX}>
      {queue.map((item, i) => (
        <LivePlanRow
          key={`${item.track_id}-${i}`}
          item={item}
          role={role}
          index={i}
          focused={isFocused(focus, role, "plan", i)}
          onFocus={() => store.setGridFocus({ role, zone: "plan", index: i })}
        />
      ))}
    </div>
  );
}

function PlanZone({
  setId,
  role,
  mode,
  onOpenPads,
  onFindVersions,
  className,
}: {
  setId: number;
  role: PlanRole;
  mode: Mode;
  onOpenPads?: (item: PlanItem) => void;
  onFindVersions?: (item: PlanItem) => void;
  /** Passed ``flex-1`` so the zone grows to fill a stretched plan band —
   * that's what makes the empty placeholder match the tallest queued column. */
  className?: string;
}) {
  const { put, removeFromRole, restoreToRole, moveInRole } = usePlanMutations(setId);
  const focus = useAppStore((s) => s.gridFocus);
  const [removed, setRemoved] = useState<RemovedPick | null>(null);
  // Read here, not passed down: a prop would still hold the plan from before the save.
  const plan = useSetPlan(setId);
  const queue: PlanItem[] = plan.data?.queues?.[role] ?? [];
  const planNow = fingerprint(plan.data);
  usePublishZone(role, "plan", queue.map((q) => q.track_id));

  useEffect(() => {
    if (removed == null) return;
    const forget = () => setRemoved((current) => (current === removed ? null : current));
    if (planNow !== removed.planAfterwards) {
      forget();
      return;
    }
    const timer = window.setTimeout(forget, UNDO_SECONDS * 1000);
    return () => window.clearTimeout(timer);
  }, [removed, planNow]);

  // A failed save stops mattering once the plan changes.
  useEffect(() => {
    if (put.isError) put.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [planNow]);

  const remove = (item: PlanItem, index: number) =>
    removeFromRole(role, index, item.track_id, (saved) =>
      setRemoved({ item, index, planAfterwards: fingerprint(saved) }),
    );
  const move = (from: number, to: number) =>
    moveInRole(role, from, to, () => {
      if (isFocused(store.peekGridFocus(), role, "plan", from)) {
        store.setGridFocus({ role, zone: "plan", index: to });
      }
    });
  const canMove = mode === "plan";

  const rows = queue.map((item, i) => (
    <QueueCard
      key={`${item.track_id}-${i}`}
      item={item}
      role={role}
      mode={mode}
      index={i}
      focused={isFocused(focus, role, "plan", i)}
      onFocus={() => store.setGridFocus({ role, zone: "plan", index: i })}
      onRemove={() => remove(item, i)}
      onMoveUp={canMove && i > 0 ? () => move(i, i - 1) : undefined}
      onMoveDown={canMove && i < queue.length - 1 ? () => move(i, i + 1) : undefined}
      onOpenPads={onOpenPads && (() => onOpenPads(item))}
      onFindVersions={onFindVersions && (() => onFindVersions(item))}
    />
  ));
  if (removed != null && planNow === removed.planAfterwards) {
    rows.splice(
      Math.min(removed.index, rows.length),
      0,
      <RemovedPickLine
        key="removed"
        title={titleOf(removed.item)}
        onUndo={() =>
          restoreToRole(role, removed.index, removed.item.track_id, () => setRemoved(null))
        }
      />,
    );
  }
  return (
    <div className={`flex flex-col gap-1 ${className ?? ""}`}>
      {put.isError && (
        <p
          role="alert"
          className="rounded-md border border-dashed border-rose-500/40 px-2 py-1 text-xs leading-snug text-rose-200"
        >
          Change not saved. {whyNotSaved(put.error)}
        </p>
      )}
      {rows}
      {queue.length === 0 && (
        <div className="flex min-h-[3rem] flex-1 items-center justify-center rounded-md border border-dashed border-neutral-800 px-2 text-[10px] text-neutral-700">
          {plan.isLoading ? "…" : "nothing queued"}
        </div>
      )}
    </div>
  );
}
