/**
 * Two-handle BPM range picker — Cmd-K filter chip.
 *
 * Same genre-band visualization as the MasterStrip's single-value
 * <BpmSlider/> (both pull from ``lib/bpm``), but with two handles so the
 * user can pick a [min, max] range. Click any genre band to snap the
 * range to its [from, to] — that's the "select techno section" gesture.
 * Click the active band again to deselect (clear filter).
 *
 * The min/max numeric inputs stay for fine-grained tuning; the visual
 * picker is the fast path for "I want roughly techno-territory candidates."
 */

import { useEffect, useRef, useState } from "react";
import {
  BPM_MAX,
  BPM_MIN,
  BPM_TICKS,
  GENRE_BANDS,
  roundTenth,
} from "../lib/bpm";

interface Props {
  /** Current min — null means "no lower bound." */
  min: number | null;
  /** Current max — null means "no upper bound." */
  max: number | null;
  onChange: (next: { min: number | null; max: number | null }) => void;
}

export function BpmRangePicker({ min, max, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // Close on outside click / Esc.
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (!wrapperRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const isActive = min != null || max != null;
  const activeBand = matchingGenreBand(min, max);
  const triggerLabel = isActive
    ? activeBand
      ? activeBand.label
      : `${min ?? "—"}–${max ?? "—"} BPM`
    : "BPM range";

  function clear() {
    onChange({ min: null, max: null });
  }

  function selectBand(band: { from: number; to: number }) {
    if (activeBand && activeBand.from === band.from) {
      // Re-click the active band → deselect.
      clear();
    } else {
      onChange({ min: band.from, max: band.to });
    }
  }

  return (
    <div ref={wrapperRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`text-[11px] px-2 py-1 rounded border transition-colors ${
          isActive
            ? "bg-amber-500/15 text-amber-200 border-amber-500/40 hover:bg-amber-500/20"
            : "bg-neutral-900 text-neutral-400 border-neutral-800 hover:border-neutral-700"
        }`}
        title="Filter results by BPM range — click a genre band to select"
      >
        <span aria-hidden className="mr-1 text-[9px]">♪</span>
        {triggerLabel}
        {isActive && (
          <span
            role="button"
            aria-label="clear BPM range"
            onClick={(e) => {
              e.stopPropagation();
              clear();
            }}
            className="ml-1 text-neutral-500 hover:text-rose-300 cursor-pointer"
          >
            ×
          </span>
        )}
      </button>
      {open && (
        <RangePopover
          min={min}
          max={max}
          onChange={onChange}
          onSelectBand={selectBand}
          activeBand={activeBand}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  );
}

function RangePopover({
  min,
  max,
  onChange,
  onSelectBand,
  activeBand,
  onClose,
}: {
  min: number | null;
  max: number | null;
  onChange: (next: { min: number | null; max: number | null }) => void;
  onSelectBand: (band: { from: number; to: number }) => void;
  activeBand: { from: number; to: number; label: string } | null;
  onClose: () => void;
}) {
  // Drafts so dragging is smooth — commit on release / blur.
  const [draftMin, setDraftMin] = useState<number>(min ?? BPM_MIN);
  const [draftMax, setDraftMax] = useState<number>(max ?? BPM_MAX);

  // Sync drafts when external props change (e.g. via band snap).
  useEffect(() => {
    setDraftMin(min ?? BPM_MIN);
    setDraftMax(max ?? BPM_MAX);
  }, [min, max]);

  function commit() {
    const lo = Math.min(draftMin, draftMax);
    const hi = Math.max(draftMin, draftMax);
    // Full-range = no filter.
    if (lo <= BPM_MIN && hi >= BPM_MAX) {
      onChange({ min: null, max: null });
    } else {
      onChange({ min: roundTenth(lo), max: roundTenth(hi) });
    }
  }

  const lo = Math.min(draftMin, draftMax);
  const hi = Math.max(draftMin, draftMax);
  const minPct = ((lo - BPM_MIN) / (BPM_MAX - BPM_MIN)) * 100;
  const maxPct = ((hi - BPM_MIN) / (BPM_MAX - BPM_MIN)) * 100;

  return (
    <div
      className="absolute left-0 top-full mt-1 z-50 w-[420px] rounded-md border border-neutral-700 bg-neutral-950/95 backdrop-blur shadow-2xl px-3 pt-3 pb-3"
      role="dialog"
      aria-label="BPM range"
    >
      {/* Header — live values + active band name */}
      <div className="flex items-baseline justify-between mb-2">
        <div className="flex items-baseline gap-1.5 font-mono">
          <span className="text-lg text-amber-200 tabular-nums leading-none">
            {lo.toFixed(0)}
          </span>
          <span className="text-neutral-500 text-xs">–</span>
          <span className="text-lg text-amber-200 tabular-nums leading-none">
            {hi.toFixed(0)}
          </span>
          <span className="text-[9px] uppercase tracking-widest text-neutral-500 ml-1">
            BPM
          </span>
        </div>
        <span className="text-[10px] uppercase tracking-widest text-amber-300 font-semibold">
          {activeBand?.label ?? "—"}
        </span>
      </div>

      {/* Dual range slider — two native inputs stacked. Custom thumbs sit
          on top so the user can grab either handle independently. */}
      <div className="relative h-6">
        {/* Active range fill */}
        <div className="absolute top-1/2 -translate-y-1/2 h-1 left-0 right-0 bg-neutral-800 rounded-full" />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-1 bg-amber-500/60 rounded-full"
          style={{ left: `${minPct}%`, right: `${100 - maxPct}%` }}
        />
        <input
          type="range"
          min={BPM_MIN}
          max={BPM_MAX}
          step={1}
          value={draftMin}
          onChange={(e) => setDraftMin(parseFloat(e.target.value))}
          onMouseUp={commit}
          onKeyUp={commit}
          onTouchEnd={commit}
          aria-label="BPM minimum"
          className="absolute inset-0 w-full appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-moz-range-thumb]:pointer-events-auto accent-amber-400"
        />
        <input
          type="range"
          min={BPM_MIN}
          max={BPM_MAX}
          step={1}
          value={draftMax}
          onChange={(e) => setDraftMax(parseFloat(e.target.value))}
          onMouseUp={commit}
          onKeyUp={commit}
          onTouchEnd={commit}
          aria-label="BPM maximum"
          className="absolute inset-0 w-full appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-moz-range-thumb]:pointer-events-auto accent-amber-400"
        />
      </div>

      {/* Tick labels */}
      <div className="flex justify-between text-[9px] text-neutral-500 font-mono mt-1 tabular-nums">
        {BPM_TICKS.map((t) => (
          <span
            key={t}
            className={lo <= t && t <= hi ? "text-amber-300" : ""}
          >
            {t}
          </span>
        ))}
      </div>

      {/* Genre band strip — click any band to snap range to it. Re-click
          the active band to clear (deselect). */}
      <div className="flex items-stretch mt-1 h-4 gap-px rounded overflow-hidden">
        {GENRE_BANDS.map((b) => {
          const widthPct = ((b.to - b.from) / (BPM_MAX - BPM_MIN)) * 100;
          const isActive = activeBand?.from === b.from;
          return (
            <button
              key={b.label}
              type="button"
              onClick={() => onSelectBand(b)}
              title={
                isActive
                  ? `${b.label} — click to deselect`
                  : `${b.label} — ${b.from}–${b.to} BPM`
              }
              className={`transition-colors hover:brightness-150 cursor-pointer ${
                isActive ? b.activeTint : b.tint
              }`}
              style={{ width: `${widthPct}%` }}
              aria-label={
                isActive
                  ? `Deselect ${b.label}`
                  : `Select ${b.label} range`
              }
            />
          );
        })}
      </div>

      {/* Footer — numeric tune + actions */}
      <div className="mt-2.5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1 text-[11px]">
          <input
            type="number"
            value={draftMin}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v)) setDraftMin(v);
            }}
            onBlur={commit}
            className="w-14 bg-neutral-900 border border-neutral-800 rounded px-1.5 py-0.5 text-neutral-100 outline-none focus:border-neutral-700 font-mono tabular-nums"
            aria-label="BPM min"
          />
          <span className="text-neutral-600">to</span>
          <input
            type="number"
            value={draftMax}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v)) setDraftMax(v);
            }}
            onBlur={commit}
            className="w-14 bg-neutral-900 border border-neutral-800 rounded px-1.5 py-0.5 text-neutral-100 outline-none focus:border-neutral-700 font-mono tabular-nums"
            aria-label="BPM max"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => {
              onChange({ min: null, max: null });
              setDraftMin(BPM_MIN);
              setDraftMax(BPM_MAX);
            }}
            className="text-[10px] text-neutral-500 hover:text-neutral-200 uppercase tracking-wider px-2 py-1"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={onClose}
            className="text-[10px] text-neutral-300 hover:text-neutral-100 uppercase tracking-wider px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

function matchingGenreBand(
  min: number | null,
  max: number | null,
): { from: number; to: number; label: string } | null {
  if (min == null || max == null) return null;
  for (const b of GENRE_BANDS) {
    if (Math.abs(min - b.from) < 0.5 && Math.abs(max - b.to) < 0.5) {
      return b;
    }
  }
  return null;
}
