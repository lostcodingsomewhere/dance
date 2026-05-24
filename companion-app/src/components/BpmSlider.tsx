import { useEffect, useRef, useState } from "react";
import { BPM_TICKS as TICKS, GENRE_BANDS, roundTenth } from "../lib/bpm";

interface BpmSliderProps {
  value: number;
  min?: number;
  max?: number;
  /** Fires once on slider release (or Enter). */
  onCommit: (bpm: number) => void;
  /** Fires when the popover should close (commit, Escape, outside click). */
  onClose: () => void;
}

/**
 * Click-to-edit BPM popover for MasterStrip. Drag the slider for coarse
 * adjustment with genre BPM anchors below the track; the central number
 * tracks live. Release (or Enter) commits to Ableton via /tempo; Escape
 * or outside-click cancels.
 *
 * The slider snaps to 0.1 BPM steps so fine adjustments survive the trip
 * through ``setTempo``.
 */
export function BpmSlider({
  value,
  min = 60,
  max = 180,
  onCommit,
  onClose,
}: BpmSliderProps) {
  const [draft, setDraft] = useState(value);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Outside-click and Escape both discard the draft. Commit only happens
  // on explicit Apply / Enter so a glancing drag can't change the master
  // tempo mid-set.
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) onClose();
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      else if (e.key === "Enter") {
        commit();
      }
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [draft, onClose]);

  function commit() {
    onCommit(roundTenth(draft));
    onClose();
  }

  // % position of current draft along the slider track (used to color the
  // genre band the thumb is currently sitting on).
  const draftPct = ((draft - min) / (max - min)) * 100;
  const activeBand = GENRE_BANDS.find((b) => draft >= b.from && draft < b.to);
  const isDirty = roundTenth(draft) !== roundTenth(value);

  return (
    <div
      ref={containerRef}
      className="absolute top-full left-2 mt-2 z-50 w-[480px] rounded-lg border border-amber-500/30 bg-neutral-950/95 backdrop-blur shadow-2xl px-4 pt-3 pb-3"
      data-testid="bpm-slider"
    >
      {/* Live value + active genre band */}
      <div className="flex items-baseline justify-between mb-2">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-3xl text-amber-200 tabular-nums leading-none">
            {draft.toFixed(1)}
          </span>
          <span className="text-[10px] uppercase tracking-widest text-neutral-500">
            BPM
          </span>
        </div>
        <span className="text-xs uppercase tracking-widest text-amber-300 font-semibold">
          {activeBand?.label ?? "—"}
        </span>
      </div>

      {/* Slider — drag only updates draft, no commit until Apply / Enter. */}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={0.1}
          value={draft}
          autoFocus
          onChange={(e) => setDraft(parseFloat(e.target.value))}
          className="w-full appearance-none h-2 rounded-full bg-neutral-800 outline-none accent-amber-400 cursor-pointer"
          aria-label="Master BPM"
        />
        {/* Thumb position marker — purely visual, sits behind the input thumb */}
        <div
          className="absolute -bottom-1 w-px h-3 bg-amber-300/30 pointer-events-none"
          style={{ left: `calc(${draftPct}% )` }}
        />
      </div>

      {/* Tick labels */}
      <div className="flex justify-between text-[9px] text-neutral-500 font-mono mt-1 tabular-nums">
        {TICKS.map((t) => (
          <span key={t} className={draft >= t - 1 && draft <= t + 1 ? "text-amber-300" : ""}>
            {t}
          </span>
        ))}
      </div>

      {/* Thin colored band strip — clickable: tapping a band snaps the draft
          to the band's center BPM. Each band gets a distinct color tint;
          the active one brightens. Labels live elsewhere (active band name
          at the top-right) to keep the strip itself uncluttered. */}
      <div className="flex items-stretch mt-1 h-3 gap-px rounded overflow-hidden">
        {GENRE_BANDS.map((b) => {
          const widthPct = ((b.to - b.from) / (max - min)) * 100;
          const isActive = activeBand && b.from === activeBand.from;
          const center = roundTenth((b.from + b.to) / 2);
          return (
            <button
              key={b.label}
              type="button"
              onClick={() => setDraft(center)}
              title={`${b.label} — ${b.from}–${b.to} BPM (click to snap to ${center})`}
              className={`transition-colors hover:brightness-150 cursor-pointer ${
                isActive ? b.activeTint : b.tint
              }`}
              style={{ width: `${widthPct}%` }}
              aria-label={`Set BPM to ${center} (${b.label})`}
            />
          );
        })}
      </div>

      {/* Footer — explicit commit. Drag and band-clicks only update the
          draft; Apply / Enter is the only way the tempo actually changes. */}
      <div className="mt-3 flex items-center justify-between gap-2">
        <span className="text-[10px] text-neutral-500">
          {isDirty ? (
            <>
              <span className="text-amber-400">●</span> uncommitted ·
              was {value.toFixed(1)}
            </>
          ) : (
            <>Drag · click bands · ←/→</>
          )}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onClose}
            className="h-7 px-2.5 rounded text-[11px] text-neutral-400 hover:text-neutral-200 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={commit}
            disabled={!isDirty}
            className={`h-7 px-3 rounded text-[11px] font-semibold transition-colors ${
              isDirty
                ? "bg-amber-500/80 hover:bg-amber-500 text-neutral-950"
                : "bg-neutral-800 text-neutral-600 cursor-not-allowed"
            }`}
            title={isDirty ? "Apply this BPM to Ableton" : "No change to apply"}
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}

