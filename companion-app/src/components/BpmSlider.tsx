import { useEffect, useRef, useState } from "react";

/**
 * Genre BPM anchors. Bands are approximate but match the conventions DJs
 * use to navigate a set ("we're in tech house territory", "drop into D&B").
 * Edit freely — these are display-only hints, not enforced ranges.
 */
const GENRE_BANDS: { from: number; to: number; label: string }[] = [
  { from: 60,  to: 90,  label: "Chill" },
  { from: 90,  to: 105, label: "Hip-Hop" },
  { from: 105, to: 118, label: "Slow Hse" },
  { from: 118, to: 128, label: "House" },
  { from: 128, to: 135, label: "Techno" },
  { from: 135, to: 145, label: "Trance" },
  { from: 145, to: 175, label: "D&B" },
];

const TICKS = [60, 80, 100, 120, 140, 160, 180];

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

  // Outside-click → close (cancels, since we only commit on explicit release).
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) onClose();
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      else if (e.key === "Enter") {
        onCommit(roundTenth(draft));
        onClose();
      }
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [draft, onClose, onCommit]);

  function commitOnRelease() {
    onCommit(roundTenth(draft));
  }

  // % position of current draft along the slider track (used to color the
  // genre band the thumb is currently sitting on).
  const draftPct = ((draft - min) / (max - min)) * 100;
  const activeBand = GENRE_BANDS.find((b) => draft >= b.from && draft < b.to);

  return (
    <div
      ref={containerRef}
      className="absolute top-full left-2 mt-2 z-50 w-[420px] rounded-lg border border-amber-500/30 bg-neutral-950/95 backdrop-blur shadow-2xl px-4 pt-3 pb-3"
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
        <span className="text-[10px] uppercase tracking-widest text-amber-300/70">
          {activeBand?.label ?? "—"}
        </span>
      </div>

      {/* Slider */}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={0.1}
          value={draft}
          autoFocus
          onChange={(e) => setDraft(parseFloat(e.target.value))}
          onMouseUp={commitOnRelease}
          onTouchEnd={commitOnRelease}
          onKeyUp={(e) => {
            if (["ArrowLeft", "ArrowRight"].includes(e.key)) commitOnRelease();
          }}
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

      {/* Genre bands */}
      <div className="flex items-stretch mt-1 h-4 gap-px rounded overflow-hidden">
        {GENRE_BANDS.map((b) => {
          const widthPct = ((b.to - b.from) / (max - min)) * 100;
          const isActive = activeBand && b.from === activeBand.from;
          return (
            <div
              key={b.label}
              className={`flex items-center justify-center text-[9px] uppercase tracking-tight truncate transition-colors ${
                isActive
                  ? "bg-amber-500/30 text-amber-100 font-semibold"
                  : "bg-neutral-900 text-neutral-500"
              }`}
              style={{ width: `${widthPct}%` }}
              title={`${b.label} — ${b.from}–${b.to} BPM`}
            >
              {b.label}
            </div>
          );
        })}
      </div>

      {/* Hint */}
      <div className="mt-2 text-[10px] text-neutral-500 flex items-center justify-between">
        <span>Drag · ←/→ · Enter to commit</span>
        <span className="text-neutral-600">Esc to cancel</span>
      </div>
    </div>
  );
}

function roundTenth(x: number): number {
  return Math.round(x * 10) / 10;
}
