import { useMemo, useRef, useState } from "react";
import type { Region } from "../types";

const SECTION_COLOR: Record<string, string> = {
  intro: "#52525b",
  buildup: "#eab308",
  drop: "#ef4444",
  breakdown: "#06b6d4",
  bridge: "#a855f7",
  outro: "#52525b",
  verse: "#3b82f6",
  chorus: "#f59e0b",
  other: "#737373",
};

const CUE_COLOR = "#fbbf24";

interface Props {
  regions: Region[] | undefined;
  durationSeconds: number | null;
  bpm: number | null;
  /** Live's current beat position (from AbletonState.beat). */
  beat: number | null;
}

type Hover =
  | { kind: "section"; region: Region; x: number }
  | { kind: "cue"; region: Region; x: number }
  | null;

/**
 * Horizontal map of a track's musical structure with a playhead, a "next
 * section in N bars" callout, and a hover-driven tooltip that surfaces
 * section name + start time + length. Snappy by design: CSS-only opacity
 * + scale transitions, no measurement loops, no layout thrash.
 */
export function StructureTimeline({ regions, durationSeconds, bpm, beat }: Props) {
  const totalMs = (durationSeconds ?? 0) * 1000;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<Hover>(null);

  const sections = useMemo(
    () => (regions ?? []).filter((r) => r.region_type === "section"),
    [regions],
  );
  const cues = useMemo(
    () => (regions ?? []).filter((r) => r.region_type === "cue"),
    [regions],
  );

  const playheadMs = useMemo(() => {
    if (beat == null || bpm == null || bpm <= 0) return null;
    const ms = (beat / bpm) * 60_000;
    if (totalMs > 0) return ms % totalMs;
    return ms;
  }, [beat, bpm, totalMs]);

  const nextSection = useMemo(() => {
    if (playheadMs == null || sections.length === 0) return null;
    return (
      sections.find((s) => s.position_ms > playheadMs) ?? sections[0]
    );
  }, [sections, playheadMs]);

  const beatsToNext = useMemo(() => {
    if (nextSection == null || playheadMs == null || bpm == null) return null;
    const deltaMs = nextSection.position_ms - playheadMs;
    if (deltaMs <= 0) return null;
    return Math.round((deltaMs / 60_000) * bpm);
  }, [nextSection, playheadMs, bpm]);

  if (totalMs <= 0 && sections.length === 0 && cues.length === 0) {
    return (
      <div className="h-20 rounded-lg border border-dashed border-neutral-800 flex items-center justify-center text-neutral-600 text-sm">
        No region map for this track yet.
      </div>
    );
  }

  const pct = (ms: number) => (totalMs > 0 ? (ms / totalMs) * 100 : 0);

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-neutral-500 uppercase text-xs tracking-widest">
          Structure
        </div>
        {nextSection && beatsToNext != null && beatsToNext > 0 && (
          <div className="text-sm">
            <span className="text-neutral-500">Next:</span>{" "}
            <span
              className="font-semibold uppercase tracking-wide"
              style={{
                color:
                  SECTION_COLOR[nextSection.section_label ?? "other"] ??
                  "#e5e5e5",
              }}
            >
              {nextSection.section_label ?? "section"}
            </span>{" "}
            <span className="text-neutral-300 font-mono tabular-nums">
              in {beatsToNext} beats
              {beatsToNext >= 4 && (
                <span className="text-neutral-500">
                  {" "}
                  · {Math.round(beatsToNext / 4)} bars
                </span>
              )}
            </span>
          </div>
        )}
      </div>

      {/* Outer wrapper is positioning context for the floating tooltip;
          inner bar clips the bands/playhead so they can't escape. We need
          the tooltip OUTSIDE the clipped region or it gets cut off below. */}
      <div
        ref={containerRef}
        className="relative"
        onMouseLeave={() => setHover(null)}
      >
        <div className="relative h-16 rounded-lg border border-neutral-800 bg-neutral-900 overflow-hidden group">
        {/* section bands */}
        {sections.map((s) => {
          const width = s.length_ms != null ? pct(s.length_ms) : null;
          const color =
            SECTION_COLOR[s.section_label ?? "other"] ?? "#525252";
          const isHovered = hover?.kind === "section" && hover.region.id === s.id;
          const isDimmed =
            hover?.kind === "section" && hover.region.id !== s.id;
          return (
            <button
              type="button"
              key={s.id}
              onMouseEnter={(e) => {
                const rect = containerRef.current?.getBoundingClientRect();
                const x = rect
                  ? e.clientX - rect.left
                  : pct(s.position_ms) * 6.4;
                setHover({ kind: "section", region: s, x });
              }}
              className="absolute top-0 bottom-0 flex items-center justify-center text-[10px] uppercase tracking-wider font-semibold transition-all duration-100 ease-out cursor-default focus:outline-none"
              style={{
                left: `${pct(s.position_ms)}%`,
                width: width != null ? `${width}%` : "auto",
                backgroundColor: color + (isHovered ? "AA" : isDimmed ? "33" : "55"),
                borderLeft: `2px solid ${color}`,
                boxShadow: isHovered
                  ? `inset 0 0 0 1px ${color}, 0 2px 8px rgba(0,0,0,0.4)`
                  : "none",
                color: "#fafafa",
                opacity: isDimmed ? 0.6 : 1,
                zIndex: isHovered ? 2 : 1,
              }}
            >
              <span className="px-1 truncate">{s.section_label}</span>
            </button>
          );
        })}

        {/* cue markers — thicker hit area for hover, thin visual */}
        {cues.map((c) => {
          const isHovered = hover?.kind === "cue" && hover.region.id === c.id;
          return (
            <button
              type="button"
              key={c.id}
              onMouseEnter={(e) => {
                const rect = containerRef.current?.getBoundingClientRect();
                const x = rect ? e.clientX - rect.left : pct(c.position_ms) * 6.4;
                setHover({ kind: "cue", region: c, x });
              }}
              className="absolute top-0 bottom-0 -translate-x-1/2 cursor-default focus:outline-none"
              style={{
                left: `${pct(c.position_ms)}%`,
                width: 8, // hit target
                zIndex: 3,
              }}
              aria-label={`Cue: ${c.name ?? "cue"}`}
            >
              <span
                className="block h-full transition-all duration-100 ease-out mx-auto"
                style={{
                  width: isHovered ? 2 : 1,
                  backgroundColor: CUE_COLOR,
                  boxShadow: isHovered ? `0 0 6px ${CUE_COLOR}` : "none",
                }}
              />
            </button>
          );
        })}

        {/* playhead — sits above sections (z=4) and pulses subtly */}
        {playheadMs != null && totalMs > 0 && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white pointer-events-none shadow-[0_0_8px_rgba(255,255,255,0.8)]"
            style={{ left: `${pct(playheadMs)}%`, zIndex: 4 }}
          />
        )}
        </div>

        {/* tooltip overlay — sibling of the clipped bar so it can extend below */}
        {hover && (
          <HoverCard
            hover={hover}
            totalMs={totalMs}
            bpm={bpm}
          />
        )}
      </div>

      {totalMs > 0 && (
        <div className="flex justify-between text-[10px] text-neutral-600 font-mono mt-1">
          <span>0:00</span>
          <span>{formatMs(totalMs / 2)}</span>
          <span>{formatMs(totalMs)}</span>
        </div>
      )}
    </div>
  );
}

function HoverCard({
  hover,
  totalMs,
  bpm,
}: {
  hover: NonNullable<Hover>;
  totalMs: number;
  bpm: number | null;
}) {
  const r = hover.region;
  const startMs = r.position_ms;
  const startLabel = formatMs(startMs);
  const lengthMs = r.length_ms ?? null;
  const lengthLabel = lengthMs != null ? formatMs(lengthMs) : null;
  const lengthBars = r.length_bars ?? (
    lengthMs != null && bpm != null
      ? Math.round(((lengthMs / 60_000) * bpm) / 4)
      : null
  );
  const colorKey = r.section_label ?? r.region_type ?? "other";
  const color = SECTION_COLOR[colorKey] ?? "#a3a3a3";
  // Clamp the tooltip so it never spills off the right edge. Estimate width
  // ~ 180px; the container is full-width of its parent so we use viewport-ish
  // math via percentage of x within the band's offset.
  const offsetClass = "translate-x-2"; // small offset so the pointer doesn't sit ON the tip
  return (
    <div
      className="absolute top-full mt-1 left-0 z-50 pointer-events-none"
      style={{ transform: `translateX(${Math.max(0, hover.x - 90)}px)` }}
    >
      <div
        className={`rounded-md border bg-neutral-950/95 backdrop-blur px-2.5 py-1.5 shadow-lg text-xs ${offsetClass}`}
        style={{ borderColor: color + "88" }}
      >
        <div className="flex items-center gap-1.5">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span
            className="font-semibold uppercase tracking-wider"
            style={{ color }}
          >
            {hover.kind === "section"
              ? r.section_label ?? "section"
              : r.name ?? "cue"}
          </span>
        </div>
        <div className="text-neutral-400 font-mono tabular-nums mt-0.5">
          @ {startLabel}
          {lengthLabel && totalMs > 0 && (
            <>
              {" · "}
              <span className="text-neutral-500">length</span> {lengthLabel}
              {lengthBars != null && lengthBars > 0 && (
                <span className="text-neutral-600"> · {lengthBars} bars</span>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function formatMs(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}
