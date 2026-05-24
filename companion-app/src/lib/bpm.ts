/**
 * Shared BPM genre bands. The same anchors drive the MasterStrip's
 * single-value tempo picker (``BpmSlider``) and the Cmd-K range picker
 * (``BpmRangePicker``). Edit freely — these are conventions DJs use to
 * navigate a set, not enforced ranges.
 */

export interface GenreBand {
  from: number;
  to: number;
  label: string;
  /** Tailwind bg class shown in the slider strip when inactive. */
  tint: string;
  /** Tailwind bg class when the picker is currently inside this band. */
  activeTint: string;
}

export const GENRE_BANDS: GenreBand[] = [
  { from: 60,  to: 95,  label: "Chill / Downtempo",    tint: "bg-sky-900/40",    activeTint: "bg-sky-500/60" },
  { from: 95,  to: 110, label: "Hip-Hop",              tint: "bg-purple-900/40", activeTint: "bg-purple-500/60" },
  { from: 110, to: 128, label: "House",                tint: "bg-violet-900/40", activeTint: "bg-violet-500/60" },
  { from: 128, to: 135, label: "Techno / Tech House",  tint: "bg-rose-900/40",   activeTint: "bg-rose-500/60" },
  { from: 135, to: 145, label: "Trance",               tint: "bg-cyan-900/40",   activeTint: "bg-cyan-500/60" },
  { from: 145, to: 180, label: "D&B / Hardstyle",      tint: "bg-amber-900/40",  activeTint: "bg-amber-500/60" },
];

export const BPM_MIN = 60;
export const BPM_MAX = 180;

/** Tick marks for the slider labels. */
export const BPM_TICKS = [60, 80, 100, 120, 140, 160, 180];

export function roundTenth(x: number): number {
  return Math.round(x * 10) / 10;
}
