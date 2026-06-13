/**
 * Stem-role display labels. Backend canonical names (StemFile.kind +
 * deck-column keys) are ``drums / bass / vocals / other / mix`` — Demucs
 * outputs and DB conventions. The UI relabels the two opaque names:
 *
 *   other → Melody   (synths, leads, pads, guitars — everything Demucs
 *                     didn't classify into drums/bass/vocals)
 *   mix   → Song     (the original full-track recording; anchor mode)
 *
 * Use ``roleLabel(kind)`` everywhere user-facing instead of dumping the
 * raw key. If we ever rename the backend kinds, only this file changes.
 */

export const STEM_COLUMNS = [
  "drums",
  "bass",
  "vocals",
  "other",
  "mix",
] as const;

export type StemRole = (typeof STEM_COLUMNS)[number];

const LABELS: Record<string, string> = {
  drums: "Drums",
  bass: "Bass",
  vocals: "Vocals",
  other: "Melody",
  mix: "Song",
};

export function roleLabel(kind: string): string {
  return LABELS[kind] ?? kind;
}

/** Two-deck UI column order. Exactly 8 columns in APC40 mk2 fader order
 * (strip 1 → drums A, strip 2 → drums B, … strip 8 → other B), interleaved
 * by role so A/B for the same role sit adjacent. The mix/anchor reference
 * is NOT a grid column — it lives as a per-deck header chip in each
 * TwoDeckStrip panel. See docs/proposals/two-deck-ui-rethink.md. */
export const TWO_DECK_COLUMN_ORDER: readonly string[] = [
  "drums_a",
  "drums_b",
  "bass_a",
  "bass_b",
  "vocals_a",
  "vocals_b",
  "other_a",
  "other_b",
];

/** Backend deck-kind → user-facing column header label. Includes the
 * A/B side suffix so the eye groups same-role pairs visually. */
const DECK_LABELS: Record<string, string> = {
  drums_a: "Drums A",
  drums_b: "Drums B",
  bass_a: "Bass A",
  bass_b: "Bass B",
  vocals_a: "Vocals A",
  vocals_b: "Vocals B",
  other_a: "Melody A",
  other_b: "Melody B",
  mix_a: "Song A",
  mix_b: "Song B",
};

export function deckColumnLabel(deckKind: string): string {
  return DECK_LABELS[deckKind] ?? deckKind;
}

/** Stem columns excluding the "song" anchor column — for "loadable stems"
 * iterations like banner load buttons. */
export const STEM_ONLY_COLUMNS: readonly StemRole[] = [
  "drums",
  "bass",
  "vocals",
  "other",
];

/** Deck-pair side: cells in Live now live on either the A or B deck per
 * source role, so the DJ can pre-stage the next song without nuking
 * what's playing. ``null`` for mix (single deck). */
export type DeckSide = "a" | "b";

/** Map a backend DeckCell.kind ("drums_a", "drums_b", "mix") back to
 * its source role ("drums", "mix"). The deck-pair reshape introduced
 * the _a/_b suffix; the frontend's mental model is still 5 columns by
 * role, so we strip the suffix at the boundary. */
export function sourceKindOf(deckKind: string): StemRole {
  if (deckKind.endsWith("_a") || deckKind.endsWith("_b")) {
    const src = deckKind.slice(0, -2);
    if (
      src === "drums" || src === "bass" || src === "vocals" || src === "other"
    ) {
      return src;
    }
  }
  if (
    deckKind === "drums" || deckKind === "bass" || deckKind === "vocals"
    || deckKind === "other" || deckKind === "mix"
  ) {
    return deckKind;
  }
  return "mix"; // fallback — keeps the type concrete
}

/** Map a backend DeckCell.kind to its A/B side, or ``null`` for mix. */
export function sideOf(deckKind: string): DeckSide | null {
  if (deckKind.endsWith("_a")) return "a";
  if (deckKind.endsWith("_b")) return "b";
  return null;
}

/** For a source role + side, build the backend deck-kind string. */
export function deckKindOf(role: StemRole, side: DeckSide | null): string {
  if (role === "mix") return "mix";
  return side ? `${role}_${side}` : role;
}

/**
 * Per-column visual tokens used across the Booth's stacked strips
 * (column headers · combo cards · scene cells · rec cards). Same tint
 * carries vertically down each column so the booth reads as 5 colored
 * stripes top-to-bottom instead of three independently-styled strips.
 *
 * - ``header``: bolder bg + border + text for the top column-header chip
 * - ``bg``:     subtle bg for cards/cells *not* in the header row
 * - ``border``: tinted border for cards/cells
 * - ``text``:   chip-style accent text (key chip, etc.)
 */
export const ROLE_STYLES: Record<
  StemRole,
  { header: string; bg: string; border: string; text: string }
> = {
  drums: {
    header: "bg-red-500/15 border-red-500/35 text-red-200",
    bg: "bg-red-500/[0.06]",
    border: "border-red-500/25",
    text: "text-red-300",
  },
  bass: {
    header: "bg-amber-500/15 border-amber-500/35 text-amber-200",
    bg: "bg-amber-500/[0.06]",
    border: "border-amber-500/25",
    text: "text-amber-300",
  },
  vocals: {
    header: "bg-lime-500/15 border-lime-500/35 text-lime-200",
    bg: "bg-lime-500/[0.06]",
    border: "border-lime-500/25",
    text: "text-lime-300",
  },
  other: {
    header: "bg-sky-500/15 border-sky-500/35 text-sky-200",
    bg: "bg-sky-500/[0.06]",
    border: "border-sky-500/25",
    text: "text-sky-300",
  },
  mix: {
    header: "bg-neutral-700/35 border-neutral-500/40 text-neutral-100",
    bg: "bg-neutral-700/[0.10]",
    border: "border-neutral-500/30",
    text: "text-neutral-200",
  },
};
