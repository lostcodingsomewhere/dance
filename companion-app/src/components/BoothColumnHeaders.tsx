import { useState } from "react";
import { useDeckMap } from "../hooks/useDeckMap";
import { useSoloTrack } from "../hooks/useTransport";
import {
  ROLE_STYLES,
  TWO_DECK_COLUMN_ORDER,
  deckColumnLabel,
  roleLabel,
  sideOf,
  sourceKindOf,
  type DeckSide,
  type StemRole,
} from "../lib/roles";
import { RoleIcon } from "./RoleIcon";

/**
 * Column header strip atop the Booth. With the two-deck layout (see
 * docs/proposals/two-deck-ui-rethink.md) there are now 10 columns: 4
 * source roles × A/B sides + 2 mix references. Songs sit center; the
 * grid mirrors around them.
 *
 * Headers carry the per-source-role color (A/B share a hue so the eye
 * groups them as one role); the A/B badge inside the chip distinguishes
 * sides. Solo (S) button per header still cues that one deck's audio
 * through Live's Solo/PFL bus to the headphone outs.
 *
 * Drag-to-reorder is removed for now — the canonical mirror layout
 * (drums A | bass A | vocals A | other A | song A | song B | other B
 * | vocals B | bass B | drums B) carries meaning, so free reordering
 * would break the spatial mental model.
 */
export function BoothColumnHeaders() {
  const deckMap = useDeckMap();
  const soloTrack = useSoloTrack();
  const columns = deckMap.data?.columns ?? null;
  const [soloed, setSoloed] = useState<Set<string>>(new Set());

  function toggleSolo(deckKind: string, trackIdx: number) {
    const wasSoloed = soloed.has(deckKind);
    const next = new Set(soloed);
    if (wasSoloed) next.delete(deckKind);
    else next.add(deckKind);
    setSoloed(next);
    soloTrack.mutate({ track: trackIdx, soloed: !wasSoloed });
  }

  if (!columns) return null;

  return (
    <div
      className="grid grid-cols-[2rem_repeat(10,minmax(0,1fr))] gap-1"
      data-testid="booth-column-headers"
    >
      <div aria-hidden="true" />
      {TWO_DECK_COLUMN_ORDER.map((deckKind) => (
        <ColumnHeaderChip
          key={deckKind}
          deckKind={deckKind}
          trackIdx={columns[deckKind]}
          isSoloed={soloed.has(deckKind)}
          onSoloToggle={() => {
            const idx = columns[deckKind];
            if (idx != null) toggleSolo(deckKind, idx);
          }}
        />
      ))}
    </div>
  );
}

function ColumnHeaderChip({
  deckKind,
  trackIdx,
  isSoloed,
  onSoloToggle,
}: {
  deckKind: string;
  trackIdx: number | undefined;
  isSoloed: boolean;
  onSoloToggle: () => void;
}) {
  const role = sourceKindOf(deckKind);
  const side: DeckSide | null = sideOf(deckKind);
  const sideBadge = side?.toUpperCase() ?? "";
  // B-side decks get a subtler header treatment — slightly dimmed so
  // A/B are visually distinguishable even though they share the role
  // color and the same role label.
  const styles = ROLE_STYLES[role as StemRole];
  // Strip the A/B suffix from the visible label — the deck container
  // above identifies which side this is. We keep the suffix in the
  // tooltip + aria for accessibility / debugging.
  const visibleLabel = roleLabel(role);
  return (
    <div
      title={`${deckColumnLabel(deckKind)} — routes to crossfader ${sideBadge}`}
      className={`flex items-center gap-1 px-1.5 py-1 rounded-md border text-[10px] uppercase tracking-wider font-semibold transition-all ${
        styles.header
      } ${side === "b" ? "opacity-75" : ""}`}
    >
      <RoleIcon role={role} size={12} />
      <span className="flex-1 truncate text-[10px]">{visibleLabel}</span>
      {trackIdx != null && (
        <button
          type="button"
          onClick={onSoloToggle}
          title={
            isSoloed
              ? `Unsolo ${deckColumnLabel(deckKind)}`
              : `Solo ${deckColumnLabel(deckKind)} (Live's solo button must be in Cue/PFL mode)`
          }
          className={`text-[9px] leading-none rounded px-1 py-0.5 transition-colors ${
            isSoloed
              ? "bg-amber-400/30 text-amber-100 border border-amber-300/60"
              : "text-neutral-500 hover:text-amber-200 border border-transparent"
          }`}
          aria-label={isSoloed ? `unsolo ${deckKind}` : `solo ${deckKind}`}
          aria-pressed={isSoloed}
        >
          S
        </button>
      )}
    </div>
  );
}
