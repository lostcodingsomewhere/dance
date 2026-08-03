import { useAbletonState } from "../hooks/useAbletonState";
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
 * docs/proposals/two-deck-ui-rethink.md) there are 8 columns: 4 source
 * roles × A/B sides, in APC40 mk2 fader order (strip 1 → drums A, strip
 * 2 → drums B, …). The mix/anchor reference is NOT a column here — it
 * lives as a per-deck chip inside each TwoDeckStrip panel header.
 *
 * Headers carry the per-source-role color (A/B share a hue so the eye
 * groups them as one role); the A/B badge inside the chip distinguishes
 * sides (A-side bright, B-side muted). An optional 1–8 fader-number badge
 * makes the screen→hand mapping explicit. Solo (S) button per header
 * cues that one deck's audio through Live's Solo/PFL bus to the headphone
 * outs — its lit state reads from the contract's ``soloed_kinds`` so it
 * can never contradict Live or the per-deck PFL toggles.
 *
 * Drag-to-reorder is removed for now — the canonical fader-order layout
 * carries meaning, so free reordering would break the spatial mental
 * model and the 1:1 APC40 strip mapping.
 */
export function BoothColumnHeaders() {
  const deckMap = useDeckMap();
  const soloTrack = useSoloTrack();
  const soloedKinds = useAbletonState().soloed_kinds;
  const columns = deckMap.data?.columns ?? null;
  // Source of truth for "is it lit": Live's actual solo bus, surfaced via
  // the WS contract. Clicking still sends the solo command (optimistic),
  // but the lit state is driven from here so the header "S" buttons and
  // the per-deck PFL toggles can't drift apart.
  const soloed = new Set(soloedKinds);

  function toggleSolo(deckKind: string, trackIdx: number) {
    const wasSoloed = soloed.has(deckKind);
    soloTrack.mutate({ track: trackIdx, soloed: !wasSoloed });
  }

  if (!columns) return null;

  return (
    <div
      className="grid grid-cols-8 gap-1"
      data-testid="booth-column-headers"
    >
      {TWO_DECK_COLUMN_ORDER.map((deckKind) => (
        <ColumnHeaderChip
          key={deckKind}
          deckKind={deckKind}
          // Derive the fader number from the ABLETON TRACK INDEX, not from
          // this column's position on screen.
          //
          // The APC40 mk2's eight channel strips address Live's first eight
          // tracks. This used to print `i + 1` — the on-screen order — which
          // asserts a hardware mapping the app cannot guarantee: deck columns
          // are appended wherever Live's track count happens to be, so if
          // anything precedes them (the user's own tracks, a different Set)
          // the badge confidently names a strip that controls something else.
          // Undefined here renders no badge rather than a wrong one.
          faderNumber={
            columns[deckKind] != null && columns[deckKind] < 8
              ? columns[deckKind] + 1
              : undefined
          }
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
  faderNumber,
  trackIdx,
  isSoloed,
  onSoloToggle,
}: {
  deckKind: string;
  /** APC40 fader strip number (1–8), or undefined when this column is not
   *  within the APC40's reach — in which case no badge is shown at all. */
  faderNumber: number | undefined;
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
  // Role label + A/B side, e.g. "DRUMS A". With the 8-column fader-order
  // layout the two columns for one role sit adjacent, so the side suffix
  // is what tells them apart at a glance.
  const visibleLabel = roleLabel(role);
  return (
    <div
      title={
        faderNumber != null
          ? `${deckColumnLabel(deckKind)} · APC40 fader ${faderNumber} — routes to crossfader ${sideBadge}`
          : `${deckColumnLabel(deckKind)} · Ableton track ${trackIdx ?? "?"} — outside the APC40's 8-strip window at its default position. Press Bank Select ▶ to bring it into reach (the faders and the pad grid move together).`
      }
      className={`flex items-center gap-1 px-1.5 py-1 rounded-md border text-[10px] uppercase tracking-wider font-semibold transition-all ${
        styles.header
      } ${side === "b" ? "opacity-75" : ""}`}
    >
      {/* Fader-number badge — the APC40 strip that really controls this
          column. Absent when the column sits outside the first eight Ableton
          tracks, because then no strip does. */}
      <span
        className={`text-[8px] font-mono tabular-nums shrink-0 leading-none ${
          faderNumber != null ? "text-neutral-500/80" : "text-amber-500/80"
        }`}
        data-testid={`fader-badge-${deckKind}`}
      >
        {faderNumber ?? "—"}
      </span>
      <RoleIcon role={role} size={12} />
      <span className="flex-1 truncate text-[10px]">
        {visibleLabel}
        {sideBadge && (
          <span className="ml-0.5 text-[9px] opacity-80">{sideBadge}</span>
        )}
      </span>
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
