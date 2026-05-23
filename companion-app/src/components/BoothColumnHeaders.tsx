import { useState } from "react";
import { useDeckMap } from "../hooks/useDeckMap";
import { useSoloTrack } from "../hooks/useTransport";
import { STEM_COLUMNS, roleLabel, type StemRole } from "../lib/roles";
import { RoleIcon } from "./RoleIcon";

/** Per-column accent — same colors used elsewhere in the Booth so the
 * header tint matches the rec-banner header tint matches the SceneGrid
 * cell border tint. One source of truth for "the drums column is red." */
const ROLE_ACCENT: Record<StemRole, string> = {
  drums:  "bg-red-500/15 border-red-500/35 text-red-200",
  bass:   "bg-amber-500/15 border-amber-500/35 text-amber-200",
  vocals: "bg-lime-500/15 border-lime-500/35 text-lime-200",
  other:  "bg-sky-500/15 border-sky-500/35 text-sky-200",
  mix:    "bg-neutral-700/35 border-neutral-500/40 text-neutral-100",
};

/**
 * The single column header strip at the top of the Booth — labels each
 * of the 5 stem columns once, with a distinct background tint so the
 * column identity carries down through ComboStrip → SceneGrid → recs
 * below.
 *
 * Hosts the per-column Solo (S) button which routes that one stem
 * through Live's Cue / PFL bus → Scarlett 4i4 outs 3/4 → headphones,
 * leaving master untouched. Solo state lives here (was previously
 * inside SceneGrid).
 *
 * Layout matches the ``[2.5rem leading + 5 stem cols] gap-1`` template
 * used by SceneGrid / ComboStrip / rec-banner so headers line up
 * vertically with the strips below.
 */
export function BoothColumnHeaders() {
  const deckMap = useDeckMap();
  const soloTrack = useSoloTrack();
  const columns = deckMap.data?.columns ?? null;
  const [soloedRoles, setSoloedRoles] = useState<Set<string>>(new Set());

  function toggleSolo(role: StemRole, trackIdx: number) {
    const wasSoloed = soloedRoles.has(role);
    const next = new Set(soloedRoles);
    if (wasSoloed) next.delete(role);
    else next.add(role);
    setSoloedRoles(next);
    soloTrack.mutate({ track: trackIdx, soloed: !wasSoloed });
  }

  if (!columns) return null;

  return (
    <div
      className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1"
      data-testid="booth-column-headers"
    >
      <div aria-hidden="true" />
      {STEM_COLUMNS.map((role) => {
        const trackIdx = columns[role];
        const isSoloed = soloedRoles.has(role);
        const accent = ROLE_ACCENT[role];
        return (
          <div
            key={role}
            className={`flex items-center gap-1.5 px-2 py-1.5 rounded-md border text-[11px] uppercase tracking-widest font-semibold ${accent}`}
          >
            <RoleIcon role={role} size={14} />
            <span className="flex-1">{roleLabel(role)}</span>
            {trackIdx != null && (
              <button
                type="button"
                onClick={() => toggleSolo(role, trackIdx)}
                title={
                  isSoloed
                    ? `Unsolo ${roleLabel(role).toLowerCase()} — return to normal routing`
                    : `Solo ${roleLabel(role).toLowerCase()} → cue out (Live's solo button must be in Cue/PFL mode)`
                }
                className={`text-[10px] leading-none rounded px-1.5 py-0.5 transition-colors ${
                  isSoloed
                    ? "bg-amber-400/30 text-amber-100 border border-amber-300/60"
                    : "text-neutral-500 hover:text-amber-200 border border-transparent"
                }`}
                aria-label={isSoloed ? `unsolo ${role}` : `solo ${role}`}
                aria-pressed={isSoloed}
              >
                S
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
