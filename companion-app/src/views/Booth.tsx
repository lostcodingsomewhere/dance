import { BoothColumnHeaders } from "../components/BoothColumnHeaders";
import { Crossfader } from "../components/Crossfader";
import { CueStrip } from "../components/CueStrip";
import { RoleColumnsGrid } from "../components/RoleColumnsGrid";
import { SceneGrid } from "../components/SceneGrid";
import { TwoDeckStrip } from "../components/TwoDeckStrip";
import { useActiveSet } from "../hooks/useSets";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";

/**
 * The Booth — the only screen you should look at during a set.
 *
 * Live-remixing layout (two-deck rethink — see
 * docs/proposals/two-deck-ui-rethink.md):
 *
 *   ┌─ MasterStrip (BPM · KEY · energy arc · session chip · ⌘K · tabs) ─┐
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ TwoDeckStrip (Deck A / Deck B stacked-stem waveforms + per-deck   │
 *   │   header: ◇ mix reference, PFL, FX, sync)                         │
 *   │ Crossfader (A | B — mirrors APC40 hardware fader)                 │
 *   │ BoothColumnHeaders (8 chips: DRUMS A · DRUMS B · BASS A · … in    │
 *   │   APC40 fader order, with Solo "S" buttons)                       │
 *   │ SceneGrid (8-column APC40 mirror — tap cells, tap rows; ▾ expand) │
 *   │ CueStrip (prelisten — headphones out)                             │
 *   │ RoleColumnsGrid (the spine: 5 role columns — drums · bass ·       │
 *   │   vocals · other · song — each with your queued plan picks on top │
 *   │   and live recs below; ⤒A/⤒B load a pick to a deck)              │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Session play count + end-session lives in the MasterStrip's SessionChip
 * (right side). The plan grid IS the set — your queued picks tail what's
 * playing, and live recs (journey-scored off the session so far) hang
 * under each role. No footer — the grid earns the bottom real estate.
 *
 * When a set is active, each role column also shows that set's queued picks
 * on top (the plan); with no active set it's just the live recs. The same
 * RoleColumnsGrid renders in the Set view (mode="plan"), so planning and
 * performing share one surface.
 *
 * Side effects:
 *   - Auto-creates a session on first Ableton play.
 *   - Auto-logs plays as Ableton fires clips loaded via Load-to-Live.
 */
export function Booth() {
  useAutoSession();
  useAutoLog();

  // The active set (if any) backs the plan zone of each role column. Null is
  // fine — the grid then shows only the live recs (no plan zone).
  const setId = useActiveSet().data?.id ?? null;

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <main className="flex-1 flex flex-col min-h-0 gap-2 px-4 py-3 overflow-y-auto">
        {/* Two-deck Traktor-style "now playing" — one stacked-stem
            waveform per deck side. */}
        <TwoDeckStrip />
        {/* Crossfader bar — mirrors APC40 hardware crossfader, also
            drag-to-set on screen. */}
        <Crossfader />
        <BoothColumnHeaders />
        <SceneGrid />
        <CueStrip />
        {/* The plan grid: 5 role columns (drums · bass · vocals · other ·
            song), each with the active set's queued picks on top and live,
            journey-scored recs below. ⤒A/⤒B load a pick into a deck. */}
        <RoleColumnsGrid setId={setId} mode="live" />
      </main>
    </div>
  );
}
