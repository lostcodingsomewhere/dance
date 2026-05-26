import { BoothColumnHeaders } from "../components/BoothColumnHeaders";
import { ColumnRecBanner } from "../components/ColumnRecBanner";
import { CueStrip } from "../components/CueStrip";
import { SceneGrid } from "../components/SceneGrid";
import { TwoDeckStrip } from "../components/TwoDeckStrip";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";

/**
 * The Booth — the only screen you should look at during a set.
 *
 * Live-remixing layout:
 *
 *   ┌─ MasterStrip (BPM · KEY · energy arc · session chip · ⌘K · tabs) ─┐
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ BoothColumnHeaders (5 colored chips: DRUMS · BASS · VOCALS ·    │
 *   │   MELODY · SONG with per-column Solo "S" buttons)                │
 *   │                                                                  │
 *   │ ComboStrip (5 cards: track identity + scrubbable waveform per    │
 *   │             role, anchor hint, click-to-snap-to-section)         │
 *   │                                                                  │
 *   │ 3×5 SceneGrid (canonical APC40 mirror — tap cells, tap rows;     │
 *   │              ▾ expand to 8 rows)                                 │
 *   │                                                                  │
 *   │ CueStrip (prelisten — same waveform features, headphones out)    │
 *   │                                                                  │
 *   │ Per-column rec banners (5 across)                                │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Session play count + end-session lives in the MasterStrip's SessionChip
 * (right side). The Set Rail (⌘\) covers planning + tail-recs. No footer
 * — the SceneGrid is the only thing that should be earning the
 * bottom-of-screen real estate.
 *
 * ColumnHeaders → ComboStrip → SceneGrid → recs all share the same
 * ``[2.5rem leading + 5 stem cols] gap-1`` grid template, so the 5
 * stem columns line up vertically and the labels at the top govern
 * everything beneath them.
 *
 * Side effects:
 *   - Auto-creates a session on first Ableton play.
 *   - Auto-logs plays as Ableton fires clips loaded via Load-to-Live.
 */
export function Booth() {
  useAutoSession();
  useAutoLog();

  // Source roles for the rec banner — 4 stem feeds + 1 song feed.
  // Recs are role-scoped (not side-scoped); each card's ⤒A/⤒B buttons
  // pick the deck at load time.
  const recRoles = ["drums", "bass", "vocals", "other", "mix"];

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <main className="flex-1 flex flex-col min-h-0 gap-2 px-4 py-3 overflow-y-auto">
        {/* Two-deck Traktor-style "now playing" — replaces ComboStrip.
            One stacked-stem waveform per deck side. */}
        <TwoDeckStrip />
        <BoothColumnHeaders />
        <SceneGrid />
        <CueStrip />
        {/* Recs banner: 5 source-role feeds. Each card spans 2 of the
            10 grid cols below so they sit above their A/B pair. */}
        <div className="grid grid-cols-[2rem_repeat(10,minmax(0,1fr))] gap-1">
          <div aria-hidden="true" />
          {recRoles.map((c) => (
            <div key={c} className="col-span-2">
              <ColumnRecBanner column={c} k={5} />
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
