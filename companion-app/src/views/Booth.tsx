import { BoothColumnHeaders } from "../components/BoothColumnHeaders";
import { ColumnRecBanner } from "../components/ColumnRecBanner";
import { ComboStrip } from "../components/ComboStrip";
import { CueStrip } from "../components/CueStrip";
import { PlayedStrip } from "../components/PlayedStrip";
import { SceneGrid } from "../components/SceneGrid";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";

const STEM_COLUMNS = ["drums", "bass", "vocals", "other", "mix"];

/**
 * The Booth — the only screen you should look at during a set.
 *
 * Live-remixing layout (no song-mode artifacts):
 *
 *   ┌─ MasterStrip (BPM · KEY · energy arc · OSC heartbeat · view tabs) ┐
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
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ PlayedStrip (set name · plays · history scroll · end set)        │
 *   └─────────────────────────────────────────────────────────────────┘
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

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <main className="flex-1 flex flex-col min-h-0 gap-3 px-4 py-3 overflow-y-auto">
        <BoothColumnHeaders />
        <ComboStrip />
        <div>
          <div className="text-[10px] uppercase tracking-widest text-neutral-500 mb-1.5">
            Scene grid · tap to fire / stop (mirrors APC40)
          </div>
          <SceneGrid />
        </div>
        <CueStrip />
        <div>
          <div className="text-[10px] uppercase tracking-widest text-neutral-500 mb-1.5">
            Next per column · live re-scored against the combo
          </div>
          <div className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1">
            {/* Leading spacer mirrors SceneGrid's row-label column so
                the 5 rec banners line up under the 5 stem columns. */}
            <div aria-hidden="true" />
            {STEM_COLUMNS.map((c) => (
              <ColumnRecBanner key={c} column={c} k={3} />
            ))}
          </div>
        </div>
      </main>
      <PlayedStrip />
    </div>
  );
}
