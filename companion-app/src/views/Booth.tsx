import { ColumnRecBanner } from "../components/ColumnRecBanner";
import { ComboStrip } from "../components/ComboStrip";
import { CueStrip } from "../components/CueStrip";
import { MasterVisualizer } from "../components/MasterVisualizer";
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
 *   │ ComboStrip (5 cards: what's playing per role, source-tracked)    │
 *   │                                                                  │
 *   │ Per-column rec banners (5 across)                                │
 *   │                                                                  │
 *   │ 8×5 SceneGrid (canonical APC40 mirror — tap cells, tap rows)     │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │ PlayedStrip (set name · plays · history scroll · end set)        │
 *   └─────────────────────────────────────────────────────────────────┘
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
        <div>
          <div className="text-[10px] uppercase tracking-widest text-neutral-500 mb-1.5">
            Scene grid · tap to fire / stop (mirrors APC40)
          </div>
          <SceneGrid />
        </div>
        <MasterVisualizer />
        <ComboStrip />
        <CueStrip />
        <div>
          <div className="text-[10px] uppercase tracking-widest text-neutral-500 mb-1.5">
            Next per column · live re-scored against the combo
          </div>
          <div className="grid grid-cols-5 gap-1.5">
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
