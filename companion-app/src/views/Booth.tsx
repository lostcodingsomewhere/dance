import { BoothColumnHeaders } from "../components/BoothColumnHeaders";
import { ColumnRecBanner } from "../components/ColumnRecBanner";
import { ComboStrip } from "../components/ComboStrip";
import { CueStrip } from "../components/CueStrip";
import { SceneGrid } from "../components/SceneGrid";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";
import { useAppStore } from "../store";

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
  // User-customizable column order — drag a column header in the booth to
  // reorder. Defaults to the canonical drums/bass/vocals/other/mix layout.
  const stemColumns = useAppStore((s) => s.stemColumnOrder);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <main className="flex-1 flex flex-col min-h-0 gap-2 px-4 py-3 overflow-y-auto">
        <BoothColumnHeaders />
        <ComboStrip />
        <SceneGrid />
        <CueStrip />
        <div className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1">
          {/* Leading spacer mirrors SceneGrid's row-label column so the
              5 rec banners line up under the 5 stem columns. */}
          <div aria-hidden="true" />
          {stemColumns.map((c) => (
            <ColumnRecBanner key={c} column={c} k={5} />
          ))}
        </div>
      </main>
    </div>
  );
}
