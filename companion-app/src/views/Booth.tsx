import { ColumnRecBanner } from "../components/ColumnRecBanner";
import { NowCard } from "../components/NowCard";
import { SceneGrid } from "../components/SceneGrid";
import { SetRail } from "../components/SetRail";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";
import { useNowPlayingTrack } from "../hooks/useNowPlayingTrack";

const STEM_COLUMNS = ["drums", "bass", "vocals", "other", "mix"];

/**
 * The Booth — the only screen you should look at during a set.
 *
 *   - SetRail (left): set arc, energy curve, played history.
 *   - NowCard (middle): currently-playing combo, structure timeline, stems.
 *   - Banners + SceneGrid (right, flex-1): per-column rec streams above the
 *     8×5 APC40-mirror grid. This is the centerpiece — most attention lives
 *     here during a swap. ⌘K opens free-text vibe search.
 *
 * Side effects:
 *   - Auto-creates a session on first Ableton play (useAutoSession).
 *   - Auto-logs plays as Ableton fires clips loaded via Load-to-Live.
 */
export function Booth() {
  useAutoSession();
  useAutoLog();
  const { trackId, source } = useNowPlayingTrack();

  return (
    <div className="flex-1 flex min-h-0">
      <SetRail />
      <NowCard trackId={trackId} liveLinked={source === "ableton"} />
      <section className="flex-1 flex flex-col min-h-0 border-l border-neutral-800 px-3 py-2 gap-2 overflow-y-auto">
        <div className="text-[10px] uppercase tracking-widest text-neutral-500">
          Per-column recs · Scene grid (APC40 mirror)
        </div>
        <div className="grid grid-cols-5 gap-1.5">
          {STEM_COLUMNS.map((c) => (
            <ColumnRecBanner key={c} column={c} k={3} />
          ))}
        </div>
        <SceneGrid />
      </section>
    </div>
  );
}
