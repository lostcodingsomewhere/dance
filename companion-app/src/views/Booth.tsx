import { NowCard } from "../components/NowCard";
import { SetRail } from "../components/SetRail";
import { UpNextRail } from "../components/UpNextRail";
import { useAutoLog } from "../hooks/useAutoLog";
import { useAutoSession } from "../hooks/useAutoSession";
import { useNowPlayingTrack } from "../hooks/useNowPlayingTrack";

/**
 * The Booth — the only screen you should look at during a set.
 *
 * Three columns:
 *   - SetRail (left): the set arc so far + recent plays
 *   - NowCard (center): currently playing + structure timeline + stems
 *   - UpNextRail (right): recommendations seeded by what's playing
 *
 * Side effects:
 *   - Auto-creates a session on first Ableton play (useAutoSession)
 *   - Auto-logs plays as Ableton fires clips loaded via Load-to-Live
 *     (useAutoLog)
 */
export function Booth() {
  useAutoSession();
  useAutoLog();
  const { trackId, source } = useNowPlayingTrack();

  return (
    <div className="flex-1 flex min-h-0">
      <SetRail />
      <NowCard trackId={trackId} liveLinked={source === "ableton"} />
      <UpNextRail seedTrackId={trackId} />
    </div>
  );
}
