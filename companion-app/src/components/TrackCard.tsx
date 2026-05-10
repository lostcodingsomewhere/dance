import { EnergyBar } from "./EnergyBar";
import { KeyBadge } from "./KeyBadge";

export interface TrackCardData {
  id: number;
  title: string | null;
  artist: string | null;
  bpm: number | null | undefined;
  key_camelot: string | null | undefined;
  floor_energy: number | null | undefined;
  score?: number | null;
}

export function TrackCard({
  track,
  actions,
  onClick,
}: {
  track: TrackCardData;
  actions?: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <div
      className={`flex items-center gap-4 p-4 bg-neutral-900 border border-neutral-800 rounded-xl ${
        onClick ? "cursor-pointer hover:bg-neutral-800" : ""
      }`}
      onClick={onClick}
    >
      <KeyBadge keyCamelot={track.key_camelot ?? null} size="md" />
      <div className="flex-1 min-w-0">
        <div className="text-lg font-semibold text-neutral-50 truncate">
          {track.title ?? `Track #${track.id}`}
        </div>
        <div className="text-sm text-neutral-400 truncate">
          {track.artist ?? "Unknown artist"}
        </div>
        <div className="mt-2 flex items-center gap-4">
          <span className="font-mono text-2xl text-neutral-100">
            {track.bpm != null ? track.bpm.toFixed(1) : "--"}
            <span className="text-xs text-neutral-500 ml-1">BPM</span>
          </span>
          <EnergyBar energy={track.floor_energy ?? null} size="sm" />
          {track.score != null && (
            <span className="text-xs text-neutral-500 font-mono">
              score {track.score.toFixed(2)}
            </span>
          )}
        </div>
      </div>
      {actions ? (
        <div className="flex items-center gap-2 shrink-0">{actions}</div>
      ) : null}
    </div>
  );
}
