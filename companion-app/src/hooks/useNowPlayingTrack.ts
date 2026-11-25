import { useEffect, useMemo, useRef, useState } from "react";
import { sourceKindOf } from "../lib/roles";
import type { DeckMap } from "../types";
import { useAbletonState } from "./useAbletonState";
import { useDeckMap } from "./useDeckMap";
import { useCurrentSession } from "./useSession";

interface PlayingSong {
  trackId: number;
  stems: number;
  lowestScene: number;
}

interface NowPlaying {
  leadingTrackId: number | null;
  trackIds: number[];
}

type SongOfCell = Map<string, number>;
type StartOrder = Map<number, number>;

const NOTHING_PLAYING: NowPlaying = { leadingTrackId: null, trackIds: [] };

function isSamePlaying(a: NowPlaying, b: NowPlaying): boolean {
  return (
    a.leadingTrackId === b.leadingTrackId &&
    a.trackIds.length === b.trackIds.length &&
    a.trackIds.every((trackId, index) => trackId === b.trackIds[index])
  );
}

function cellKey(scene: number, kind: string): string {
  return `${scene}:${kind}`;
}

// The bridge mutes Live's Mix tracks, so a mix clip plays along unheard.
function isHeardStem(kind: string): boolean {
  return sourceKindOf(kind) !== "mix";
}

function songsByCell(deckMap: DeckMap | undefined): SongOfCell {
  return new Map(
    (deckMap?.cells ?? []).map((cell) => [cellKey(cell.scene_index, cell.kind), cell.track_id]),
  );
}

function songsPlaying(
  columns: DeckMap["columns"] | undefined,
  songOfCell: SongOfCell,
  playingClips: Record<string, number>,
): PlayingSong[] {
  const songs = new Map<number, PlayingSong>();
  for (const [kind, liveTrack] of Object.entries(columns ?? {})) {
    const scene = playingClips[liveTrack];
    if (scene == null) continue;
    const trackId = songOfCell.get(cellKey(scene, kind));
    if (trackId == null) continue;
    const song = songs.get(trackId);
    songs.set(trackId, {
      trackId,
      stems: (song?.stems ?? 0) + (isHeardStem(kind) ? 1 : 0),
      lowestScene: Math.min(song?.lowestScene ?? scene, scene),
    });
  }
  return [...songs.values()];
}

function rememberStarts(startOrder: StartOrder, songs: PlayingSong[]): void {
  const playing = new Set(songs.map((song) => song.trackId));
  for (const trackId of [...startOrder.keys()]) {
    if (!playing.has(trackId)) startOrder.delete(trackId);
  }
  const justStarted = songs.filter((song) => !startOrder.has(song.trackId));
  const latest = Math.max(0, ...startOrder.values());
  for (const song of justStarted) startOrder.set(song.trackId, latest + 1);
}

function leadingSong(songs: PlayingSong[], startOrder: StartOrder): PlayingSong | undefined {
  return [...songs].sort(
    (a, b) =>
      b.stems - a.stems ||
      (startOrder.get(b.trackId) ?? 0) - (startOrder.get(a.trackId) ?? 0) ||
      a.lowestScene - b.lowestScene,
  )[0];
}

export function useNowPlayingTrack(): {
  trackId: number | null;
  source: "ableton" | "session" | "none";
  playingTrackIds: number[];
} {
  const playingClips = useAbletonState().playing_clips;
  const deckMap = useDeckMap().data;
  const lastPlay = useCurrentSession().data?.plays.at(-1);
  const songOfCell = useMemo(() => songsByCell(deckMap), [deckMap]);
  const startOrder = useRef<StartOrder>(new Map());
  const [nowPlaying, setNowPlaying] = useState(NOTHING_PLAYING);

  useEffect(() => {
    const songs = songsPlaying(deckMap?.columns, songOfCell, playingClips ?? {});
    rememberStarts(startOrder.current, songs);
    const next: NowPlaying = {
      leadingTrackId: leadingSong(songs, startOrder.current)?.trackId ?? null,
      trackIds: songs.map((song) => song.trackId).sort((a, b) => a - b),
    };
    setNowPlaying((shown) => (isSamePlaying(shown, next) ? shown : next));
  }, [deckMap, songOfCell, playingClips]);

  const { leadingTrackId, trackIds: playingTrackIds } = nowPlaying;
  if (leadingTrackId != null) {
    return { trackId: leadingTrackId, source: "ableton", playingTrackIds };
  }
  if (lastPlay) return { trackId: lastPlay.track_id, source: "session", playingTrackIds };
  return { trackId: null, source: "none", playingTrackIds };
}
