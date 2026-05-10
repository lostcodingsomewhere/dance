import { useMutation } from "@tanstack/react-query";
import { useMemo } from "react";
import * as api from "../api";
import { LoadActions } from "../components/LoadActions";
import { TrackCard } from "../components/TrackCard";
import { useRecommend } from "../hooks/useRecommend";
import { useAddPlay, useCurrentSession } from "../hooks/useSession";
import { store, useAppStore } from "../store";

export function UpNext() {
  const session = useCurrentSession();
  const pinnedSeeds = useAppStore((s) => s.pinnedSeeds);

  // Seeds = pinned + most-recent-play (if any).
  const lastPlayTrackId = session.data?.plays.at(-1)?.track_id ?? null;
  const seeds = useMemo(() => {
    const set = new Set<number>(pinnedSeeds);
    if (lastPlayTrackId != null) set.add(lastPlayTrackId);
    return [...set];
  }, [pinnedSeeds, lastPlayTrackId]);

  const recs = useRecommend(seeds, 12);
  const addPlay = useAddPlay(session.data?.id ?? null);
  const fireClip = useMutation({
    mutationFn: ({ track, scene }: { track: number; scene: number }) =>
      api.abletonFireClip(track, scene),
  });

  return (
    <div className="flex-1 flex flex-col gap-4 p-6 overflow-auto">
      <div className="flex items-baseline gap-3">
        <h1 className="text-3xl font-bold text-neutral-50">Up Next</h1>
        <span className="text-neutral-500 text-sm">
          {seeds.length} seed{seeds.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {seeds.length === 0 && (
          <div className="text-neutral-500 text-sm">
            No seeds yet — pin a track or start a session with at least one play.
          </div>
        )}
        {seeds.map((id) => {
          const isPinned = pinnedSeeds.includes(id);
          return (
            <span
              key={id}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-900 border border-neutral-800 text-sm text-neutral-300"
            >
              <span>#{id}</span>
              {isPinned ? (
                <button
                  type="button"
                  onClick={() => store.unpin(id)}
                  className="text-neutral-500 hover:text-neutral-200 px-1"
                  aria-label={`Remove seed ${id}`}
                  title="Unpin"
                >
                  ×
                </button>
              ) : (
                <span className="text-neutral-600 text-[10px] uppercase">
                  playing
                </span>
              )}
            </span>
          );
        })}
      </div>

      {recs.isError && (
        <div className="rounded-lg border border-red-700 bg-red-950/40 text-red-200 p-3 text-sm">
          Failed to fetch recommendations: {(recs.error as Error).message}
        </div>
      )}

      <div className="flex flex-col gap-3">
        {recs.isLoading && seeds.length > 0 && (
          <div className="text-neutral-500">Loading recommendations…</div>
        )}
        {recs.data?.map((rec) => (
          <TrackCard
            key={rec.track_id}
            track={{
              id: rec.track_id,
              title: rec.title,
              artist: rec.artist,
              bpm: rec.bpm,
              key_camelot: rec.key_camelot,
              floor_energy: rec.floor_energy,
              score: rec.score,
            }}
            actions={
              <>
                <LoadActions path={rec.file_path} />
                <button
                  type="button"
                  onClick={() =>
                    fireClip.mutate({ track: rec.track_id, scene: 0 })
                  }
                  className="min-h-[44px] min-w-[64px] px-4 rounded-lg bg-emerald-500 text-neutral-950 font-semibold hover:bg-emerald-400"
                >
                  Play
                </button>
                <button
                  type="button"
                  onClick={() => store.pin(rec.track_id)}
                  className="min-h-[44px] px-4 rounded-lg bg-neutral-800 text-neutral-200 hover:bg-neutral-700 font-semibold"
                >
                  Pin
                </button>
                <button
                  type="button"
                  disabled={session.data?.id == null}
                  onClick={() =>
                    addPlay.mutate({ track_id: rec.track_id })
                  }
                  className="min-h-[44px] px-4 rounded-lg bg-neutral-800 text-neutral-200 hover:bg-neutral-700 font-semibold disabled:opacity-40"
                >
                  + Session
                </button>
              </>
            }
          />
        ))}
      </div>
    </div>
  );
}
