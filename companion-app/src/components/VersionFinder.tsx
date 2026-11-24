import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { findVersionsOnSpotify, ingestSpotifyTrack, serverMessage } from "../api";
import { useAppendToPlan } from "../hooks/useSetPlan";
import { formatDurationMs } from "../lib/format";
import type { SpotifyVersion } from "../types";

const READY = "complete";
const FAILED = "error";

function versionsKey(trackId: number) {
  return ["spotify", "versions", trackId] as const;
}

export function VersionFinder({ trackId, setId }: { trackId: number; setId: number }) {
  const found = useQuery({
    queryKey: versionsKey(trackId),
    queryFn: () => findVersionsOnSpotify(trackId),
    retry: false,
    gcTime: 0,
  });

  if (found.isLoading) {
    return <p className="text-sm text-neutral-500">Searching Spotify…</p>;
  }
  if (found.isError) {
    return <p className="text-sm text-rose-300">{serverMessage(found.error as Error)}</p>;
  }
  const versions = found.data?.versions ?? [];
  return (
    <div className="flex flex-col gap-3">
      {versions.length === 0 ? (
        <p className="text-sm text-neutral-400">Spotify lists no other version of this song.</p>
      ) : (
        <div className="flex flex-col gap-0.5">
          {versions.map((version) => (
            <VersionRow
              key={version.spotify_id}
              version={version}
              originalId={trackId}
              setId={setId}
            />
          ))}
        </div>
      )}
      <p className="text-[11px] leading-relaxed text-neutral-600">
        Searched Spotify for <span className="font-mono">{found.data?.query}</span>. Get downloads
        a version and queues it after this song. It is ready to play once Process now has run in
        the Pipeline view.
      </p>
    </div>
  );
}

function whereItStands(version: SpotifyVersion): string | null {
  if (version.library_track_id == null) return null;
  if (version.download_failed) return "download failed";
  if (version.library_state === READY) return "in your library";
  if (version.library_state === FAILED) return "failed, see Pipeline";
  return "on its way";
}

function VersionRow({
  version,
  originalId,
  setId,
}: {
  version: SpotifyVersion;
  originalId: number;
  setId: number;
}) {
  const queryClient = useQueryClient();
  const addToPlan = useAppendToPlan();
  const [getting, setGetting] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const standing = whereItStands(version);
  const canBeAskedFor = version.library_track_id == null || version.download_failed;

  async function get() {
    setGetting(true);
    setFailure(null);
    try {
      const queued = await ingestSpotifyTrack(version);
      await addToPlan.mutateAsync({
        setId,
        role: "song",
        trackId: queued.track_id,
        afterTrackId: originalId,
      });
      queryClient.invalidateQueries({ queryKey: ["pipeline"] });
      await queryClient.invalidateQueries({ queryKey: versionsKey(originalId) });
    } catch (error) {
      setFailure(serverMessage(error as Error));
    } finally {
      setGetting(false);
    }
  }

  return (
    <div
      role="group"
      aria-label={`Version on Spotify: ${version.title}`}
      className="flex items-center gap-3 rounded-md px-2 py-1.5 hover:bg-neutral-900/60"
    >
      {version.image_url ? (
        <img src={version.image_url} alt="" className="h-9 w-9 shrink-0 rounded object-cover" />
      ) : (
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded bg-neutral-800 text-[10px] text-neutral-600">
          ♪
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium text-neutral-100">{version.title}</div>
        <div className="truncate text-xs text-neutral-500">{version.artist}</div>
      </div>
      <span className="shrink-0 rounded bg-fuchsia-500/25 px-1.5 py-0.5 text-[11px] text-fuchsia-200/90">
        {version.label} · {formatDurationMs(version.duration_ms)}
      </span>
      {failure && (
        <span className="max-w-[12rem] truncate text-[11px] text-rose-300" title={failure}>
          {failure}
        </span>
      )}
      {standing && (
        <span
          className={`shrink-0 text-right text-[11px] ${
            version.download_failed ? "text-rose-300" : "w-28 text-neutral-500"
          }`}
          title={version.download_failed ? "The Pipeline view says why" : undefined}
        >
          {standing}
        </span>
      )}
      {canBeAskedFor && (
        <span className={`flex shrink-0 justify-end ${standing ? "" : "w-28"}`}>
          <button
            type="button"
            onClick={get}
            disabled={getting}
            title="Download this version and queue it after the song in the Song column"
            className="min-h-[32px] rounded-md bg-emerald-700/80 px-3 text-xs font-semibold text-white hover:bg-emerald-600 disabled:bg-neutral-800 disabled:text-neutral-500"
          >
            {getting ? "Getting…" : version.download_failed ? "Try again" : "Get"}
          </button>
        </span>
      )}
    </div>
  );
}
