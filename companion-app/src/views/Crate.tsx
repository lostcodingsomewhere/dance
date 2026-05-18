import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { recommendByText } from "../api";
import { Stack } from "../components/Stack";
import { TrackCard } from "../components/TrackCard";
import { useTracks } from "../hooks/useTracks";
import { store, useAppStore } from "../store";
import type { Recommendation } from "../types";
import { LoadActions } from "../components/LoadActions";

const CAMELOT_KEYS = [
  "",
  "1A", "1B", "2A", "2B", "3A", "3B",
  "4A", "4B", "5A", "5B", "6A", "6B",
  "7A", "7B", "8A", "8B", "9A", "9B",
  "10A", "10B", "11A", "11B", "12A", "12B",
];

/**
 * The Crate — pre-set planning surface. Vibe search + standard filters,
 * paired with the Stack panel (the place to stage a list of tracks for an
 * upcoming set and batch-export their .als files).
 */
export function Crate() {
  const [search, setSearch] = useState("");
  const [vibeQuery, setVibeQuery] = useState("");
  const [bpmMin, setBpmMin] = useState("");
  const [bpmMax, setBpmMax] = useState("");
  const [key, setKey] = useState("");
  const [energy, setEnergy] = useState("");
  const stackIds = useAppStore((s) => s.stack);
  const stackSet = useMemo(() => new Set(stackIds), [stackIds]);

  const filters = useMemo(
    () => ({
      limit: 200,
      bpm_min: bpmMin ? Number(bpmMin) : undefined,
      bpm_max: bpmMax ? Number(bpmMax) : undefined,
      key: key || undefined,
      energy: energy ? Number(energy) : undefined,
    }),
    [bpmMin, bpmMax, key, energy],
  );
  const tracks = useTracks(filters);

  const vibe = useMutation<Recommendation[], Error, string>({
    mutationFn: (q: string) => recommendByText(q, 20),
  });

  const runVibe = (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = vibeQuery.trim();
    if (q) vibe.mutate(q);
  };

  const clearVibe = () => {
    vibe.reset();
    setVibeQuery("");
  };

  const showingVibe = vibe.data !== undefined;

  const filtered = useMemo(() => {
    if (!tracks.data) return [];
    const q = search.trim().toLowerCase();
    if (!q) return tracks.data;
    return tracks.data.filter((t) => {
      const hay = `${t.title ?? ""} ${t.artist ?? ""}`.toLowerCase();
      return hay.includes(q);
    });
  }, [tracks.data, search]);

  return (
    <div className="flex-1 grid grid-cols-[1fr_360px] gap-4 p-6 overflow-hidden">
      <div className="flex flex-col gap-4 min-h-0">
        <h1 className="text-3xl font-bold text-neutral-50">Crate</h1>

        <form onSubmit={runVibe} className="flex gap-2 items-center">
          <span className="text-2xl select-none" aria-hidden>
            ✦
          </span>
          <input
            type="text"
            placeholder="Describe a vibe: 'punchy techy with vocals', 'deep rolling bassline'…"
            value={vibeQuery}
            onChange={(e) => setVibeQuery(e.target.value)}
            className="flex-1 h-12 px-3 rounded-lg bg-neutral-900 border-2 border-purple-700/50 focus:border-purple-500 text-neutral-100 placeholder:text-neutral-600 outline-none"
          />
          <button
            type="submit"
            disabled={!vibeQuery.trim() || vibe.isPending}
            className="h-12 px-5 rounded-lg bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white font-semibold"
          >
            {vibe.isPending ? "Searching…" : "Vibe Search"}
          </button>
          {showingVibe && (
            <button
              type="button"
              onClick={clearVibe}
              className="h-12 px-3 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-sm"
            >
              Clear
            </button>
          )}
        </form>

        {vibe.isError && (
          <div className="rounded-lg border border-red-700 bg-red-950/40 text-red-200 p-3 text-sm">
            Vibe search failed: {vibe.error.message}
          </div>
        )}

        {!showingVibe && (
          <div className="flex flex-wrap gap-2 items-center">
            <input
              type="search"
              placeholder="Title or artist…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="flex-1 min-w-[200px] h-10 px-3 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100 placeholder:text-neutral-600 outline-none focus:border-neutral-600"
            />
            <input
              type="number"
              placeholder="BPM ≥"
              value={bpmMin}
              onChange={(e) => setBpmMin(e.target.value)}
              className="w-24 h-10 px-2 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100"
            />
            <input
              type="number"
              placeholder="BPM ≤"
              value={bpmMax}
              onChange={(e) => setBpmMax(e.target.value)}
              className="w-24 h-10 px-2 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100"
            />
            <select
              value={key}
              onChange={(e) => setKey(e.target.value)}
              className="h-10 px-2 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100"
            >
              {CAMELOT_KEYS.map((k) => (
                <option key={k || "any"} value={k}>
                  {k || "Any key"}
                </option>
              ))}
            </select>
            <select
              value={energy}
              onChange={(e) => setEnergy(e.target.value)}
              className="h-10 px-2 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100"
            >
              <option value="">Any energy</option>
              {Array.from({ length: 10 }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  Energy {n}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="text-sm text-neutral-500">
          {showingVibe
            ? `${vibe.data?.length ?? 0} vibe matches for "${vibeQuery}"`
            : tracks.isLoading
              ? "Loading…"
              : `${filtered.length} of ${tracks.data?.length ?? 0} tracks`}
        </div>

        {tracks.isError && !showingVibe && (
          <div className="rounded-lg border border-red-700 bg-red-950/40 text-red-200 p-3 text-sm">
            Failed to load tracks: {(tracks.error as Error).message}
          </div>
        )}

        <div className="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
          {showingVibe
            ? vibe.data?.map((r) => (
                <TrackCard
                  key={r.track_id}
                  track={{
                    id: r.track_id,
                    title: r.title,
                    artist: r.artist,
                    bpm: r.bpm,
                    key_camelot: r.key_camelot,
                    floor_energy: r.floor_energy,
                  }}
                  badge={`✦ ${r.score.toFixed(2)}`}
                  actions={
                    <CrateActions
                      trackId={r.track_id}
                      path={r.file_path}
                      inStack={stackSet.has(r.track_id)}
                    />
                  }
                />
              ))
            : filtered.map((t) => (
                <TrackCard
                  key={t.id}
                  track={{
                    id: t.id,
                    title: t.title,
                    artist: t.artist,
                    bpm: t.analysis?.bpm,
                    key_camelot: t.analysis?.key_camelot,
                    floor_energy: t.analysis?.floor_energy,
                    tags: t.tags,
                  }}
                  actions={
                    <CrateActions
                      trackId={t.id}
                      path={t.file_path}
                      inStack={stackSet.has(t.id)}
                    />
                  }
                />
              ))}
        </div>
      </div>

      {/* Right rail: Stack */}
      <div className="min-h-0 overflow-y-auto">
        <Stack />
      </div>
    </div>
  );
}

function CrateActions({
  trackId,
  path,
  inStack,
}: {
  trackId: number;
  path: string | null | undefined;
  inStack: boolean;
}) {
  return (
    <>
      <LoadActions trackId={trackId} path={path} />
      <button
        type="button"
        onClick={() =>
          inStack ? store.removeFromStack(trackId) : store.addToStack(trackId)
        }
        className={`min-h-[40px] px-3 rounded-md text-xs font-semibold ${
          inStack
            ? "bg-amber-400 text-neutral-950 hover:bg-amber-300"
            : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700"
        }`}
        title={inStack ? "Remove from Stack" : "Add to Stack"}
      >
        {inStack ? "Stacked" : "+ Stack"}
      </button>
    </>
  );
}
