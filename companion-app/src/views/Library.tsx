import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { recommendByText } from "../api";
import { LoadActions } from "../components/LoadActions";
import { PinButton } from "../components/PinButton";
import { TrackCard } from "../components/TrackCard";
import { useTracks } from "../hooks/useTracks";
import type { Recommendation } from "../types";

const CAMELOT_KEYS = [
  "",
  "1A", "1B", "2A", "2B", "3A", "3B",
  "4A", "4B", "5A", "5B", "6A", "6B",
  "7A", "7B", "8A", "8B", "9A", "9B",
  "10A", "10B", "11A", "11B", "12A", "12B",
];

export function Library() {
  const [search, setSearch] = useState("");
  const [vibeQuery, setVibeQuery] = useState("");
  const [bpmMin, setBpmMin] = useState("");
  const [bpmMax, setBpmMax] = useState("");
  const [key, setKey] = useState("");
  const [energy, setEnergy] = useState("");

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

  // Vibe search uses CLAP text-to-audio retrieval. Triggered explicitly, not
  // on every keystroke — first call lazy-loads the CLAP model server-side.
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

  // When vibe results are active, they take precedence over the filtered list.
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
    <div className="flex-1 flex flex-col gap-4 p-6 overflow-hidden">
      <h1 className="text-3xl font-bold text-neutral-50">Library</h1>

      {/* Vibe search bar — dedicated, prominent */}
      <form onSubmit={runVibe} className="flex gap-2 items-center">
        <span className="text-2xl select-none" aria-hidden>✦</span>
        <input
          type="text"
          placeholder="Describe a vibe: 'punchy techy with vocals', 'deep rolling bassline', 'afro-house drums'…"
          value={vibeQuery}
          onChange={(e) => setVibeQuery(e.target.value)}
          className="flex-1 h-14 px-4 rounded-lg bg-neutral-900 border-2 border-purple-700/50 focus:border-purple-500 text-neutral-100 text-lg placeholder:text-neutral-600 outline-none"
        />
        <button
          type="submit"
          disabled={!vibeQuery.trim() || vibe.isPending}
          className="h-14 px-6 rounded-lg bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white font-semibold"
        >
          {vibe.isPending ? "Searching…" : "Vibe Search"}
        </button>
        {showingVibe && (
          <button
            type="button"
            onClick={clearVibe}
            className="h-14 px-4 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200"
          >
            Clear
          </button>
        )}
      </form>

      {/* Vibe error surface */}
      {vibe.isError && (
        <div className="rounded-lg border border-red-700 bg-red-950/40 text-red-200 p-3 text-sm">
          Vibe search failed: {vibe.error.message}
        </div>
      )}

      {/* Standard filters — only when not in vibe mode */}
      {!showingVibe && (
        <div className="flex flex-wrap gap-3 items-center">
          <input
            type="search"
            placeholder="Title or artist…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 min-w-[240px] h-12 px-4 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100 placeholder:text-neutral-600 outline-none focus:border-neutral-600"
          />
          <input
            type="number"
            placeholder="BPM ≥"
            value={bpmMin}
            onChange={(e) => setBpmMin(e.target.value)}
            className="w-28 h-12 px-3 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100 placeholder:text-neutral-600"
          />
          <input
            type="number"
            placeholder="BPM ≤"
            value={bpmMax}
            onChange={(e) => setBpmMax(e.target.value)}
            className="w-28 h-12 px-3 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100 placeholder:text-neutral-600"
          />
          <select
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="h-12 px-3 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100"
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
            className="h-12 px-3 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-100"
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

      {/* Status line */}
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

      <div className="flex-1 overflow-auto flex flex-col gap-2 pr-1">
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
                  <>
                    <LoadActions path={r.file_path} />
                    <PinButton trackId={r.track_id} />
                  </>
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
                  <>
                    <LoadActions path={t.file_path} />
                    <PinButton trackId={t.id} />
                  </>
                }
              />
            ))}
      </div>
    </div>
  );
}
