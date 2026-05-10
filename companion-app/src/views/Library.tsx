import { useMemo, useState } from "react";
import { PinButton } from "../components/PinButton";
import { TrackCard } from "../components/TrackCard";
import { useTracks } from "../hooks/useTracks";

const CAMELOT_KEYS = [
  "",
  "1A",
  "1B",
  "2A",
  "2B",
  "3A",
  "3B",
  "4A",
  "4B",
  "5A",
  "5B",
  "6A",
  "6B",
  "7A",
  "7B",
  "8A",
  "8B",
  "9A",
  "9B",
  "10A",
  "10B",
  "11A",
  "11B",
  "12A",
  "12B",
];

export function Library() {
  const [search, setSearch] = useState("");
  const [bpmMin, setBpmMin] = useState<string>("");
  const [bpmMax, setBpmMax] = useState<string>("");
  const [key, setKey] = useState<string>("");
  const [energy, setEnergy] = useState<string>("");

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

      <div className="flex flex-wrap gap-3 items-center">
        <input
          type="search"
          placeholder="Search title or artist…"
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

      <div className="text-sm text-neutral-500">
        {tracks.isLoading
          ? "Loading…"
          : `${filtered.length} of ${tracks.data?.length ?? 0} tracks`}
      </div>

      {tracks.isError && (
        <div className="rounded-lg border border-red-700 bg-red-950/40 text-red-200 p-3 text-sm">
          Failed to load tracks: {(tracks.error as Error).message}
        </div>
      )}

      <div className="flex-1 overflow-auto flex flex-col gap-2 pr-1">
        {filtered.map((t) => (
          <TrackCard
            key={t.id}
            track={{
              id: t.id,
              title: t.title,
              artist: t.artist,
              bpm: t.analysis?.bpm,
              key_camelot: t.analysis?.key_camelot,
              floor_energy: t.analysis?.floor_energy,
            }}
            actions={<PinButton trackId={t.id} />}
          />
        ))}
      </div>
    </div>
  );
}
