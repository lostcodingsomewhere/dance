import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { pushTrackToLive, recommendByText, revealPath } from "../api";
import { useDeckMap } from "../hooks/useDeckMap";
import { useRecommend } from "../hooks/useRecommend";
import { useTrack } from "../hooks/useTracks";
import { store, useAppStore } from "../store";
import type { Recommendation } from "../types";
import { EnergyBar } from "./EnergyBar";
import { KeyBadge } from "./KeyBadge";

/** Quick-vibe chips → re-rank Up Next via CLAP text-to-audio retrieval. */
const QUICK_VIBES: { label: string; query: string }[] = [
  { label: "Bigger", query: "big room peak time techno hands up" },
  { label: "Deeper", query: "deep rolling bassline hypnotic" },
  { label: "Calmer", query: "warm melodic deep emotional" },
  { label: "Darker", query: "dark techno minor key driving" },
  { label: "Vocal", query: "strong female vocals house" },
  { label: "Instrumental", query: "instrumental tech house no vocals" },
];

interface Props {
  /** Track currently playing — seeds the default recommendation list. */
  seedTrackId: number | null;
}

/**
 * Right rail of The Booth. Two modes:
 *   1. **Seed mode** (default) — top-N edges out of the currently-playing track.
 *   2. **Vibe-override mode** — quick-vibe chip clicked or "more like this"
 *      tapped on a rec → replace the list with CLAP text-retrieval results.
 *      Cleared by tapping the chip again or "Back to recs".
 */
export function UpNextRail({ seedTrackId }: Props) {
  const seedRecs = useRecommend(seedTrackId != null ? [seedTrackId] : [], 8);
  const seedTrack = useTrack(seedTrackId);
  const deckMap = useDeckMap();
  const loadedTrackIds = useMemo(
    () => new Set((deckMap.data?.scenes ?? []).map((s) => s.track_id)),
    [deckMap.data],
  );

  const [vibe, setVibe] = useState<{ label: string; query: string } | null>(
    null,
  );
  const vibeQuery = useMutation<Recommendation[], Error, string>({
    mutationFn: (q: string) => recommendByText(q, 8),
  });

  const showingVibe = vibe != null;
  const recs = showingVibe ? vibeQuery.data : seedRecs.data;
  const recsLoading = showingVibe ? vibeQuery.isPending : seedRecs.isLoading;
  const recsError = showingVibe ? vibeQuery.error : seedRecs.error;

  function pickVibe(v: { label: string; query: string }) {
    if (vibe?.label === v.label) {
      setVibe(null);
      vibeQuery.reset();
      return;
    }
    setVibe(v);
    vibeQuery.mutate(v.query);
  }

  function moreLikeThis(rec: Recommendation) {
    const q = `${rec.title ?? ""} ${rec.artist ?? ""}`.trim() || `track ${rec.track_id}`;
    setVibe({ label: `like ${rec.title?.slice(0, 18) ?? "this"}`, query: q });
    vibeQuery.mutate(q);
  }

  function clearVibe() {
    setVibe(null);
    vibeQuery.reset();
  }

  return (
    <aside className="w-96 shrink-0 border-l border-neutral-800 flex flex-col min-h-0">
      <div className="px-4 pt-4 pb-1 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-widest text-neutral-500">
          Up Next
        </h2>
        <button
          type="button"
          onClick={() => store.openCommandBar()}
          className="text-[11px] text-neutral-500 hover:text-neutral-200"
          title="Vibe-search overrides the recommendations"
        >
          vibe ⌘K
        </button>
      </div>

      <div className="px-3 pb-2 flex flex-wrap gap-1">
        {QUICK_VIBES.map((v) => {
          const active = vibe?.label === v.label;
          return (
            <button
              key={v.label}
              type="button"
              onClick={() => pickVibe(v)}
              className={`px-2 py-1 rounded-md text-[11px] font-semibold ${
                active
                  ? "bg-purple-600 text-white"
                  : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
              }`}
              title={`Re-rank Up Next: ${v.query}`}
            >
              {v.label}
            </button>
          );
        })}
        {showingVibe && (
          <button
            type="button"
            onClick={clearVibe}
            className="ml-auto px-2 py-1 rounded-md text-[11px] text-neutral-400 hover:text-neutral-200"
            title="Go back to recommendations from the playing track"
          >
            ← back
          </button>
        )}
      </div>

      {showingVibe && (
        <div className="px-4 pb-2 text-[11px] text-purple-300/80 truncate">
          ✦ {vibe.label} — "{vibe.query}"
        </div>
      )}

      {seedTrackId == null && !showingVibe && (
        <div className="px-4 pb-4 text-sm text-neutral-500">
          Recommendations appear once a track is playing — or pick a vibe
          chip above.
        </div>
      )}

      {recsLoading && (
        <div className="px-4 pb-4 text-sm text-neutral-500">Loading…</div>
      )}

      {recsError && (
        <div className="mx-3 mb-3 rounded border border-rose-700 bg-rose-950/40 text-rose-200 p-2 text-xs">
          {(recsError as Error).message}
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-3 pb-4 flex flex-col gap-2">
        {recs?.map((rec) => (
          <UpNextCard
            key={rec.track_id}
            rec={rec}
            seedKey={seedTrack.data?.analysis?.key_camelot ?? null}
            seedBpm={seedTrack.data?.analysis?.bpm ?? null}
            seedEnergy={seedTrack.data?.analysis?.floor_energy ?? null}
            alreadyLoaded={loadedTrackIds.has(rec.track_id)}
            onMoreLikeThis={() => moreLikeThis(rec)}
          />
        ))}
      </div>
    </aside>
  );
}

function UpNextCard({
  rec,
  seedKey,
  seedBpm,
  seedEnergy,
  alreadyLoaded,
  onMoreLikeThis,
}: {
  rec: Recommendation;
  seedKey: string | null;
  seedBpm: number | null;
  seedEnergy: number | null;
  alreadyLoaded: boolean;
  onMoreLikeThis: () => void;
}) {
  const compat = useMemo(
    () => computeCompat(rec, { seedKey, seedBpm, seedEnergy }),
    [rec, seedKey, seedBpm, seedEnergy],
  );
  const nextScene = useAppStore((s) => {
    const used = Object.keys(s.loadedDecks).map(Number);
    return used.length === 0 ? 0 : Math.max(...used) + 1;
  });
  const [loading, setLoading] = useState(false);
  const [loadedMsg, setLoadedMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadToLive() {
    const sceneIndex = store.peekNextScene();
    try {
      setError(null);
      setLoading(true);
      const result = await pushTrackToLive(rec.track_id, {
        includeStems: true,
        sceneIndex,
      });
      const stemIndices = Object.values(result.track_indices);
      store.registerDeck({
        track_id: rec.track_id,
        scene_index: result.scene_index,
        stem_track_indices: stemIndices,
        loaded_at: Date.now(),
      });
      const fully = result.stems_loaded === 4;
      setLoadedMsg(
        fully
          ? `✓ scene ${result.scene_index + 1} — fire it`
          : `⚠ ${result.stems_loaded}/4 stems`,
      );
      // Only reveal in Finder when auto-load was partial/none — full loads
      // don't need the user to drag anything.
      if (!fully && rec.file_path) {
        try {
          await revealPath(rec.file_path);
        } catch {
          // best-effort
        }
      }
      setTimeout(() => setLoadedMsg(null), 3500);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-neutral-800 hover:border-neutral-700 bg-neutral-900 hover:bg-neutral-900/70 p-3 flex flex-col gap-2 transition-colors duration-100 ease-out">
      <div className="flex items-start gap-2">
        <KeyBadge keyCamelot={rec.key_camelot ?? null} size="sm" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-neutral-100 truncate">
            {rec.title ?? `Track #${rec.track_id}`}
          </div>
          <div className="text-xs text-neutral-500 truncate">
            {rec.artist ?? "Unknown"}
          </div>
        </div>
        <button
          type="button"
          onClick={onMoreLikeThis}
          className="text-neutral-500 hover:text-purple-300 text-xs px-1"
          title="More like this — re-rank Up Next around this rec"
        >
          ✦+
        </button>
      </div>

      <div className="flex items-center gap-2 flex-wrap text-[11px]">
        <span className="font-mono text-neutral-300 tabular-nums">
          {rec.bpm != null ? rec.bpm.toFixed(1) : "--"}
          <span className="text-neutral-500 ml-0.5">BPM</span>
        </span>
        <EnergyBar energy={rec.floor_energy ?? null} size="sm" />
        <CompatChip {...compat} />
      </div>

      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={loadToLive}
          disabled={loading || alreadyLoaded}
          className={`flex-1 min-h-[36px] px-3 rounded-md text-xs font-semibold ${
            alreadyLoaded
              ? "bg-emerald-700/30 text-emerald-200 ring-1 ring-emerald-500/30"
              : "bg-purple-700 hover:bg-purple-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-white"
          }`}
          title={
            alreadyLoaded
              ? "Already staged in Live on a scene"
              : `Stage on scene ${nextScene + 1}`
          }
        >
          {alreadyLoaded
            ? "✓ Loaded"
            : loading
              ? "Loading…"
              : loadedMsg ?? `Load → scene ${nextScene + 1}`}
        </button>
      </div>

      {error && (
        <div className="text-[11px] text-rose-300" title={error}>
          {error.slice(0, 60)}
        </div>
      )}
    </div>
  );
}

function CompatChip(c: ReturnType<typeof computeCompat>) {
  const tone =
    c.score >= 3
      ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
      : c.score >= 1
        ? "bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30"
        : "bg-neutral-800 text-neutral-400";
  return (
    <span
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono ${tone}`}
      title={c.detail}
    >
      <span className="text-neutral-500">K</span>
      {c.keyLabel}
      <span className="text-neutral-500 ml-0.5">B</span>
      {c.bpmLabel}
    </span>
  );
}

function computeCompat(
  rec: Recommendation,
  seed: { seedKey: string | null; seedBpm: number | null; seedEnergy: number | null },
): {
  keyLabel: string;
  bpmLabel: string;
  detail: string;
  score: number;
} {
  const bpmDiff =
    seed.seedBpm != null && rec.bpm != null
      ? Math.round(rec.bpm - seed.seedBpm)
      : null;
  const keyShift = camelotShift(seed.seedKey, rec.key_camelot ?? null);
  let score = 0;
  if (bpmDiff != null && Math.abs(bpmDiff) <= 3) score += 2;
  else if (bpmDiff != null && Math.abs(bpmDiff) <= 6) score += 1;
  if (keyShift != null && Math.abs(keyShift) <= 1) score += 2;
  else if (keyShift === 7 || keyShift === -7) score += 1;
  const keyLabel =
    keyShift == null
      ? "?"
      : keyShift > 0
        ? `+${keyShift}`
        : String(keyShift);
  const bpmLabel =
    bpmDiff == null ? "?" : bpmDiff > 0 ? `+${bpmDiff}` : String(bpmDiff);
  const keyExpl =
    keyShift == null
      ? "key unknown"
      : keyShift === 0
        ? "same key"
        : Math.abs(keyShift) === 1
          ? `adjacent key (${keyShift > 0 ? "+1" : "-1"} on Camelot)`
          : Math.abs(keyShift) === 7
            ? "relative major/minor flip"
            : `key shift ${keyShift}`;
  const bpmExpl =
    bpmDiff == null
      ? "BPM unknown"
      : bpmDiff === 0
        ? "same BPM"
        : `BPM ${bpmDiff > 0 ? "+" : ""}${bpmDiff}`;
  return {
    keyLabel,
    bpmLabel,
    detail: `${keyExpl} · ${bpmExpl}`,
    score,
  };
}

function camelotShift(a: string | null, b: string | null): number | null {
  if (!a || !b) return null;
  const re = /^(\d{1,2})([AB])$/;
  const ma = re.exec(a);
  const mb = re.exec(b);
  if (!ma || !mb) return null;
  const na = Number(ma[1]);
  const nb = Number(mb[1]);
  let diff = nb - na;
  if (ma[2] !== mb[2]) {
    if (diff === 0) return 7;
  }
  if (diff > 6) diff -= 12;
  if (diff < -6) diff += 12;
  return diff;
}
