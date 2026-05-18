import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";
import type { DeckScene } from "../types";

const ROLES = ["drums", "bass", "vocals", "other", "mix"] as const;
type Role = (typeof ROLES)[number];

const ROLE_ACCENT: Record<Role, { dot: string; chip: string }> = {
  drums:  { dot: "bg-red-500",    chip: "border-red-500/30 text-red-300" },
  bass:   { dot: "bg-amber-500",  chip: "border-amber-500/30 text-amber-300" },
  vocals: { dot: "bg-lime-400",   chip: "border-lime-500/30 text-lime-300" },
  other:  { dot: "bg-sky-400",    chip: "border-sky-500/30 text-sky-300" },
  mix:    { dot: "bg-neutral-200", chip: "border-neutral-500/40 text-neutral-200" },
};

/**
 * Horizontal 5-card row showing the *current active combo* — one card per
 * stem role with the source-track metadata of whatever's playing in that
 * role. In live-remixing there's no single "now playing" track; this strip
 * makes that honest by showing the source of each stem independently.
 *
 * When all 5 cells point at the same scene, the user is in "anchor mode"
 * (a whole row was fired); we surface that explicitly so they know the
 * combo is the original song.
 */
export function ComboStrip() {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();

  const columns = deckMap.data?.columns ?? null;
  const scenes = deckMap.data?.scenes ?? [];
  const playing = ableton.playing_clips ?? {};

  const sceneByIdx = useMemo(() => {
    const m = new Map<number, DeckScene>();
    for (const s of scenes) m.set(s.scene_index, s);
    return m;
  }, [scenes]);

  // Per-role: the scene currently playing in that role's column (or null).
  const cards = useMemo(() => {
    if (!columns) return null;
    return ROLES.map((role) => {
      const trackIdx = columns[role];
      const sceneIdx = trackIdx != null ? playing[trackIdx] : undefined;
      const scene = sceneIdx != null ? sceneByIdx.get(sceneIdx) : undefined;
      return { role, sceneIdx, scene };
    });
  }, [columns, playing, sceneByIdx]);

  // Anchor detection — if all non-empty cards point at the same scene index,
  // the user fired a whole row. The combo IS the original song combination.
  const anchorSceneIdx = useMemo(() => {
    if (!cards) return null;
    const occupied = cards.filter((c) => c.sceneIdx != null);
    if (occupied.length === 0) return null;
    const first = occupied[0].sceneIdx;
    if (occupied.every((c) => c.sceneIdx === first)) return first ?? null;
    return null;
  }, [cards]);
  const anchorScene = anchorSceneIdx != null ? sceneByIdx.get(anchorSceneIdx) : null;

  if (!columns) {
    return (
      <div className="rounded-lg border border-dashed border-neutral-800 px-4 py-3 text-xs text-neutral-600">
        Waiting for Ableton — load a track from the recs banner to begin a combo.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1" data-testid="combo-strip">
      <div className="flex items-baseline justify-between px-1">
        <div className="text-[10px] uppercase tracking-widest text-neutral-500">
          Current combo
        </div>
        {anchorScene ? (
          <div className="text-[10px] text-emerald-300/90 uppercase tracking-widest">
            ⚓ anchored to scene {anchorSceneIdx! + 1} ·{" "}
            <span className="text-neutral-200 normal-case tracking-normal">
              {anchorScene.title ?? `Track #${anchorScene.track_id}`}
            </span>
          </div>
        ) : (
          <div className="text-[10px] text-neutral-600 uppercase tracking-widest">
            live remix
          </div>
        )}
      </div>
      <div className="grid grid-cols-5 gap-1.5">
        {cards?.map((c) => (
          <ComboCard
            key={c.role}
            role={c.role}
            scene={c.scene}
            isAnchorPart={anchorSceneIdx != null && c.sceneIdx === anchorSceneIdx}
          />
        ))}
      </div>
    </div>
  );
}

function ComboCard({
  role,
  scene,
  isAnchorPart,
}: {
  role: Role;
  scene: DeckScene | undefined;
  isAnchorPart: boolean;
}) {
  const accent = ROLE_ACCENT[role];
  if (!scene) {
    return (
      <div className="rounded-md border border-neutral-900 bg-neutral-950/60 px-2 py-2 h-16">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-neutral-700">
          <span className={`w-1.5 h-1.5 rounded-full ${accent.dot} opacity-40`} />
          {role}
        </div>
        <div className="text-[11px] text-neutral-700 italic mt-1">silent</div>
      </div>
    );
  }
  return (
    <div
      className={`rounded-md border px-2 py-2 h-16 ${
        isAnchorPart
          ? "border-emerald-500/30 bg-emerald-500/5"
          : "border-neutral-800 bg-neutral-900/40"
      }`}
    >
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider">
        <span className={`w-1.5 h-1.5 rounded-full ${accent.dot}`} />
        <span className={accent.chip.split(" ").pop()}>{role}</span>
        {scene.key_camelot && (
          <span className="ml-auto font-mono text-neutral-400">
            {scene.key_camelot}
          </span>
        )}
      </div>
      <div className="text-xs text-neutral-100 truncate font-medium leading-tight mt-1">
        {scene.title ?? `Track #${scene.track_id}`}
      </div>
      <div className="text-[10px] text-neutral-500 truncate font-mono">
        {scene.bpm != null ? `${scene.bpm.toFixed(1)} BPM` : "—"}
        {scene.artist && (
          <span className="text-neutral-600 normal-case font-sans">
            {" · "}
            {scene.artist}
          </span>
        )}
      </div>
    </div>
  );
}
