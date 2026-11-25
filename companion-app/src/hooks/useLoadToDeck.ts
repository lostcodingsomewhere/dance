import { useMutation, useQueryClient, type QueryClient } from "@tanstack/react-query";

import { abletonDeckMap, pushTrackToLive, type LoadTrackResult } from "../api";
import { pageOfScene } from "../lib/pages";
import { sideOf, sideTrackIndices } from "../lib/roles";
import { store } from "../store";
import type { DeckCell, DeckMap, PlanRole } from "../types";

import { useWarpCheck } from "./useWarpCheck";

const DECK_MAP = ["ableton", "decks"];
const SONG_COLUMN = "mix";
const ONLY_THE_SONG = "No stems were loaded, only the song.";

function loadKinds(role: PlanRole): string[] | undefined {
  return role === "song" ? undefined : [role];
}

/**
 * Load a track onto deck A or B, with everything that has to happen after.
 *
 * Extracted from the card's ⤒A/⤒B buttons so the keyboard commit and the
 * mouse commit run the SAME code. The follow-up work here is not optional
 * decoration — each piece fixes a bug that was found on the real rig:
 *
 *  - ``armDeck`` points that deck's ▶ at what was just loaded. Without it the
 *    play button falls back to an anchor heuristic that replays whatever is
 *    already going.
 *  - ``registerDeck`` is what makes the play LOGGABLE; the auto-logger only
 *    recognises a playing clip whose deck it knows about.
 *  - ``scheduleWarpCheck`` audits the stems against each other once Live's
 *    analysis settles, because Live warps stems independently and gets it
 *    wrong often enough to matter.
 *
 * A second commit path that quietly skipped any of these would reintroduce
 * the exact bugs they were added to close — hence one hook, three callers.
 */
export interface LoadToDeckVars {
  trackId: number;
  role: PlanRole;
  title: string | null;
  /** Left out, Live picks the deck and the answer says which. */
  side?: "a" | "b";
}

function songName({ title, trackId }: LoadToDeckVars): string {
  return title ?? `Track #${trackId}`;
}

function deckName(side?: "a" | "b" | null): string {
  return side ? `Deck ${side.toUpperCase()}` : "a deck";
}

function loadingOnto(load: LoadToDeckVars, side?: "a" | "b" | null): string {
  return `load ${songName(load)} onto ${deckName(side)}`;
}

function songPads(map: DeckMap | undefined, trackId: number): DeckCell[] {
  return (map?.cells ?? []).filter(
    (cell) => cell.track_id === trackId && cell.kind.startsWith(SONG_COLUMN),
  );
}

function padKey(cell: DeckCell): string {
  return `${cell.scene_index}:${cell.kind}`;
}

// stems_loaded leaves out the Song column, so the deck map says whether the song got there.
async function onlyTheSongLanded(
  qc: QueryClient,
  load: LoadToDeckVars,
  result: LoadTrackResult,
  padsBefore: Set<string>,
): Promise<boolean> {
  if (load.role !== "song") return false;
  const side = result.side ?? load.side ?? null;
  try {
    const map = await qc.fetchQuery({ queryKey: DECK_MAP, queryFn: abletonDeckMap });
    return songPads(map, load.trackId).some(
      (cell) =>
        !padsBefore.has(padKey(cell)) &&
        (side === null || sideOf(cell.kind) === side) &&
        pageOfScene(cell.scene_index) === pageOfScene(result.scene_index),
    );
  } catch {
    return false;
  }
}

export function useLoadToDeck() {
  const qc = useQueryClient();
  const scheduleWarpCheck = useWarpCheck();
  return useMutation({
    mutationFn: ({ trackId, role, side }: LoadToDeckVars) =>
      pushTrackToLive(trackId, {
        includeStems: true,
        kinds: loadKinds(role),
        side,
        page: store.peekBoothPage(),
      }),
    meta: { attempt: (load: LoadToDeckVars) => loadingOnto(load, load.side) },
    onMutate: ({ trackId }) => ({
      songPadsBefore: new Set(songPads(qc.getQueryData<DeckMap>(DECK_MAP), trackId).map(padKey)),
    }),
    onSuccess: async (result, load, { songPadsBefore }) => {
      qc.invalidateQueries({ queryKey: DECK_MAP });
      const landedOn = result.side ?? load.side ?? null;
      const noStems = result.stems_loaded === 0;
      const onlyTheSong =
        noStems && (await onlyTheSongLanded(qc, load, result, songPadsBefore));
      if (noStems && !onlyTheSong) {
        store.reportFailure(
          loadingOnto(load, landedOn),
          result.warnings.length > 0 ? result.warnings : ["No stems were loaded."],
        );
        return;
      }
      const label = `${songName(load)} → ${deckName(landedOn)}`;
      // Immediate misses (missing stem file, Live unreachable) surface now…
      store.setLoadWarnings(
        label,
        onlyTheSong ? [ONLY_THE_SONG, ...result.warnings] : result.warnings,
      );
      // …and the warp audit follows once Live's analysis settles.
      scheduleWarpCheck(result.scene_index, label);
      store.armDeck(landedOn, result.scene_index);
      store.registerDeck({
        track_id: load.trackId,
        scene_index: result.scene_index,
        side: landedOn,
        stem_track_indices: sideTrackIndices(result.track_indices, landedOn),
        loaded_at: Date.now(),
      });
    },
  });
}
