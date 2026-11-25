import { useMutation, useQueryClient } from "@tanstack/react-query";

import { pushTrackToLive } from "../api";
import { sideTrackIndices } from "../lib/roles";
import { store } from "../store";
import type { PlanRole } from "../types";

import { useWarpCheck } from "./useWarpCheck";

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
    onSuccess: (result, load) => {
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
      const landedOn = result.side ?? load.side ?? null;
      if (result.stems_loaded === 0) {
        store.reportFailure(
          loadingOnto(load, landedOn),
          result.warnings.length > 0 ? result.warnings : ["No stems were loaded."],
        );
        return;
      }
      const label = `${songName(load)} → ${deckName(landedOn)}`;
      // Immediate misses (missing stem file, Live unreachable) surface now…
      store.setLoadWarnings(label, result.warnings);
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
