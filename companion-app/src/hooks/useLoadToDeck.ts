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
 * the exact bugs they were added to close — hence one hook, two callers.
 */
export interface LoadToDeckVars {
  trackId: number;
  role: PlanRole;
  title: string | null;
  side: "a" | "b";
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
      }),
    onSuccess: (result, { trackId, title, side }) => {
      qc.invalidateQueries({ queryKey: ["ableton", "decks"] });
      const label = `${title ?? `Track #${trackId}`} → Deck ${side.toUpperCase()}`;
      // Immediate misses (missing stem file, Live unreachable) surface now…
      store.setLoadWarnings(label, result.warnings);
      // …and the warp audit follows once Live's analysis settles.
      scheduleWarpCheck(result.scene_index, label);
      store.armDeck(result.side ?? side, result.scene_index);
      store.registerDeck({
        track_id: trackId,
        scene_index: result.scene_index,
        side: result.side ?? side,
        stem_track_indices: sideTrackIndices(
          result.track_indices,
          result.side ?? side,
        ),
        loaded_at: Date.now(),
      });
    },
  });
}
