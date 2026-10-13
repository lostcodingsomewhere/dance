import { useSyncExternalStore } from "react";
import { createSharedSocket } from "../lib/sharedSocket";
import type { AbletonState } from "../types";
import { EMPTY_ABLETON_STATE } from "../types";

/**
 * The one /ws connection for the whole app. Every frame is a full
 * AbletonState snapshot, so the newest simply replaces the last. React is
 * told at most once per animation frame (see ``lib/sharedSocket``).
 */
const abletonSocket = createSharedSocket<AbletonState, AbletonState>({
  path: "/ws",
  initial: EMPTY_ABLETON_STATE,
  reduce: (_prev, next) => next,
});

/** Subscribes to /ws and returns the latest Ableton state pushed by the bridge.
 *
 * Any number of components can call this: they share one socket, and all of
 * them see the same snapshot in the same render. */
export function useAbletonState(): AbletonState {
  return useSyncExternalStore(
    abletonSocket.subscribe,
    abletonSocket.getSnapshot,
    abletonSocket.getServerSnapshot,
  );
}
