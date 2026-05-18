import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as api from "../api";

/** Bridge is considered stale if this much time has passed since the last
 * successful ping. The dance app polls /ableton/state every 2 s, so 5 s
 * absorbs one missed beat without false-alarming. */
const STALE_AFTER_MS = 5_000;

/**
 * Cheap liveness ping for AbletonOSC. The user can minimize Live during a set
 * (per the live-remixing redesign), but they still need confidence the engine
 * is responsive. This hook polls /ableton/state every 2 s and surfaces
 * "alive" vs "stale" without dragging in the full WebSocket subscription.
 *
 * Returns:
 *   alive       — true if last ping succeeded within STALE_AFTER_MS
 *   lastAliveAt — wall-clock timestamp of the most recent success (or null)
 */
export function useBridgeHeartbeat(): {
  alive: boolean;
  lastAliveAt: number | null;
} {
  const q = useQuery({
    queryKey: ["bridge", "heartbeat"],
    queryFn: () => api.abletonGetState(),
    refetchInterval: 2000,
    staleTime: 1500,
    retry: 0,
  });

  // Re-render every ~2 s so "alive" can flip to false from staleness alone
  // even if no successful or failed fetch has occurred.
  const [, setTick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 2000);
    return () => clearInterval(t);
  }, []);

  const lastSuccessAt = q.dataUpdatedAt > 0 ? q.dataUpdatedAt : null;
  const alive =
    lastSuccessAt != null && Date.now() - lastSuccessAt < STALE_AFTER_MS;

  return { alive, lastAliveAt: lastSuccessAt };
}
