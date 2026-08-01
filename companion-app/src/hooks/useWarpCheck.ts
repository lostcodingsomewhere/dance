import { useCallback, useEffect, useRef } from "react";

import { warpCheck } from "../api";
import { store } from "../store";

/**
 * Live needs ~13-15s after a load before its auto-warp analysis lands.
 * Until then every clip reports a placeholder length (duration at the
 * current project tempo), so all stems agree and a check run any earlier
 * reports all-clear on a scene that's actually broken. Measured on Live
 * 12.4.2 — see docs/proposals/warp-guard.md.
 */
const SETTLE_MS = 18_000;

/**
 * Schedules the warp audit for a scene once Live's analysis has settled,
 * and pushes anything it finds into the load-warnings banner.
 *
 * Returned callback is safe to call on every load; a second call for the
 * same scene replaces the pending timer rather than stacking checks.
 */
export function useWarpCheck() {
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());

  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const t of pending.values()) clearTimeout(t);
      pending.clear();
    };
  }, []);

  return useCallback((sceneIndex: number, label: string) => {
    const existing = timers.current.get(sceneIndex);
    if (existing) clearTimeout(existing);
    timers.current.set(
      sceneIndex,
      setTimeout(() => {
        timers.current.delete(sceneIndex);
        warpCheck(sceneIndex)
          .then((r) => store.setLoadWarnings(label, r.warnings))
          // A failed audit must never surface as a scary banner — the load
          // itself already succeeded, and a missing audit is not a defect
          // the DJ can act on mid-set.
          .catch(() => undefined);
      }, SETTLE_MS),
    );
  }, []);
}
