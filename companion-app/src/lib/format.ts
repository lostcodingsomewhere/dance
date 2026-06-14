/**
 * Time/number formatting helpers shared across companion-app views.
 *
 * Kept in one place so SceneGrid, TwoDeckStrip, CueStrip, SetRail, SetEditor,
 * CommandBar, etc. render times consistently. Before this lived here, the
 * mm:ss formatting was inlined in CommandBar.tsx and nowhere else — so
 * stem/song rows everywhere else had no duration display at all.
 */

/**
 * Format a duration in seconds as ``M:SS`` (or ``H:MM:SS`` past an hour).
 * Returns the placeholder ``"—"`` when given null/undefined so call sites
 * can safely render whatever the API gave them without a guard.
 *
 *   formatDuration(0)            → "0:00"
 *   formatDuration(45.4)         → "0:45"
 *   formatDuration(127)          → "2:07"
 *   formatDuration(3725)         → "1:02:05"
 *   formatDuration(null)         → "—"
 */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  }
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Render a remaining-time as ``-M:SS``. Convenience over ``formatDuration``
 * so playing cards can show ``-1:23`` and a glance reads "1:23 till end."
 *
 *   formatRemaining(127)         → "-2:07"
 *   formatRemaining(null)        → "—"
 */
export function formatRemaining(secondsRemaining: number | null | undefined): string {
  if (secondsRemaining == null || !Number.isFinite(secondsRemaining)) return "—";
  if (secondsRemaining <= 0) return "0:00";
  return `-${formatDuration(secondsRemaining)}`;
}

/**
 * Format milliseconds as ``M:SS``. Spotify durations come in ms — this is a
 * thin wrapper so callers don't repeat ``ms / 1000`` everywhere.
 */
export function formatDurationMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return formatDuration(ms / 1000);
}
