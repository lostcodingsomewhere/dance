import { store, useAppStore } from "../store";

/**
 * Warnings from the last load-to-Live — in practice, the warp check.
 *
 * Why this is a sticky banner and not a 4-second toast like the SceneGrid's
 * mud nudge: a mis-warped stem is not a style opinion, it's the deck being
 * wrong. It sounds exactly like a mistake the DJ made, which is the worst
 * failure mode while learning — you can't tell tool from technique. So it
 * stays until dismissed, and it says which cell and which button fixes it.
 *
 * Rendered by the app shell so it covers every load path: the plan grid's
 * ⤒A/⤒B, ⌘K's Load, and the SceneGrid's row loads.
 */
export function LoadWarnings() {
  const lw = useAppStore((s) => s.loadWarnings);
  if (!lw) return null;
  return (
    <div
      className="mx-4 mb-2 rounded-md border border-amber-400/50 bg-amber-500/10 px-3 py-2"
      role="status"
      data-testid="load-warnings"
    >
      <div className="flex items-start gap-2">
        <span className="text-amber-300 text-sm leading-none pt-0.5">⚠</span>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-amber-200">
            {lw.title}
          </div>
          <ul className="mt-1 flex flex-col gap-0.5">
            {lw.warnings.map((w) => (
              <li key={w} className="text-[11px] leading-snug text-amber-100/90">
                {w}
              </li>
            ))}
          </ul>
        </div>
        <button
          type="button"
          onClick={() => store.clearLoadWarnings()}
          className="shrink-0 w-6 h-6 rounded text-amber-200/70 hover:text-amber-100 hover:bg-amber-500/20 transition-colors inline-flex items-center justify-center"
          title="Dismiss"
        >
          ×
        </button>
      </div>
    </div>
  );
}
