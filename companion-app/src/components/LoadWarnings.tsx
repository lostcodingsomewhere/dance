import { store, useAppStore, type LoadWarning } from "../store";

const MOST_SHOWN = 4;

/**
 * Warnings from loads and commands that failed, newest first — in practice,
 * mostly the warp check.
 *
 * Why this is a sticky banner and not a 4-second toast like the SceneGrid's
 * mud nudge: a mis-warped stem is not a style opinion, it's the deck being
 * wrong. It sounds exactly like a mistake the DJ made, which is the worst
 * failure mode while learning — you can't tell tool from technique. So each
 * line stays until dismissed, and it says which cell and which button fixes it.
 *
 * Rendered by the app shell so it covers every load path: the plan grid's
 * ⤒A/⤒B, ⌘K's Load, and the SceneGrid's row loads.
 */
export function LoadWarnings() {
  const all = useAppStore((s) => s.loadWarnings);
  if (all.length === 0) return null;
  const shown = all.slice(0, MOST_SHOWN);
  const hidden = all.length - shown.length;
  return (
    <div
      className="mx-4 mb-2 rounded-md border border-amber-400/50 bg-amber-500/10 px-3 py-2"
      role="status"
      aria-label="Warnings"
      data-testid="load-warnings"
    >
      <ul className="flex flex-col gap-1.5">
        {shown.map((warning) => (
          <WarningLine key={warning.id} warning={warning} />
        ))}
      </ul>
      {all.length > 1 && (
        <div className="mt-1.5 flex items-center justify-between pl-6 text-[10px] font-semibold uppercase tracking-wider text-amber-200/70">
          <span>{hidden > 0 && `+${hidden} more`}</span>
          <button
            type="button"
            onClick={() => store.clearLoadWarnings()}
            className="rounded px-1.5 py-0.5 uppercase tracking-wider hover:text-amber-100 hover:bg-amber-500/20 transition-colors"
          >
            Dismiss all
          </button>
        </div>
      )}
    </div>
  );
}

function WarningLine({ warning }: { warning: LoadWarning }) {
  const fitsOnOneLine = warning.warnings.length === 1;
  return (
    <li className="flex items-start gap-2">
      <span aria-hidden className="text-amber-300 text-sm leading-none pt-0.5">
        ⚠
      </span>
      <div className="min-w-0 flex-1 text-[11px] leading-snug text-amber-100/90">
        {fitsOnOneLine ? (
          <p>
            <span className="font-semibold text-amber-200">{warning.title}:</span>{" "}
            {warning.warnings[0]}
          </p>
        ) : (
          <>
            <p className="font-semibold text-amber-200">{warning.title}</p>
            {warning.warnings.map((line, at) => (
              <p key={at}>{line}</p>
            ))}
          </>
        )}
      </div>
      {warning.times > 1 && (
        <span className="shrink-0 rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-100">
          {warning.times} times
        </span>
      )}
      <button
        type="button"
        onClick={() => store.dismissLoadWarning(warning.id)}
        className="shrink-0 w-6 h-6 rounded text-amber-200/70 hover:text-amber-100 hover:bg-amber-500/20 transition-colors inline-flex items-center justify-center"
        aria-label="Dismiss"
        title="Dismiss"
      >
        ×
      </button>
    </li>
  );
}
