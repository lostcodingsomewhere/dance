import { serverMessage, type SkippedPick } from "../api";
import { useRevealInFinder, type SetSentToLive } from "../hooks/useSetExport";
import { useSetPlan } from "../hooks/useSetPlan";
import { roleLabel } from "../lib/roles";
import type { SetPlan } from "../types";

const STRIP_LABEL = "shrink-0 text-[10px] uppercase tracking-wider";
const ACTION_BUTTON =
  "h-6 shrink-0 rounded border border-neutral-800 bg-neutral-900 px-2 text-[11px] text-neutral-300 transition-colors hover:bg-neutral-800 disabled:opacity-50";

function counted(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function fileNameOf(path: string): string {
  return path.split("/").pop() || path;
}

function titlesByTrackId(plan: SetPlan | null | undefined): Map<number, string> {
  const titles = new Map<number, string>();
  for (const item of Object.values(plan?.queues ?? {}).flat()) {
    if (item.title) titles.set(item.track_id, item.title);
  }
  return titles;
}

function skippedLine(pick: SkippedPick, titles: Map<number, string>): string {
  const title = titles.get(pick.track_id) ?? `track ${pick.track_id}`;
  return `${title} (${roleLabel(pick.role).toLowerCase()}): ${pick.reason}`;
}

interface Props {
  setId: number;
  sent: SetSentToLive | undefined;
  failure: Error | null;
  onDismiss: () => void;
}

export function SetExportStrip({ setId, sent, failure, onDismiss }: Props) {
  return (
    <section
      aria-label="Live Set result"
      className="flex items-start gap-3 border-b border-neutral-800 bg-neutral-900/50 px-6 py-2 text-xs"
    >
      <div className="min-w-0 flex-1">
        {sent && <WrittenSet setId={setId} sent={sent} />}
        {failure && (
          <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className={`${STRIP_LABEL} text-rose-400/80`}>Not opened</span>
            <Failure error={failure} />
          </p>
        )}
      </div>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss Live Set result"
        title="Dismiss"
        className="-my-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded text-sm text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-100"
      >
        ×
      </button>
    </section>
  );
}

function WrittenSet({ setId, sent }: { setId: number; sent: SetSentToLive }) {
  const revealInFinder = useRevealInFinder();
  const { exported, openedIn, openFailure } = sent;

  return (
    <>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        {openedIn ? (
          <span role="status" className="shrink-0 text-emerald-400">
            {`Opened in ${openedIn}`}
          </span>
        ) : (
          <span className={`${STRIP_LABEL} text-amber-300`}>Written, not opened</span>
        )}
        <span title={exported.out_path} className="max-w-full truncate font-medium text-neutral-100">
          {fileNameOf(exported.out_path)}
        </span>
        <span className="shrink-0 tabular-nums text-neutral-500">
          {`${counted(exported.page_count, "page")} · ${counted(exported.clip_count, "clip")}`}
        </span>
        <button
          type="button"
          onClick={() => revealInFinder.mutate(exported.out_path)}
          className={ACTION_BUTTON}
        >
          Show in Finder
        </button>
        {openFailure && <Failure error={openFailure} />}
        {revealInFinder.isError && <Failure error={revealInFinder.error} />}
      </div>
      {exported.skipped.length > 0 && <SkippedPicks setId={setId} skipped={exported.skipped} />}
    </>
  );
}

function SkippedPicks({ setId, skipped }: { setId: number; skipped: SkippedPick[] }) {
  const plan = useSetPlan(setId);
  const titles = titlesByTrackId(plan.data);

  return (
    <details className="mt-1.5">
      <summary className="w-fit cursor-pointer text-amber-300 hover:text-amber-200">
        {`${counted(skipped.length, "pick")} left out`}
      </summary>
      <ul className="mt-1 space-y-0.5 pl-3.5 text-neutral-400">
        {skipped.map((pick, index) => (
          <li key={index}>{skippedLine(pick, titles)}</li>
        ))}
      </ul>
    </details>
  );
}

function Failure({ error }: { error: Error }) {
  return (
    <span role="alert" className="text-rose-400">
      {serverMessage(error)}
    </span>
  );
}
