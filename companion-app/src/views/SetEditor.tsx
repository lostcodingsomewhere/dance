/**
 * Set editor — the plan. A set IS its plan: per-role queues (drums · bass ·
 * vocals · other · song) of the stems you intend to bring in. This is the one
 * place you author it: queue picks from the per-role recs (scored against the
 * rest of the plan + the plan's journey) or ⌘K, then perform them in the
 * Booth, where the very same RoleColumnsGrid renders live (recs tail what's
 * playing) with ⤒A/⤒B deck-load.
 */

import type React from "react";
import { useRef } from "react";

import { CueStrip } from "../components/CueStrip";
import { RoleColumnsGrid } from "../components/RoleColumnsGrid";
import { SetExportStrip } from "../components/SetExportStrip";
import { SetMenu } from "../components/SetMenu";
import { useOpenSetInLive } from "../hooks/useSetExport";
import { useActivateSet, useActiveSet, useCreateSet, useSets } from "../hooks/useSets";
import { roleLabel } from "../lib/roles";
import { store } from "../store";
import { PLAN_ROLES, type DanceSet, type SetSummary } from "../types";

const COLUMN_LABELS = PLAN_ROLES.map((role) => roleLabel(role === "song" ? "mix" : role)).join(", ");
const CENTERED_PAGE =
  "flex-1 flex flex-col items-center justify-center px-8 py-12 text-center gap-4";

export function SetEditor() {
  const active = useActiveSet();
  const set = active.data;

  if (active.isLoading) return <CenteredMsg>Loading…</CenteredMsg>;
  if (!set && active.isError) {
    return <ActiveSetUnknown retrying={active.isFetching} onRetry={() => active.refetch()} />;
  }
  if (!set) return <NoSetState />;

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <Header set={set} />
      {/* Preview waveform/scrubber — only renders while auditioning a pick
          (▶ on any card). Lets you scrub the cue the same as in the Booth. */}
      <CueStrip />
      <div className="flex-1 px-6 py-4 min-h-0 overflow-y-auto">
        <RoleColumnsGrid setId={set.id} mode="plan" />
      </div>
    </div>
  );
}

function Header({ set }: { set: DanceSet }) {
  const openSetInLive = useOpenSetInLive();
  const openButton = useRef<HTMLButtonElement>(null);
  const finishedForThisSet =
    openSetInLive.variables === set.id && (openSetInLive.isSuccess || openSetInLive.isError);

  function dismissResult() {
    openSetInLive.reset();
    openButton.current?.focus();
  }

  return (
    <>
      <div className="flex items-center gap-3 px-6 py-3 border-b border-neutral-800">
        <button
          type="button"
          onClick={() => store.setView("booth")}
          className="text-xs text-neutral-500 hover:text-neutral-100 shrink-0"
        >
          ← Booth
        </button>
        <span className="text-[10px] uppercase tracking-wider text-neutral-500 shrink-0">
          Set
        </span>
        <SetMenu set={set} variant="full" />
        <span className="ml-auto hidden lg:inline text-[11px] text-neutral-500 shrink-0">
          queue stems per role · ＋ from recs or ⌘K
        </span>
        <button
          ref={openButton}
          type="button"
          onClick={() => openSetInLive.mutate(set.id)}
          disabled={openSetInLive.isPending}
          title="Write this plan as a new Live Set file and open it in Ableton Live"
          className="h-7 px-3 rounded bg-violet-700 hover:bg-violet-600 text-white text-xs shrink-0 disabled:opacity-50"
        >
          {openSetInLive.isPending ? "Opening…" : "Open in Live"}
        </button>
      </div>
      {finishedForThisSet && (
        <SetExportStrip
          key={openSetInLive.submittedAt}
          setId={set.id}
          sent={openSetInLive.data}
          failure={openSetInLive.error}
          onDismiss={dismissResult}
        />
      )}
    </>
  );
}

function ActiveSetUnknown({ retrying, onRetry }: { retrying: boolean; onRetry: () => void }) {
  return (
    <div className={CENTERED_PAGE}>
      <div role="alert" className="flex flex-col items-center gap-4">
        <h1 className="text-xl font-medium text-neutral-100">Could not reach the server</h1>
        <p className="text-sm text-neutral-400 max-w-lg leading-relaxed">
          The app cannot tell which set is active. Check that the server is running, then retry.
        </p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        disabled={retrying}
        className="h-9 px-4 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-100 text-sm disabled:opacity-50"
      >
        {retrying ? "Retrying…" : "Retry"}
      </button>
    </div>
  );
}

function picksIn(set: SetSummary): string {
  return `${set.track_count} pick${set.track_count === 1 ? "" : "s"}`;
}

function ExistingSets({
  sets,
  disabled,
  onOpen,
}: {
  sets: SetSummary[];
  disabled: boolean;
  onOpen: (setId: number) => void;
}) {
  return (
    <div className="w-full max-w-md">
      <h2 className="mb-1 text-[10px] uppercase tracking-wider text-neutral-500">Your sets</h2>
      <ul
        aria-label="Your sets"
        className="divide-y divide-neutral-800 rounded-md border border-neutral-800 text-left"
      >
        {sets.map((set) => (
          <li key={set.id} className="flex items-center gap-3 px-3 py-2">
            <span className="min-w-0 flex-1 truncate text-sm text-neutral-100">{set.name}</span>
            <span className="shrink-0 text-xs tabular-nums text-neutral-400">{picksIn(set)}</span>
            <button
              type="button"
              aria-label={`Open ${set.name}`}
              onClick={() => onOpen(set.id)}
              disabled={disabled}
              className="h-7 shrink-0 rounded bg-neutral-800 px-3 text-xs text-neutral-100 hover:bg-neutral-700 disabled:opacity-50"
            >
              Open
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function NoSetState() {
  const create = useCreateSet();
  const activate = useActivateSet();
  const sets = useSets();
  const existing = sets.data ?? [];
  const pending = create.isPending || activate.isPending;
  function bootstrap() {
    create.mutate(
      { name: "My set" },
      { onSuccess: (created) => activate.mutate(created.id) },
    );
  }
  return (
    <div className={CENTERED_PAGE}>
      <h1 className="text-xl font-medium text-neutral-100">No active set yet</h1>
      <p className="text-sm text-neutral-500 max-w-lg leading-relaxed">
        A set is a plan — per-role queues of the stems you'll bring in ({COLUMN_LABELS}). Create
        one, then queue picks from the recs or ⌘K. The same grid drives the Booth live, where you
        load picks onto a deck.
      </p>
      {existing.length > 0 && (
        <ExistingSets sets={existing} disabled={pending} onOpen={(id) => activate.mutate(id)} />
      )}
      {sets.isError && sets.data == null && (
        <p role="alert" className="text-sm text-rose-300">
          Could not reach the server, so your sets are not listed.{" "}
          <button
            type="button"
            onClick={() => sets.refetch()}
            className="underline hover:text-rose-100"
          >
            Retry
          </button>
        </p>
      )}
      {(create.isError || activate.isError) && (
        <p role="alert" className="text-sm text-rose-300">
          Could not reach the server. Try again.
        </p>
      )}
      <button
        type="button"
        onClick={bootstrap}
        disabled={pending}
        className="h-9 px-4 rounded bg-violet-700 hover:bg-violet-600 text-white text-sm disabled:opacity-50"
      >
        {pending ? "…" : "+ create empty set"}
      </button>
    </div>
  );
}

function CenteredMsg({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex-1 flex items-center justify-center text-sm text-neutral-500">
      {children}
    </div>
  );
}
