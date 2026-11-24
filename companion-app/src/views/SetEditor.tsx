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
import { useExportSetToLive } from "../hooks/useSetExport";
import { useActivateSet, useActiveSet, useCreateSet } from "../hooks/useSets";
import { store } from "../store";
import type { DanceSet } from "../types";

export function SetEditor() {
  const active = useActiveSet();
  const set = active.data;

  if (active.isLoading) return <CenteredMsg>Loading…</CenteredMsg>;
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
  const exportToLive = useExportSetToLive();
  const exportButton = useRef<HTMLButtonElement>(null);
  const exportFinishedForThisSet =
    exportToLive.variables === set.id && (exportToLive.isSuccess || exportToLive.isError);

  function dismissExportResult() {
    exportToLive.reset();
    exportButton.current?.focus();
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
          ref={exportButton}
          type="button"
          onClick={() => exportToLive.mutate(set.id)}
          disabled={exportToLive.isPending}
          title="Write this set's whole plan as one Ableton Live Set"
          className="h-7 px-3 rounded bg-violet-700 hover:bg-violet-600 text-white text-xs shrink-0 disabled:opacity-50"
        >
          {exportToLive.isPending ? "Exporting…" : "Export to Live"}
        </button>
      </div>
      {exportFinishedForThisSet && (
        <SetExportStrip
          key={exportToLive.submittedAt}
          setId={set.id}
          exported={exportToLive.data}
          failure={exportToLive.error}
          onDismiss={dismissExportResult}
        />
      )}
    </>
  );
}

function NoSetState() {
  const create = useCreateSet();
  const activate = useActivateSet();
  const pending = create.isPending || activate.isPending;
  function bootstrap() {
    create.mutate(
      { name: "My set" },
      { onSuccess: (created) => activate.mutate(created.id) },
    );
  }
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-8 py-12 text-center gap-4">
      <h1 className="text-xl font-medium text-neutral-100">No active set yet</h1>
      <p className="text-sm text-neutral-500 max-w-lg leading-relaxed">
        A set is a plan — per-role queues of the stems you'll bring in (drums,
        bass, vocals, other, song). Create one, then queue picks from the recs
        or ⌘K. The same grid drives the Booth live, where you load picks onto a
        deck.
      </p>
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
