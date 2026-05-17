import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import * as api from "../api";
import { Modal } from "../components/Modal";
import { usePipelineJobs } from "../hooks/usePipelineJobs";
import {
  PIPELINE_STAGES,
  type IngestPreviewResponse,
  type IngestPreviewRow,
  type Job,
  type PipelineRecentTrack,
} from "../types";

interface ScanToast {
  text: string;
  tone: "ok" | "err";
}

const POLL_MS = 3000;
const RECENT_LIMIT = 30;
const FILTERED_LIMIT = 200;

const STAGE_COLOR: Record<string, string> = {
  pending: "bg-neutral-800 text-neutral-400",
  analyzing: "bg-amber-500/30 text-amber-200 ring-1 ring-amber-400/50",
  analyzed: "bg-amber-500/10 text-amber-300",
  separating: "bg-orange-500/30 text-orange-200 ring-1 ring-orange-400/50",
  separated: "bg-orange-500/10 text-orange-300",
  analyzing_stems: "bg-blue-500/30 text-blue-200 ring-1 ring-blue-400/50",
  stems_analyzed: "bg-blue-500/10 text-blue-300",
  detecting_regions: "bg-purple-500/30 text-purple-200 ring-1 ring-purple-400/50",
  regions_detected: "bg-purple-500/10 text-purple-300",
  embedding: "bg-cyan-500/30 text-cyan-200 ring-1 ring-cyan-400/50",
  embedded: "bg-cyan-500/10 text-cyan-300",
  complete: "bg-emerald-500/20 text-emerald-200",
  error: "bg-rose-500/30 text-rose-200 ring-1 ring-rose-400/50",
};

const ITEM_STATUS_COLOR: Record<string, string> = {
  pending: "text-neutral-500",
  ok: "text-emerald-400",
  skip: "text-neutral-400",
  fail: "text-rose-400",
};

function rowKey(row: IngestPreviewRow): string {
  return `${row.artist}|${row.title}`;
}

function formatRelative(iso: string | null): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

// =============================================================================
// Ingest panel (lives inside the modal)
// =============================================================================

function RowList({
  rows,
  selected,
  onToggle,
  emptyLabel,
}: {
  rows: IngestPreviewRow[];
  selected: Set<string>;
  onToggle: (key: string) => void;
  emptyLabel: string;
}) {
  if (rows.length === 0) {
    return <div className="text-xs text-neutral-600 italic py-2">{emptyLabel}</div>;
  }
  return (
    <ul className="text-xs max-h-56 overflow-y-auto divide-y divide-neutral-800/40 rounded border border-neutral-800/60">
      {rows.map((r) => {
        const key = rowKey(r);
        const isSelected = selected.has(key);
        return (
          <li
            key={key}
            className="flex items-center gap-2 py-1.5 px-2 hover:bg-neutral-900/60"
          >
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => onToggle(key)}
              className="accent-emerald-500"
            />
            <div className="flex-1 min-w-0">
              <span
                className={`truncate ${isSelected ? "text-neutral-200" : "text-neutral-500"}`}
              >
                <span className="text-neutral-500">{r.artist}</span> — {r.title}
              </span>
              {r.duplicate_of != null && (
                <span className="ml-2 text-neutral-600">↺ track #{r.duplicate_of}</span>
              )}
              {r.target_exists && (
                <span className="ml-2 text-neutral-600">↺ file on disk</span>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function IngestPanel({ onCommitted }: { onCommitted: () => void }) {
  const qc = useQueryClient();
  const [csvText, setCsvText] = useState("");
  const [preview, setPreview] = useState<IngestPreviewResponse | null>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const previewMutation = useMutation({
    mutationFn: () => api.ingestPreview(csvText),
    onSuccess: (data) => {
      setPreview(data);
      setPreviewErr(null);
      setSelected(new Set(data.new_rows.map(rowKey)));
    },
    onError: (err: Error) => {
      setPreviewErr(err.message);
      setPreview(null);
    },
  });

  const commitMutation = useMutation({
    mutationFn: () =>
      api.ingestCommit(csvText, { selected_keys: Array.from(selected) }),
    onSuccess: () => {
      setCsvText("");
      setPreview(null);
      setSelected(new Set());
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
      onCommitted();
    },
    onError: (err: Error) => setPreviewErr(err.message),
  });

  function toggle(key: string): void {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }
  function selectAll(rows: IngestPreviewRow[]): void {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const r of rows) next.add(rowKey(r));
      return next;
    });
  }
  function deselectAll(rows: IngestPreviewRow[]): void {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const r of rows) next.delete(rowKey(r));
      return next;
    });
  }
  async function handleFile(file: File): Promise<void> {
    const text = await file.text();
    setCsvText(text);
    setTimeout(() => previewMutation.mutate(), 0);
  }

  const selectedCount = selected.size;
  const canCommit = !!preview && selectedCount > 0 && !commitMutation.isPending;

  return (
    <div>
      <p className="text-xs text-neutral-500 mb-2">
        Export a Spotify playlist via{" "}
        <a
          href="https://exportify.net"
          target="_blank"
          rel="noopener noreferrer"
          className="text-neutral-400 underline hover:text-neutral-200"
        >
          exportify.net
        </a>
        , then paste or drop the CSV here.
      </p>

      <textarea
        value={csvText}
        onChange={(e) => setCsvText(e.target.value)}
        placeholder="Paste Exportify CSV here, or drop a .csv file…"
        rows={4}
        className="w-full bg-neutral-900 border border-neutral-800 rounded p-2 font-mono text-xs text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-neutral-600"
        onDragOver={(e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
        }}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      />

      <div className="flex items-center gap-3 mt-3">
        <button
          type="button"
          disabled={!csvText.trim() || previewMutation.isPending}
          onClick={() => previewMutation.mutate()}
          className="px-4 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-100 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {previewMutation.isPending ? "Parsing…" : "Preview"}
        </button>
        <label className="text-xs text-neutral-400 cursor-pointer hover:text-neutral-200">
          or pick file…
          <input
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
        </label>
        {previewErr && <span className="text-rose-400 text-xs">{previewErr}</span>}
      </div>

      {preview && (
        <div className="mt-4 space-y-3">
          <div className="flex gap-2 flex-wrap text-xs">
            <span className="rounded bg-emerald-500/20 text-emerald-200 px-2 py-1">
              {preview.new_rows.length} new
            </span>
            <span className="rounded bg-neutral-800 text-neutral-400 px-2 py-1">
              {preview.duplicates.length} duplicate
            </span>
            <span className="rounded bg-amber-500/20 text-amber-200 px-2 py-1">
              {preview.parse_errors.length} parse error
            </span>
            <span className="rounded bg-cyan-500/20 text-cyan-200 px-2 py-1">
              {selectedCount} selected
            </span>
          </div>

          {preview.parse_errors.length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-amber-300">
                Show parse errors ({preview.parse_errors.length})
              </summary>
              <ul className="mt-1 ml-4 text-neutral-400 list-disc">
                {preview.parse_errors.slice(0, 10).map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </details>
          )}

          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-emerald-300">
                New ({preview.new_rows.length})
              </span>
              <span className="text-xs text-neutral-500 space-x-2">
                <button
                  type="button"
                  onClick={() => selectAll(preview.new_rows)}
                  className="hover:text-neutral-200"
                >
                  all
                </button>
                <button
                  type="button"
                  onClick={() => deselectAll(preview.new_rows)}
                  className="hover:text-neutral-200"
                >
                  none
                </button>
              </span>
            </div>
            <RowList
              rows={preview.new_rows}
              selected={selected}
              onToggle={toggle}
              emptyLabel="No new rows — everything is a duplicate."
            />
          </div>

          {preview.duplicates.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-neutral-400">
                  Duplicates ({preview.duplicates.length})
                </span>
                <span className="text-xs text-neutral-500 space-x-2">
                  <button
                    type="button"
                    onClick={() => selectAll(preview.duplicates)}
                    className="hover:text-neutral-200"
                  >
                    all
                  </button>
                  <button
                    type="button"
                    onClick={() => deselectAll(preview.duplicates)}
                    className="hover:text-neutral-200"
                  >
                    none
                  </button>
                </span>
              </div>
              <RowList
                rows={preview.duplicates}
                selected={selected}
                onToggle={toggle}
                emptyLabel=""
              />
            </div>
          )}

          <div className="flex items-center gap-3 pt-2 border-t border-neutral-800">
            <button
              type="button"
              disabled={!canCommit}
              onClick={() => commitMutation.mutate()}
              className="ml-auto px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-neutral-950 text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {commitMutation.isPending
                ? "Starting…"
                : `Download ${selectedCount} track${selectedCount === 1 ? "" : "s"}`}
            </button>
          </div>
          <p className="text-xs text-neutral-500 pt-1">
            Files land in <code>~/Music/DJ/library/</code>. After downloads
            finish, click <em>Process now</em> in the Pipeline header to
            ingest them.
          </p>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Job card
// =============================================================================

function JobCard({ job }: { job: Job }) {
  const finished = job.counts.ok + job.counts.skip + job.counts.fail;
  const pct = job.total > 0 ? Math.round((finished / job.total) * 100) : 0;
  const isActive = job.status === "queued" || job.status === "running";
  const showAllItems = job.kind === "pipeline_run";
  const itemsToShow = showAllItems
    ? job.items
    : job.items.filter((it) => it.status !== "pending").slice(-8).reverse();

  return (
    <div className="rounded-lg border border-neutral-800 p-3 bg-neutral-950">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-neutral-500">{job.id}</span>
            <span
              className={`text-xs px-2 py-0.5 rounded ${
                job.status === "done"
                  ? "bg-emerald-500/20 text-emerald-300"
                  : job.status === "error"
                    ? "bg-rose-500/20 text-rose-300"
                    : "bg-amber-500/30 text-amber-200 ring-1 ring-amber-400/50"
              }`}
            >
              {job.status}
            </span>
            <span className="text-xs text-neutral-400">{job.kind}</span>
          </div>
          <div className="text-xs text-neutral-500 mt-1">
            {job.counts.ok} ok · {job.counts.skip} skip · {job.counts.fail} fail
            {" · "}
            {job.counts.pending} pending of {job.total}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-2xl text-neutral-100 tabular-nums">
            {pct}%
          </div>
        </div>
      </div>

      <div className="h-1.5 rounded bg-neutral-800 overflow-hidden mt-2">
        <div
          className={`h-full transition-all duration-500 ${
            isActive
              ? "bg-amber-400"
              : job.status === "error"
                ? "bg-rose-500"
                : "bg-emerald-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {job.error && <div className="text-xs text-rose-400 mt-2">{job.error}</div>}

      {itemsToShow.length > 0 && (
        <ul className="mt-2 text-xs space-y-0.5">
          {itemsToShow.map((it, i) => (
            <li key={i} className="truncate">
              <span className={`${ITEM_STATUS_COLOR[it.status]} font-mono mr-2`}>
                {it.status === "ok"
                  ? "✓"
                  : it.status === "fail"
                    ? "✗"
                    : it.status === "skip"
                      ? "—"
                      : "·"}
              </span>
              <span className="text-neutral-300">{it.label}</span>
              {it.message && (
                <span className="text-neutral-500 ml-2">{it.message}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// =============================================================================
// Main view
// =============================================================================

export function PipelineOps() {
  const qc = useQueryClient();
  const [ingestOpen, setIngestOpen] = useState(false);
  /** When set, the activity table filters to this state. Driven by tile clicks. */
  const [filterState, setFilterState] = useState<string | null>(null);
  const [scanToast, setScanToast] = useState<ScanToast | null>(null);

  const status = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: api.getPipelineStatus,
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
  });

  const recent = useQuery({
    queryKey: ["pipeline-recent", filterState],
    queryFn: () =>
      api.getPipelineRecent(filterState ? FILTERED_LIMIT : RECENT_LIMIT, filterState),
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
  });

  const watch = useQuery({
    queryKey: ["pipeline-watch"],
    queryFn: api.getWatchState,
    refetchInterval: 30000,
  });

  const watchMutation = useMutation({
    mutationFn: api.setWatchState,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["pipeline-watch"] }),
  });

  const scanMutation = useMutation({
    mutationFn: api.scanLibrary,
    onSuccess: (data) => {
      setScanToast({
        text: `Scanned: ${data.new} new · ${data.updated} updated · ${data.unchanged} unchanged${
          data.errors ? ` · ${data.errors} errors` : ""
        }`,
        tone: data.errors ? "err" : "ok",
      });
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
      qc.invalidateQueries({ queryKey: ["pipeline-recent", filterState] });
      setTimeout(() => setScanToast(null), 5000);
    },
    onError: (err: Error) => {
      setScanToast({ text: err.message, tone: "err" });
      setTimeout(() => setScanToast(null), 5000);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteTrack,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
      qc.invalidateQueries({ queryKey: ["pipeline-recent", filterState] });
    },
  });

  function confirmDelete(track: PipelineRecentTrack): void {
    const label = `${track.title ?? `(untitled)`} — ${track.artist ?? "?"}`;
    if (window.confirm(`Delete track #${track.id} (${label})?\n\nThis removes the row from the DB and any stems/analysis. The audio file in your library stays on disk.`)) {
      deleteMutation.mutate(track.id);
    }
  }

  const jobs = usePipelineJobs();
  const sortedJobs = useMemo(
    () =>
      Object.values(jobs)
        .sort((a, b) => b.created_at - a.created_at)
        .slice(0, 5),
    [jobs],
  );
  const activePipelineRun = useMemo(
    () =>
      Object.values(jobs).find(
        (j) =>
          j.kind === "pipeline_run" &&
          (j.status === "queued" || j.status === "running"),
      ),
    [jobs],
  );

  const processMutation = useMutation({
    mutationFn: api.triggerPipelineProcess,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["pipeline-status"] }),
  });

  // While a pipeline run is in flight, refresh faster.
  useEffect(() => {
    if (activePipelineRun) {
      const id = setInterval(() => {
        qc.invalidateQueries({ queryKey: ["pipeline-status"] });
        qc.invalidateQueries({ queryKey: ["pipeline-recent", filterState] });
      }, 1000);
      return () => clearInterval(id);
    }
  }, [activePipelineRun, filterState, qc]);

  const counts = status.data?.counts ?? {};
  const total = status.data?.total ?? 0;
  const weighted = status.data?.weighted_progress ?? 0;
  const inProgress = !!status.data?.in_progress;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-neutral-800 flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Pipeline</h1>
          <p className="text-neutral-500 text-sm">
            {inProgress ? "Processing…" : "Idle"} · {total} tracks ·{" "}
            <span className="text-neutral-300">
              {weighted.toFixed(1)}% through pipeline
            </span>
            {status.data && status.data.complete > 0 && (
              <span className="text-neutral-500">
                {" "}
                · {status.data.complete} fully complete
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label
            className={`px-3 py-2 rounded-lg text-sm font-semibold cursor-pointer flex items-center gap-2 ${
              watch.data?.enabled
                ? "bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-400/40"
                : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
            }`}
            title={
              watch.data?.enabled
                ? `Watching — auto-triggers Process every ${watch.data.interval_seconds}s if work is pending`
                : "Auto-run pipeline when there's work to do"
            }
          >
            <input
              type="checkbox"
              checked={!!watch.data?.enabled}
              onChange={(e) => watchMutation.mutate(e.target.checked)}
              disabled={watchMutation.isPending}
              className="accent-emerald-500"
            />
            <span>
              Watch{watch.data?.enabled ? ` · ${watch.data.interval_seconds}s` : ""}
            </span>
          </label>
          <button
            type="button"
            onClick={() => scanMutation.mutate()}
            disabled={scanMutation.isPending}
            className="px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-100 text-sm font-semibold disabled:opacity-40"
            title="Scan library_dir for new files. Doesn't run stages — use Process now after."
          >
            {scanMutation.isPending ? "Scanning…" : "Scan"}
          </button>
          <button
            type="button"
            onClick={() => setIngestOpen(true)}
            className="px-3 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-100 text-sm font-semibold"
          >
            Ingest CSV…
          </button>
          <button
            type="button"
            onClick={() => processMutation.mutate()}
            disabled={!!activePipelineRun || processMutation.isPending}
            className={`px-4 py-2 rounded-lg font-semibold text-sm ${
              activePipelineRun
                ? "bg-amber-500/30 text-amber-200 ring-1 ring-amber-400/50 cursor-not-allowed"
                : "bg-emerald-500 hover:bg-emerald-400 text-neutral-950"
            } disabled:opacity-70`}
            title={
              activePipelineRun
                ? "A pipeline run is already in flight"
                : "Run dance process (ingest + every stage)"
            }
          >
            {activePipelineRun
              ? "Pipeline running…"
              : processMutation.isPending
                ? "Starting…"
                : "Process now"}
          </button>
          <div className="flex items-center gap-2 text-xs text-neutral-400 ml-2">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                status.isFetching
                  ? "bg-emerald-400 animate-pulse"
                  : "bg-neutral-600"
              }`}
            />
            Auto-refresh {POLL_MS / 1000}s
          </div>
        </div>
      </div>

      {scanToast && (
        <div
          className={`px-6 py-2 text-sm ${
            scanToast.tone === "ok"
              ? "bg-emerald-500/10 text-emerald-200"
              : "bg-rose-500/10 text-rose-200"
          }`}
        >
          {scanToast.text}
        </div>
      )}

      {/* Weighted progress bar */}
      <div className="px-6 pt-4">
        <div className="h-2 rounded bg-neutral-800 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${weighted}%` }}
          />
        </div>
      </div>

      {/* State grid — clickable tiles */}
      <div className="px-6 py-4 grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-2">
        {PIPELINE_STAGES.map(({ key, label }) => {
          const count = counts[key] ?? 0;
          const dim = count === 0 ? "opacity-40" : "";
          const isSelected = filterState === key;
          return (
            <button
              type="button"
              key={key}
              onClick={() =>
                setFilterState((prev) => (prev === key ? null : key))
              }
              className={`text-left rounded-lg px-3 py-2 transition-all ${
                STAGE_COLOR[key] ?? "bg-neutral-800"
              } ${dim} ${
                isSelected
                  ? "ring-2 ring-neutral-100 scale-[1.02]"
                  : "hover:ring-1 hover:ring-neutral-600 cursor-pointer"
              }`}
              title={
                count > 0
                  ? `Click to ${isSelected ? "clear filter" : `show all ${count} ${key} tracks`}`
                  : "0 tracks"
              }
            >
              <div className="text-[10px] uppercase tracking-wider opacity-80">
                {label}
              </div>
              <div className="font-mono text-2xl tabular-nums">{count}</div>
            </button>
          );
        })}
      </div>

      {/* Two columns: left = jobs, right = activity (full width when no jobs) */}
      <div
        className={`px-6 pb-4 flex-1 min-h-0 grid grid-cols-1 ${
          sortedJobs.length > 0 ? "lg:grid-cols-2" : ""
        } gap-4`}
      >
        {sortedJobs.length > 0 && (
          <div className="flex flex-col gap-3 min-h-0 overflow-y-auto">
            <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider">
              Jobs
            </h2>
            <div className="space-y-2">
              {sortedJobs.map((job) => (
                <JobCard key={job.id} job={job} />
              ))}
            </div>
          </div>
        )}

        <div className="flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider">
              {filterState
                ? `Tracks in “${filterState}”`
                : "Recent activity"}
            </h2>
            {filterState && (
              <button
                type="button"
                onClick={() => setFilterState(null)}
                className="text-xs text-neutral-500 hover:text-neutral-200"
              >
                clear filter ×
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto rounded-lg border border-neutral-800">
            {recent.isLoading && (
              <div className="p-4 text-neutral-500 text-sm">Loading…</div>
            )}
            {recent.error && (
              <div className="p-4 text-rose-400 text-sm">
                Failed to load. Backend up?
              </div>
            )}
            {recent.data && recent.data.length === 0 && (
              <div className="p-4 text-neutral-500 text-sm">
                {filterState
                  ? `No tracks in “${filterState}”.`
                  : "No tracks yet. Drop audio in ~/Music/DJ/library/ and click Process now."}
              </div>
            )}
            {recent.data && recent.data.length > 0 && (
              <>
                <table className="w-full text-sm">
                  <thead className="text-xs text-neutral-500 uppercase tracking-wider sticky top-0 bg-neutral-950">
                    <tr>
                      <th className="text-left px-3 py-2 w-12">#</th>
                      <th className="text-left px-3 py-2">Track</th>
                      <th className="text-left px-3 py-2 w-32">State</th>
                      <th className="text-right px-3 py-2 w-20">Updated</th>
                      <th className="w-8"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.data.map((t: PipelineRecentTrack) => (
                      <tr
                        key={t.id}
                        className="border-t border-neutral-800/50 hover:bg-neutral-900 group"
                      >
                        <td className="px-3 py-2 text-neutral-500 font-mono">
                          {t.id}
                        </td>
                        <td className="px-3 py-2">
                          <div className="truncate max-w-md">
                            <span className="font-medium text-neutral-100">
                              {t.title ?? "(untitled)"}
                            </span>{" "}
                            <span className="text-neutral-500">
                              — {t.artist ?? "?"}
                            </span>
                          </div>
                          {t.error_message && (
                            <div className="text-rose-400 text-xs mt-0.5 truncate">
                              {t.error_message}
                            </div>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <button
                            type="button"
                            onClick={() => setFilterState(t.state)}
                            className={`inline-block px-2 py-0.5 rounded text-xs hover:ring-1 hover:ring-neutral-500 cursor-pointer ${
                              STAGE_COLOR[t.state] ?? "bg-neutral-800"
                            }`}
                            title={`Filter to ${t.state}`}
                          >
                            {t.state}
                          </button>
                        </td>
                        <td className="px-3 py-2 text-right text-neutral-500 text-xs tabular-nums">
                          {formatRelative(t.updated_at)}
                        </td>
                        <td className="pr-3">
                          <button
                            type="button"
                            onClick={() => confirmDelete(t)}
                            disabled={
                              deleteMutation.isPending &&
                              deleteMutation.variables === t.id
                            }
                            className="opacity-0 group-hover:opacity-100 text-neutral-500 hover:text-rose-400 text-lg leading-none transition-opacity disabled:opacity-40"
                            title="Delete this track from the DB (audio file stays on disk)"
                            aria-label={`Delete track ${t.id}`}
                          >
                            ×
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="text-xs text-neutral-500 text-center py-2 border-t border-neutral-800/50">
                  Showing {recent.data.length}
                  {filterState
                    ? ` of ${counts[filterState] ?? "?"}`
                    : " most recent"}
                  {!filterState && total > RECENT_LIMIT && (
                    <span> · click a tile above to see all of one state</span>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <Modal
        open={ingestOpen}
        onClose={() => setIngestOpen(false)}
        title="Ingest from CSV"
        widthClass="max-w-2xl"
      >
        <IngestPanel onCommitted={() => setIngestOpen(false)} />
      </Modal>
    </div>
  );
}
