import { useQuery } from "@tanstack/react-query";
import * as api from "../api";
import { PIPELINE_STAGES, type PipelineRecentTrack } from "../types";

const POLL_MS = 3000;

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

function formatRelative(iso: string | null): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function PipelineOps() {
  const status = useQuery({
    queryKey: ["pipeline-status"],
    queryFn: api.getPipelineStatus,
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
  });

  const recent = useQuery({
    queryKey: ["pipeline-recent"],
    queryFn: () => api.getPipelineRecent(30),
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: true,
  });

  const counts = status.data?.counts ?? {};
  const total = status.data?.total ?? 0;
  const completePct = total > 0 ? Math.round(((counts.complete ?? 0) / total) * 100) : 0;
  const inProgress = !!status.data?.in_progress;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-neutral-800 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Pipeline</h1>
          <p className="text-neutral-500 text-sm">
            {inProgress ? "Processing…" : "Idle"} · {total} tracks · {completePct}% complete
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-neutral-400">
          <span className={`inline-block w-2 h-2 rounded-full ${
            status.isFetching ? "bg-emerald-400 animate-pulse" : "bg-neutral-600"
          }`} />
          Auto-refresh {POLL_MS / 1000}s
        </div>
      </div>

      {/* Progress bar */}
      <div className="px-6 pt-4">
        <div className="h-2 rounded bg-neutral-800 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${completePct}%` }}
          />
        </div>
      </div>

      {/* State grid */}
      <div className="px-6 py-4 grid grid-cols-[repeat(auto-fit,minmax(160px,1fr))] gap-2">
        {PIPELINE_STAGES.map(({ key, label }) => {
          const count = counts[key] ?? 0;
          const dim = count === 0 ? "opacity-40" : "";
          return (
            <div
              key={key}
              className={`rounded-lg px-3 py-2 ${STAGE_COLOR[key] ?? "bg-neutral-800"} ${dim}`}
            >
              <div className="text-[10px] uppercase tracking-wider opacity-80">
                {label}
              </div>
              <div className="font-mono text-2xl tabular-nums">{count}</div>
            </div>
          );
        })}
      </div>

      {/* Recent activity */}
      <div className="px-6 pb-4 flex-1 min-h-0 flex flex-col">
        <h2 className="text-sm font-semibold text-neutral-400 uppercase tracking-wider mb-2">
          Recent activity
        </h2>
        <div className="flex-1 overflow-y-auto rounded-lg border border-neutral-800">
          {recent.isLoading && (
            <div className="p-4 text-neutral-500 text-sm">Loading…</div>
          )}
          {recent.error && (
            <div className="p-4 text-rose-400 text-sm">
              Failed to load recent activity. Backend up?
            </div>
          )}
          {recent.data && recent.data.length === 0 && (
            <div className="p-4 text-neutral-500 text-sm">
              No tracks yet. Drop audio in <code>~/Music/DJ/library/</code> and run{" "}
              <code>dance process</code>.
            </div>
          )}
          {recent.data && recent.data.length > 0 && (
            <table className="w-full text-sm">
              <thead className="text-xs text-neutral-500 uppercase tracking-wider">
                <tr>
                  <th className="text-left px-3 py-2 w-12">#</th>
                  <th className="text-left px-3 py-2">Track</th>
                  <th className="text-left px-3 py-2 w-40">State</th>
                  <th className="text-right px-3 py-2 w-24">Updated</th>
                </tr>
              </thead>
              <tbody>
                {recent.data.map((t: PipelineRecentTrack) => (
                  <tr
                    key={t.id}
                    className="border-t border-neutral-800/50 hover:bg-neutral-900"
                  >
                    <td className="px-3 py-2 text-neutral-500 font-mono">{t.id}</td>
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
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-xs ${
                          STAGE_COLOR[t.state] ?? "bg-neutral-800"
                        }`}
                      >
                        {t.state}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-neutral-500 text-xs tabular-nums">
                      {formatRelative(t.updated_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
