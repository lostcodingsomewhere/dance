/**
 * Compact host-deps health indicator.
 *
 * Polls ``/api/v1/health/deps`` every 30 s and renders a small dot in
 * MasterStrip: green when everything's ready, amber when an optional
 * dep is missing (Spotify creds, cookies), red when a required one is
 * (yt-dlp, ffmpeg). Click to open a checklist tooltip with the remediation
 * hint per row.
 *
 * The single most useful thing this surfaces is "you can't ingest yet" —
 * before the user tries to add a song from Spotify and watches it flip
 * to ⚠ failed. Diagnose proactively instead of reactively.
 */

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

interface DepCheck {
  key: string;
  label: string;
  status: "ok" | "missing" | "optional";
  detail: string;
  required: boolean;
}

interface DepsReport {
  ok: boolean;
  all_green: boolean;
  checks: DepCheck[];
}

async function fetchDeps(): Promise<DepsReport> {
  const r = await fetch("/api/v1/health/deps");
  if (!r.ok) throw new Error(`deps HTTP ${r.status}`);
  const body = await r.json();
  // Be defensive — tests or older servers may return a partial shape.
  return {
    ok: typeof body?.ok === "boolean" ? body.ok : true,
    all_green: typeof body?.all_green === "boolean" ? body.all_green : true,
    checks: Array.isArray(body?.checks) ? body.checks : [],
  };
}

export function DepsChip() {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const q = useQuery({
    queryKey: ["health", "deps"],
    queryFn: fetchDeps,
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (!wrapperRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Pre-resolve state for the trigger: red if a required dep is missing,
  // amber if only optional ones are, green when everything's there.
  const report = q.data;
  let dot = "bg-neutral-700";
  let title = "Loading deps health…";
  if (q.isError) {
    dot = "bg-rose-500";
    title = "Couldn't reach the backend deps endpoint";
  } else if (report) {
    if (!report.ok) {
      dot = "bg-rose-500";
      const missing = report.checks
        .filter((c) => c.required && c.status === "missing")
        .map((c) => c.label)
        .join(", ");
      title = `Missing required: ${missing}`;
    } else if (!report.all_green) {
      dot = "bg-amber-400";
      title = "Optional deps missing — click for details";
    } else {
      dot = "bg-emerald-400";
      title = "All deps ready";
    }
  }

  return (
    <div ref={wrapperRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="h-7 w-7 inline-flex items-center justify-center rounded hover:bg-neutral-800"
        title={title}
        aria-label="dependency health"
      >
        <span className={`block w-2 h-2 rounded-full ${dot}`} />
      </button>
      {open && report && (
        <div
          className="absolute right-0 top-full mt-1 z-50 w-80 rounded-md border border-neutral-700 bg-neutral-950 shadow-2xl"
          role="menu"
        >
          <div className="px-3 py-2 border-b border-neutral-900 text-[10px] uppercase tracking-wider text-neutral-500">
            Host dependencies
          </div>
          <ul className="py-1">
            {report.checks.map((c) => (
              <li key={c.key} className="px-3 py-2 text-xs">
                <div className="flex items-center gap-2">
                  <StatusGlyph status={c.status} />
                  <span className="text-neutral-100 font-medium">
                    {c.label}
                  </span>
                  {!c.required && (
                    <span className="text-[9px] uppercase tracking-wider text-neutral-600 ml-auto">
                      optional
                    </span>
                  )}
                </div>
                <p className="mt-0.5 pl-6 text-[11px] text-neutral-500 leading-snug">
                  {c.detail}
                </p>
              </li>
            ))}
          </ul>
          <div className="px-3 py-2 border-t border-neutral-900 text-[10px] text-neutral-600 flex items-center justify-between">
            <span>Auto-refreshes every 30s</span>
            <button
              type="button"
              onClick={() => q.refetch()}
              disabled={q.isFetching}
              className="text-neutral-400 hover:text-neutral-100 uppercase tracking-wider disabled:opacity-50"
            >
              ↻ refresh
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function StatusGlyph({ status }: { status: "ok" | "missing" | "optional" }) {
  if (status === "ok") {
    return <span className="text-emerald-400 w-4 inline-block">✓</span>;
  }
  if (status === "missing") {
    return <span className="text-rose-400 w-4 inline-block">✗</span>;
  }
  return <span className="text-neutral-600 w-4 inline-block">○</span>;
}
