/**
 * Per-slot stem-filter chip with an inline popover picker.
 *
 * Renders a compact pill:
 *   - `ALL` (gray)              when stem_kinds is null
 *   - e.g. `DR·VX` (violet)     when filtered
 *
 * Click to open a 4-toggle popover (DRUMS / BASS / VOCALS / OTHER) +
 * "load all" reset. Saves via PATCH on every toggle for instant feel.
 *
 * Used in the SetRail row + SetEditor row — same component, same data,
 * same PATCH path.
 */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateSetTrackStemKinds } from "../api";
import type { DanceSet } from "../types";

const ALL_KINDS = ["drums", "bass", "vocals", "other"] as const;
const SHORT: Record<string, string> = {
  drums: "DR",
  bass: "BA",
  vocals: "VX",
  other: "ME",
};

interface Props {
  setId: number;
  trackId: number;
  stemKinds: string[] | null;
  /** Compact (rail) shrinks padding + font. Full (editor) is larger. */
  variant?: "compact" | "full";
}

export function StemKindsChip({
  setId,
  trackId,
  stemKinds,
  variant = "compact",
}: Props) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const qc = useQueryClient();

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

  const save = useMutation({
    mutationFn: (next: string[] | null) =>
      updateSetTrackStemKinds(setId, trackId, next),
    onSuccess: (updated: DanceSet) => {
      qc.setQueryData(["sets", "active"], updated);
      qc.setQueryData(["sets", updated.id], updated);
    },
  });

  function toggle(kind: string) {
    const current = stemKinds ?? [...ALL_KINDS];
    const has = current.includes(kind);
    const next = has
      ? current.filter((k) => k !== kind)
      : [...current, kind];
    // Empty selection = no filter (back to ALL); the API would reject
    // an empty list explicitly, so collapse here.
    if (next.length === 0 || next.length === ALL_KINDS.length) {
      save.mutate(null);
    } else {
      save.mutate(next);
    }
  }

  const isAll = !stemKinds;
  const label = isAll
    ? "ALL"
    : stemKinds.map((k) => SHORT[k] ?? k.slice(0, 2).toUpperCase()).join("·");

  const triggerCls =
    variant === "compact"
      ? "text-[9px] font-mono uppercase tracking-wider px-1 py-0.5 rounded"
      : "text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded";
  const colorCls = isAll
    ? "text-neutral-500 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800"
    : "text-violet-200 bg-violet-700/30 hover:bg-violet-700/40 border border-violet-700/50";

  return (
    <div ref={wrapperRef} className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={
          isAll
            ? "Load all 4 stems when fired — click to filter"
            : `Only load these stems: ${stemKinds.join(", ")}`
        }
        className={`${triggerCls} ${colorCls}`}
      >
        {label}
      </button>
      {open && (
        <div
          className="absolute right-0 top-full mt-1 z-50 w-44 rounded-md border border-neutral-700 bg-neutral-950 shadow-2xl overflow-hidden"
          role="menu"
        >
          <div className="px-2 py-1.5 text-[9px] uppercase tracking-wider text-neutral-500 border-b border-neutral-900">
            Stems to load
          </div>
          <div className="py-1">
            {ALL_KINDS.map((k) => {
              const checked = isAll || stemKinds.includes(k);
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => toggle(k)}
                  disabled={save.isPending}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-neutral-100 hover:bg-neutral-900 disabled:opacity-50"
                >
                  <span
                    className={`w-3 ${
                      checked ? "text-emerald-400" : "text-neutral-700"
                    }`}
                  >
                    {checked ? "✓" : "·"}
                  </span>
                  <span className="flex-1 text-left uppercase">{k}</span>
                  <span className="text-[10px] text-neutral-500 font-mono">
                    {SHORT[k]}
                  </span>
                </button>
              );
            })}
          </div>
          {!isAll && (
            <button
              type="button"
              onClick={() => save.mutate(null)}
              disabled={save.isPending}
              className="w-full text-[10px] px-3 py-1.5 border-t border-neutral-900 text-neutral-500 hover:text-neutral-100 hover:bg-neutral-900 disabled:opacity-50 uppercase tracking-wider"
            >
              ↺ reset to ALL
            </button>
          )}
        </div>
      )}
    </div>
  );
}
