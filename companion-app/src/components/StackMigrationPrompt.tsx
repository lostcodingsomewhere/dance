/**
 * One-shot Stack → Set migration prompt.
 *
 * The pre-Set world stored "tracks I'm planning to use today" in
 * ``localStorage.dance.companion.state.v2.stack`` (an array of track IDs).
 * When the user upgrades to the Set Rail, we import that list as a named
 * Set, activate it, and clear the old localStorage value so the prompt
 * never re-fires.
 *
 * If the user has no stack, or already has an active set, the prompt
 * doesn't render. If they dismiss without importing, we drop a sentinel
 * so we don't pester them on every reload.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { activateSet, addTrackToSet, createSet } from "../api";
import { useActiveSet } from "../hooks/useSets";

const STORAGE_KEY = "dance.companion.state.v2";
const DISMISS_KEY = "dance.companion.stack_migration_dismissed";

function readLegacyStack(): number[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { stack?: number[] };
    return Array.isArray(parsed.stack) ? parsed.stack.filter((n) => typeof n === "number") : [];
  } catch {
    return [];
  }
}

function clearLegacyStack(): void {
  if (typeof window === "undefined") return;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    delete parsed.stack;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
  } catch {
    // ignore — if the JSON is broken there's nothing to migrate anyway
  }
}

export function StackMigrationPrompt() {
  const qc = useQueryClient();
  const active = useActiveSet();
  const [legacy, setLegacy] = useState<number[]>([]);
  const [name, setName] = useState("Imported Stack");
  const dismissed =
    typeof window !== "undefined" &&
    !!window.localStorage.getItem(DISMISS_KEY);

  useEffect(() => {
    setLegacy(readLegacyStack());
  }, []);

  const importMutation = useMutation({
    mutationFn: async () => {
      const created = await createSet(name.trim() || "Imported Stack");
      // POST each track in order. Sequential keeps positions stable.
      for (const trackId of legacy) {
        try {
          await addTrackToSet(created.id, trackId);
        } catch {
          // best-effort: a since-deleted track shouldn't block import
        }
      }
      // Only activate if the user has no active set already.
      if (!active.data) await activateSet(created.id);
      clearLegacyStack();
      return created;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sets"] });
      setLegacy([]);
    },
  });

  function dismiss() {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(DISMISS_KEY, "1");
    }
    setLegacy([]);
  }

  if (active.isLoading) return null;
  if (dismissed) return null;
  if (legacy.length === 0) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm px-4"
      role="dialog"
      aria-modal="true"
      aria-label="Stack migration"
    >
      <div className="w-full max-w-md rounded-xl border border-violet-700/50 bg-neutral-950 shadow-2xl p-5 space-y-3">
        <h2 className="text-lg font-semibold text-neutral-100">
          Import your old Stack as a Set?
        </h2>
        <p className="text-sm text-neutral-400 leading-relaxed">
          You have <strong>{legacy.length}</strong> track
          {legacy.length === 1 ? "" : "s"} from the old Stack. Sets persist
          across reloads, surface tail recs, and can be reloaded between
          gigs. Name your imported set:
        </p>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-neutral-900 border border-neutral-800 rounded px-3 py-2 text-sm text-neutral-100 outline-none focus:border-violet-700"
          autoFocus
        />
        {importMutation.isError && (
          <div className="text-xs text-rose-300">
            Import failed: {(importMutation.error as Error).message}
          </div>
        )}
        <div className="flex items-center gap-2 justify-end pt-2">
          <button
            type="button"
            onClick={dismiss}
            disabled={importMutation.isPending}
            className="text-xs text-neutral-500 hover:text-neutral-200 px-3 py-2"
          >
            Skip
          </button>
          <button
            type="button"
            onClick={() => importMutation.mutate()}
            disabled={importMutation.isPending || !name.trim()}
            className="h-9 px-4 rounded bg-violet-700 hover:bg-violet-600 text-white text-sm font-medium disabled:opacity-50"
          >
            {importMutation.isPending ? "Importing…" : "Import as Set"}
          </button>
        </div>
      </div>
    </div>
  );
}
