/**
 * SetMenu — the active-set chip + dropdown of all set actions.
 *
 * Click the name → popover with:
 *   • All sets (current row shows ✓, click to activate)
 *   • Rename current (inline edit in the popover)
 *   • Duplicate as new (creates a copy, activates it)
 *   • New empty set (opens a name input)
 *   • Delete current (confirm dialog)
 *
 * Used in the SetRail drawer header (compact mode) and the SetEditor view
 * header (full mode). Same affordances either way.
 */

import { useEffect, useRef, useState } from "react";
import { createSet, deleteSet, getPlan, serverMessage } from "../api";
import { putPlanWithoutDeletedTracks } from "../hooks/useSetPlan";
import {
  useActivateSet,
  useCreateSet,
  useDeleteSet,
  useSets,
  useUpdateSet,
} from "../hooks/useSets";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { DanceSet, SetPlan } from "../types";

function trackIdsByRole(plan: SetPlan): Record<string, number[]> {
  return Object.fromEntries(
    Object.entries(plan.queues).map(([role, picks]) => [role, picks.map((pick) => pick.track_id)]),
  );
}

interface CopyMade {
  copy: DanceSet;
  picksLeftOut: number;
}

async function copyWithItsPlan(source: DanceSet, name: string): Promise<CopyMade> {
  const plan = await getPlan(source.id);
  const copy = await createSet(name);
  try {
    const { picksLeftOut } = await putPlanWithoutDeletedTracks(copy.id, trackIdsByRole(plan));
    return { copy, picksLeftOut };
  } catch (error) {
    await deleteSet(copy.id).catch(() => undefined);
    throw error;
  }
}

function leftOutWords(picksLeftOut: number): string {
  const picks = picksLeftOut === 1 ? "1 pick" : `${picksLeftOut} picks`;
  return `This copy leaves out ${picks} whose track was deleted.`;
}

class CopyNotOpened extends Error {}

function duplicateFailureWords(error: Error): string {
  if (error instanceof CopyNotOpened) {
    return "The copy was made but could not be opened. Pick it under Switch to.";
  }
  const reason = serverMessage(error);
  const serverGaveAReason = reason !== error.message;
  return serverGaveAReason
    ? `Could not duplicate this set. The server said: ${reason}`
    : "Could not duplicate this set. Try again.";
}

interface Props {
  set: DanceSet;
  /** Compact = rail header (tight, white-text trigger). Full = SetEditor
   *  view header (larger trigger). Both render the same popover. */
  variant?: "compact" | "full";
}

export function SetMenu({ set, variant = "compact" }: Props) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // Close popover on outside click or Esc.
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

  const triggerCls =
    variant === "compact"
      ? "flex items-center gap-1 text-sm font-medium text-neutral-100 hover:text-violet-300 truncate"
      : "flex items-center gap-1 text-lg font-medium text-neutral-100 hover:text-violet-300";

  return (
    <div ref={wrapperRef} className="relative flex-1 min-w-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title="Switch / rename / new / delete"
        className={triggerCls}
      >
        <span className="truncate">{set.name}</span>
        <span aria-hidden className="text-neutral-500 text-xs shrink-0">▾</span>
      </button>
      {open && <Popover set={set} onClose={() => setOpen(false)} />}
    </div>
  );
}

function Popover({ set, onClose }: { set: DanceSet; onClose: () => void }) {
  const sets = useSets();
  const activate = useActivateSet();
  const update = useUpdateSet();
  const del = useDeleteSet();
  const createMut = useCreateSet();
  const qc = useQueryClient();

  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(set.name);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState<"new" | "duplicate" | null>(null);
  const [picksLeftOut, setPicksLeftOut] = useState(0);

  function saveRename() {
    const trimmed = nameDraft.trim();
    if (trimmed && trimmed !== set.name) {
      update.mutate(
        { id: set.id, patch: { name: trimmed } },
        { onSuccess: () => setEditingName(false) },
      );
    } else {
      setEditingName(false);
      setNameDraft(set.name);
    }
  }

  function createBlank() {
    const name = newName.trim();
    if (!name) return;
    createMut.mutate(
      { name },
      {
        onSuccess: (created) =>
          activate.mutate(created.id, {
            onSuccess: () => {
              setNewName("");
              setCreating(null);
              onClose();
            },
          }),
      },
    );
  }

  const duplicate = useMutation({
    meta: { showsOwnFailure: true },
    mutationFn: async (name: string) => {
      const { copy, picksLeftOut } = await copyWithItsPlan(set, name);
      try {
        await activate.mutateAsync(copy.id);
      } catch {
        qc.invalidateQueries({ queryKey: ["sets"] });
        throw new CopyNotOpened();
      }
      return picksLeftOut;
    },
    onSuccess: (leftOut) => {
      setNewName("");
      setCreating(null);
      if (leftOut === 0) onClose();
      else setPicksLeftOut(leftOut);
    },
  });
  const busy = createMut.isPending || activate.isPending || duplicate.isPending;

  function duplicateCurrent() {
    if (busy) return;
    duplicate.mutate(newName.trim() || `${set.name} (copy)`);
  }

  function confirmDelete() {
    if (
      window.confirm(
        `Delete set "${set.name}"? Track plans cannot be recovered.`,
      )
    ) {
      del.mutate(set.id, { onSuccess: () => onClose() });
    }
  }

  const allSets = sets.data ?? [];

  return (
    <div
      className="absolute left-0 top-full mt-1 z-50 w-72 rounded-md border border-neutral-700 bg-neutral-950 shadow-2xl overflow-hidden"
      role="menu"
    >
      {/* Header — inline-rename when active. */}
      <div className="px-3 py-2 border-b border-neutral-900">
        <div className="text-[9px] uppercase tracking-wider text-neutral-500">
          Current set
        </div>
        {editingName ? (
          <input
            autoFocus
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            onBlur={saveRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") saveRename();
              if (e.key === "Escape") {
                setNameDraft(set.name);
                setEditingName(false);
              }
            }}
            className="mt-0.5 w-full text-sm font-medium text-neutral-100 bg-neutral-900 border border-violet-700 rounded px-2 py-1 outline-none"
          />
        ) : (
          <button
            type="button"
            onClick={() => setEditingName(true)}
            className="mt-0.5 w-full text-left text-sm font-medium text-neutral-100 hover:text-violet-300"
            title="Click to rename"
          >
            {set.name}
          </button>
        )}
      </div>

      {/* Switcher list. */}
      {allSets.length > 1 && (
        <div className="max-h-64 overflow-y-auto py-1">
          <div className="px-3 py-1 text-[9px] uppercase tracking-wider text-neutral-500">
            Switch to
          </div>
          {allSets.map((s) => {
            const isCurrent = s.id === set.id;
            return (
              <button
                key={s.id}
                type="button"
                disabled={isCurrent || activate.isPending}
                onClick={() =>
                  activate.mutate(s.id, { onSuccess: () => onClose() })
                }
                className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs ${
                  isCurrent
                    ? "text-neutral-300 cursor-default"
                    : "text-neutral-100 hover:bg-neutral-900"
                }`}
              >
                <span className="w-3 text-emerald-400">
                  {isCurrent ? "✓" : ""}
                </span>
                <span className="flex-1 truncate text-left">{s.name}</span>
                <span className="text-neutral-500 tabular-nums">
                  {s.track_count}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Create / duplicate inline forms. */}
      {creating !== null && (
        <div className="px-3 py-2 border-t border-neutral-900 bg-neutral-900/40 space-y-2">
          <div className="text-[9px] uppercase tracking-wider text-neutral-500">
            {creating === "duplicate"
              ? `Duplicate "${set.name}" as`
              : "New set name"}
          </div>
          <input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder={
              creating === "duplicate"
                ? `${set.name} (copy)`
                : "e.g. Warehouse Sat"
            }
            className="w-full text-sm bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-neutral-100 outline-none focus:border-violet-700"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                creating === "duplicate" ? duplicateCurrent() : createBlank();
              }
              if (e.key === "Escape" && !busy) {
                setNewName("");
                setCreating(null);
              }
            }}
          />
          {creating === "duplicate" && duplicate.isError && (
            <p role="alert" className="text-xs leading-snug text-rose-300">
              {duplicateFailureWords(duplicate.error)}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setNewName("");
                setCreating(null);
              }}
              disabled={busy}
              className="text-xs text-neutral-500 hover:text-neutral-200 px-2 py-1 disabled:opacity-50 disabled:hover:text-neutral-500"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() =>
                creating === "duplicate" ? duplicateCurrent() : createBlank()
              }
              disabled={busy}
              className="text-xs px-3 py-1 rounded bg-violet-700 hover:bg-violet-600 text-white disabled:opacity-50"
            >
              {busy ? "…" : "Create"}
            </button>
          </div>
        </div>
      )}

      {creating === null && picksLeftOut > 0 && (
        <p
          role="status"
          className="px-3 py-2 border-t border-neutral-900 text-xs leading-snug text-neutral-300"
        >
          {leftOutWords(picksLeftOut)}
        </p>
      )}

      {/* Action list. */}
      {creating === null && (
        <div className="border-t border-neutral-900 py-1">
          <ActionRow
            icon="✏"
            label="Rename"
            onClick={() => {
              setNameDraft(set.name);
              setEditingName(true);
            }}
          />
          <ActionRow
            icon="⎘"
            label="Duplicate as new"
            onClick={() => {
              duplicate.reset();
              setPicksLeftOut(0);
              setCreating("duplicate");
            }}
          />
          <ActionRow
            icon="+"
            label="New empty set…"
            onClick={() => setCreating("new")}
          />
          <ActionRow
            icon="⌫"
            label="Delete current"
            danger
            onClick={confirmDelete}
          />
        </div>
      )}
    </div>
  );
}

function ActionRow({
  icon,
  label,
  onClick,
  danger = false,
}: {
  icon: string;
  label: string;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs ${
        danger
          ? "text-rose-300 hover:bg-rose-500/20"
          : "text-neutral-200 hover:bg-neutral-900"
      }`}
    >
      <span className="w-4 text-center text-neutral-500" aria-hidden>
        {icon}
      </span>
      <span>{label}</span>
    </button>
  );
}
