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
import { addTrackToSet, createSet } from "../api";
import {
  useActivateSet,
  useCreateSet,
  useDeleteSet,
  useSets,
  useUpdateSet,
} from "../hooks/useSets";
import { useQueryClient } from "@tanstack/react-query";
import type { DanceSet } from "../types";

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

  async function duplicateCurrent() {
    const name = newName.trim() || `${set.name} (copy)`;
    try {
      const copy = await createSet(name);
      for (const t of set.tracks) {
        try {
          await addTrackToSet(copy.id, t.track_id, { note: t.note ?? undefined });
        } catch {
          // skip tracks that have since been deleted
        }
      }
      await activate.mutateAsync(copy.id);
      qc.invalidateQueries({ queryKey: ["sets"] });
      setNewName("");
      setCreating(null);
      onClose();
    } catch (e) {
      console.error("duplicate failed", e);
    }
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
              if (e.key === "Escape") {
                setNewName("");
                setCreating(null);
              }
            }}
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setNewName("");
                setCreating(null);
              }}
              className="text-xs text-neutral-500 hover:text-neutral-200 px-2 py-1"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() =>
                creating === "duplicate" ? duplicateCurrent() : createBlank()
              }
              disabled={createMut.isPending || activate.isPending}
              className="text-xs px-3 py-1 rounded bg-violet-700 hover:bg-violet-600 text-white disabled:opacity-50"
            >
              {createMut.isPending || activate.isPending ? "…" : "Create"}
            </button>
          </div>
        </div>
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
            onClick={() => setCreating("duplicate")}
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
