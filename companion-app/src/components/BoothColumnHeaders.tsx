import { useState } from "react";
import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  useSortable,
  arrayMove,
  horizontalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { useDeckMap } from "../hooks/useDeckMap";
import { useSoloTrack } from "../hooks/useTransport";
import { ROLE_STYLES, roleLabel, type StemRole } from "../lib/roles";
import { store, useAppStore } from "../store";
import { RoleIcon } from "./RoleIcon";

/**
 * The single column header strip at the top of the Booth — labels each
 * of the 5 stem columns once, with a distinct background tint so the
 * column identity carries down through ComboStrip → SceneGrid → recs
 * below.
 *
 * Hosts the per-column Solo (S) button which routes that one stem
 * through Live's Cue / PFL bus → Scarlett 4i4 outs 3/4 → headphones,
 * leaving master untouched. Solo state lives here (was previously
 * inside SceneGrid).
 *
 * Reorderable via @dnd-kit/sortable: grab any header chip and drag it
 * onto another to reorder visual column order. Persisted via
 * ``store.stemColumnOrder``. The underlying Ableton track indices don't
 * move — we just iterate the column list in the user's chosen order
 * everywhere downstream. Double-click any header to reset to canonical.
 *
 * Layout matches the ``[2.5rem leading + 5 stem cols] gap-1`` template
 * used by SceneGrid / ComboStrip / rec-banner so headers line up
 * vertically with the strips below.
 */
export function BoothColumnHeaders() {
  const deckMap = useDeckMap();
  const soloTrack = useSoloTrack();
  const columns = deckMap.data?.columns ?? null;
  const stemColumns = useAppStore((s) => s.stemColumnOrder);
  const [soloedRoles, setSoloedRoles] = useState<Set<string>>(new Set());

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  );

  function toggleSolo(role: StemRole, trackIdx: number) {
    const wasSoloed = soloedRoles.has(role);
    const next = new Set(soloedRoles);
    if (wasSoloed) next.delete(role);
    else next.add(role);
    setSoloedRoles(next);
    soloTrack.mutate({ track: trackIdx, soloed: !wasSoloed });
  }

  function onDragEnd(e: DragEndEvent) {
    const { active, over } = e;
    if (!over || active.id === over.id) return;
    const oldIdx = stemColumns.indexOf(String(active.id));
    const newIdx = stemColumns.indexOf(String(over.id));
    if (oldIdx < 0 || newIdx < 0) return;
    store.setStemColumnOrder(arrayMove(stemColumns, oldIdx, newIdx));
  }

  if (!columns) return null;

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={onDragEnd}
    >
      <SortableContext items={stemColumns} strategy={horizontalListSortingStrategy}>
        <div
          className="grid grid-cols-[2.5rem_repeat(5,minmax(0,1fr))] gap-1"
          data-testid="booth-column-headers"
        >
          <div aria-hidden="true" />
          {stemColumns.map((role) => (
            <ColumnHeaderChip
              key={role}
              role={role}
              trackIdx={columns[role]}
              isSoloed={soloedRoles.has(role)}
              onSoloToggle={() => {
                const idx = columns[role];
                if (idx != null) toggleSolo(role as StemRole, idx);
              }}
            />
          ))}
        </div>
      </SortableContext>
    </DndContext>
  );
}

function ColumnHeaderChip({
  role,
  trackIdx,
  isSoloed,
  onSoloToggle,
}: {
  role: string;
  trackIdx: number | undefined;
  isSoloed: boolean;
  onSoloToggle: () => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: role });
  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
  };
  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      // Double-click any header to reset to canonical order — quick
      // escape hatch if the user reorders by accident.
      onDoubleClick={() => store.resetStemColumnOrder()}
      title="Drag to reorder columns · double-click to reset"
      className={`flex items-center gap-1.5 px-2 py-1.5 rounded-md border text-[11px] uppercase tracking-widest font-semibold cursor-grab active:cursor-grabbing touch-none transition-all ${
        ROLE_STYLES[role as StemRole].header
      } ${isDragging ? "opacity-40 z-30" : ""}`}
    >
      <RoleIcon role={role} size={14} />
      <span className="flex-1">{roleLabel(role)}</span>
      {trackIdx != null && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onSoloToggle();
          }}
          // dnd-kit listens via pointer events, so we need to stop them
          // bubbling from the Solo button — otherwise a click here
          // briefly initiates a drag.
          onPointerDown={(e) => e.stopPropagation()}
          title={
            isSoloed
              ? `Unsolo ${roleLabel(role).toLowerCase()} — return to normal routing`
              : `Solo ${roleLabel(role).toLowerCase()} → cue out (Live's solo button must be in Cue/PFL mode)`
          }
          className={`text-[10px] leading-none rounded px-1.5 py-0.5 transition-colors ${
            isSoloed
              ? "bg-amber-400/30 text-amber-100 border border-amber-300/60"
              : "text-neutral-500 hover:text-amber-200 border border-transparent"
          }`}
          aria-label={isSoloed ? `unsolo ${role}` : `solo ${role}`}
          aria-pressed={isSoloed}
        >
          S
        </button>
      )}
    </div>
  );
}
