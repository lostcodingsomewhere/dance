/**
 * Set editor — expanded version of the Set Rail.
 *
 * Two-pane:
 *   left  — ordered track list with reorder, per-track note editing, remove.
 *           Header lets you rename the active set, switch to another, create
 *           a new one, or delete the current one.
 *   right — discovery pane: tabs between Library (fuzzy search via
 *           ``/tracks/search``) and Tail recs (the rail's arc-fit
 *           suggestions). Both have one-tap "Add to set".
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { searchTracks, updateSetTrackNote } from "../api";
import {
  useActiveSet,
  useActivateSet,
  useAddTrackToSet,
  useCreateSet,
  useMoveTrackInSet,
  useRemoveTrackFromSet,
  useTailRecs,
} from "../hooks/useSets";
import { store } from "../store";
import type { DanceSet, SetTrack, Track } from "../types";
import { EnergyBar } from "../components/EnergyBar";
import { KeyBadge } from "../components/KeyBadge";
import { SetMenu } from "../components/SetMenu";

export function SetEditor() {
  const active = useActiveSet();
  const set = active.data;

  if (active.isLoading) {
    return <CenteredMsg>Loading…</CenteredMsg>;
  }
  if (!set) {
    return <NoSetState />;
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <Header set={set} />
      <div className="flex-1 grid grid-cols-[1fr_1fr] gap-4 px-6 py-4 min-h-0">
        <LeftPane set={set} />
        <RightPane set={set} />
      </div>
    </div>
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
      <h1 className="text-xl font-medium text-neutral-100">
        No active set yet
      </h1>
      <p className="text-sm text-neutral-500 max-w-lg leading-relaxed">
        Sets persist across reloads and sessions. Name them, fill them via ⌘K,
        and reload any of them as the active rail.
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

function Header({ set }: { set: DanceSet }) {
  return (
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
      <span className="text-xs text-neutral-500 tabular-nums shrink-0">
        {set.tracks.length} tracks
      </span>
    </div>
  );
}

function LeftPane({ set }: { set: DanceSet }) {
  return (
    <section className="flex flex-col min-h-0 border border-neutral-800 rounded-md">
      <div className="px-3 py-2 border-b border-neutral-900 text-[10px] uppercase tracking-wider text-neutral-500">
        Track list
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {set.tracks.length === 0 ? (
          <div className="text-xs text-neutral-500 italic px-1 py-3">
            Empty set — pick from the right pane or open ⌘K to find tracks.
          </div>
        ) : (
          <ol className="flex flex-col gap-1">
            {set.tracks.map((t, i) => (
              <EditableTrackRow
                key={t.track_id}
                setId={set.id}
                track={t}
                index={i}
                isLast={i === set.tracks.length - 1}
              />
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function EditableTrackRow({
  setId,
  track,
  index,
  isLast,
}: {
  setId: number;
  track: SetTrack;
  index: number;
  isLast: boolean;
}) {
  const move = useMoveTrackInSet();
  const remove = useRemoveTrackFromSet();
  const [noteOpen, setNoteOpen] = useState(false);
  const [noteDraft, setNoteDraft] = useState(track.note ?? "");

  return (
    <li className="rounded-md border border-neutral-800 bg-neutral-900/40">
      <div className="flex items-center gap-2 px-2 py-1.5">
        <span className="font-mono text-[10px] text-neutral-500 tabular-nums w-6 text-right">
          {index + 1}
        </span>
        <KeyBadge keyCamelot={track.key_camelot ?? null} size="sm" />
        <div className="flex-1 min-w-0">
          <div className="text-sm text-neutral-100 truncate font-medium">
            {track.title ?? `Track #${track.track_id}`}
          </div>
          <div className="text-xs text-neutral-500 truncate">
            {track.artist ?? "—"}
            {track.bpm != null && (
              <span className="ml-2 font-mono">{track.bpm.toFixed(0)} BPM</span>
            )}
          </div>
        </div>
        <EnergyBar energy={track.floor_energy ?? null} size="sm" />
        <button
          type="button"
          disabled={index === 0 || move.isPending}
          onClick={() =>
            move.mutate({ setId, trackId: track.track_id, position: index - 1 })
          }
          className="w-6 h-6 text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800 rounded inline-flex items-center justify-center disabled:opacity-30"
          title="Move up"
        >
          ▲
        </button>
        <button
          type="button"
          disabled={isLast || move.isPending}
          onClick={() =>
            move.mutate({ setId, trackId: track.track_id, position: index + 1 })
          }
          className="w-6 h-6 text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800 rounded inline-flex items-center justify-center disabled:opacity-30"
          title="Move down"
        >
          ▼
        </button>
        <button
          type="button"
          onClick={() => setNoteOpen((v) => !v)}
          className={`w-6 h-6 rounded inline-flex items-center justify-center ${
            track.note
              ? "text-amber-300 hover:bg-amber-500/20"
              : "text-neutral-500 hover:text-neutral-100 hover:bg-neutral-800"
          }`}
          title={track.note ? `Note: ${track.note}` : "Add a note"}
        >
          📝
        </button>
        <button
          type="button"
          disabled={remove.isPending}
          onClick={() => remove.mutate({ setId, trackId: track.track_id })}
          className="w-6 h-6 text-neutral-500 hover:text-rose-200 hover:bg-rose-500/20 rounded inline-flex items-center justify-center disabled:opacity-30"
          title="Remove from set"
        >
          ×
        </button>
      </div>
      {noteOpen && (
        <NoteEditor
          setId={setId}
          trackId={track.track_id}
          initial={noteDraft}
          onSaved={(v) => {
            setNoteDraft(v);
            setNoteOpen(false);
          }}
          onCancel={() => setNoteOpen(false)}
        />
      )}
    </li>
  );
}

function NoteEditor({
  setId,
  trackId,
  initial,
  onSaved,
  onCancel,
}: {
  setId: number;
  trackId: number;
  initial: string;
  onSaved: (v: string) => void;
  onCancel: () => void;
}) {
  const [v, setV] = useState(initial);

  async function save() {
    await updateSetTrackNote(setId, trackId, v);
    onSaved(v);
  }

  return (
    <div className="px-2 pb-2 flex items-center gap-2">
      <input
        autoFocus
        value={v}
        onChange={(e) => setV(e.target.value)}
        placeholder='e.g. "cue at bar 33", "after the breakdown"'
        className="flex-1 bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-xs text-neutral-100 outline-none focus:border-neutral-700"
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
          if (e.key === "Escape") onCancel();
        }}
      />
      <button
        type="button"
        onClick={() => void save()}
        className="text-xs px-2 py-1 rounded bg-violet-700 hover:bg-violet-600 text-white"
      >
        Save
      </button>
    </div>
  );
}

function RightPane({ set }: { set: DanceSet }) {
  const [tab, setTab] = useState<"library" | "tail">("library");
  return (
    <section className="flex flex-col min-h-0 border border-neutral-800 rounded-md">
      <div className="px-3 py-2 border-b border-neutral-900 flex items-center gap-2">
        <PaneTab active={tab === "library"} onClick={() => setTab("library")}>
          Library
        </PaneTab>
        <PaneTab active={tab === "tail"} onClick={() => setTab("tail")}>
          Tail recs
        </PaneTab>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {tab === "library" ? (
          <LibraryBrowse set={set} />
        ) : (
          <TailRecsPane set={set} />
        )}
      </div>
    </section>
  );
}

function PaneTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`text-[11px] uppercase tracking-wider px-2 py-1 rounded ${
        active
          ? "bg-neutral-800 text-neutral-100"
          : "text-neutral-500 hover:text-neutral-200"
      }`}
    >
      {children}
    </button>
  );
}

function LibraryBrowse({ set }: { set: DanceSet }) {
  const [q, setQ] = useState("");
  const fuzzy = useQuery({
    queryKey: ["tracks", "search", q.toLowerCase()],
    queryFn: () => searchTracks({ q, limit: 30 }),
    staleTime: 5_000,
  });
  const inSet = new Set(set.tracks.map((t) => t.track_id));
  return (
    <div className="flex flex-col gap-2">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search title or artist…"
        className="bg-neutral-900 border border-neutral-800 rounded px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-neutral-700"
      />
      {fuzzy.isLoading && (
        <div className="text-xs text-neutral-600">…</div>
      )}
      {fuzzy.data?.length === 0 && (
        <div className="text-xs text-neutral-600 italic">No matches.</div>
      )}
      <ol className="flex flex-col gap-1">
        {fuzzy.data?.map((t) => (
          <LibraryRow
            key={t.id}
            track={t}
            setId={set.id}
            alreadyInSet={inSet.has(t.id)}
          />
        ))}
      </ol>
    </div>
  );
}

function LibraryRow({
  track,
  setId,
  alreadyInSet,
}: {
  track: Track;
  setId: number;
  alreadyInSet: boolean;
}) {
  const add = useAddTrackToSet();
  return (
    <li className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-neutral-900/60 border border-transparent hover:border-neutral-800">
      <KeyBadge keyCamelot={track.analysis?.key_camelot ?? null} size="sm" />
      <div className="flex-1 min-w-0">
        <div className="text-sm text-neutral-100 truncate font-medium">
          {track.title ?? `Track #${track.id}`}
        </div>
        <div className="text-xs text-neutral-500 truncate">
          {track.artist ?? "—"}
          {track.analysis?.bpm != null && (
            <span className="ml-2 font-mono">
              {track.analysis.bpm.toFixed(0)} BPM
            </span>
          )}
        </div>
      </div>
      <EnergyBar energy={track.analysis?.floor_energy ?? null} size="sm" />
      <button
        type="button"
        disabled={alreadyInSet || add.isPending}
        onClick={() => add.mutate({ setId, trackId: track.id })}
        className="text-xs px-2 py-1 rounded bg-violet-700 hover:bg-violet-600 text-white disabled:opacity-30 disabled:bg-neutral-800"
        title={alreadyInSet ? "Already in set" : "Add to set"}
      >
        {alreadyInSet ? "✓ in set" : "+ Set"}
      </button>
    </li>
  );
}

function TailRecsPane({ set }: { set: DanceSet }) {
  const recs = useTailRecs(set.id, { k: 20 });
  const add = useAddTrackToSet();
  if (set.tracks.length === 0) {
    return (
      <div className="text-xs text-neutral-500 italic px-1 py-3">
        Add a few tracks first — tail recs score against the arc the set is
        building.
      </div>
    );
  }
  return (
    <div>
      {recs.isLoading && <div className="text-xs text-neutral-600">…</div>}
      {recs.data?.recs.length === 0 && (
        <div className="text-xs text-neutral-600 italic">
          No candidates — try widening the library first.
        </div>
      )}
      <ol className="flex flex-col gap-1">
        {recs.data?.recs.map((r) => (
          <li
            key={r.track_id}
            className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-neutral-900/60 border border-transparent hover:border-neutral-800"
          >
            <span className="font-mono text-[10px] text-neutral-500 tabular-nums w-8 text-right">
              {Math.round(r.score * 100)}
            </span>
            <KeyBadge keyCamelot={r.key_camelot ?? null} size="sm" />
            <div className="flex-1 min-w-0">
              <div className="text-sm text-neutral-100 truncate font-medium">
                {r.track_title ?? `Track #${r.track_id}`}
              </div>
              <div className="text-xs text-neutral-500 truncate">
                {r.track_artist ?? "—"}
                {r.bpm != null && (
                  <span className="ml-2 font-mono">{r.bpm.toFixed(0)}</span>
                )}
                {r.reasons.length > 0 && (
                  <span className="ml-2 text-neutral-600">
                    · {r.reasons.join(" · ")}
                  </span>
                )}
              </div>
            </div>
            <EnergyBar energy={r.floor_energy ?? null} size="sm" />
            <button
              type="button"
              onClick={() => add.mutate({ setId: set.id, trackId: r.track_id })}
              className="text-xs px-2 py-1 rounded bg-violet-700 hover:bg-violet-600 text-white"
            >
              + Set
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
