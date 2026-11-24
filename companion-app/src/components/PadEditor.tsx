import { useState } from "react";

import { serverMessage } from "../api";
import { useMovePad, usePads, useResetPad } from "../hooks/usePads";
import { useTrackWaveform } from "../hooks/useWaveform";
import { formatDurationMs } from "../lib/format";
import { roleLabel, STEM_COLUMNS } from "../lib/roles";
import type { Pad, PadRole, PadTarget } from "../types";
import { Waveform } from "./Waveform";

const TARGETS: readonly PadTarget[] = ["all", ...STEM_COLUMNS];

interface Tone {
  head: string;
  pole: string;
}

const INTRO_OUTRO_TONE: Tone = {
  head: "bg-violet-400 text-neutral-950",
  pole: "border-violet-400",
};
const NEUTRAL_TONE: Tone = {
  head: "bg-neutral-400 text-neutral-950",
  pole: "border-neutral-400",
};
const SECTION_TONES: Record<string, Tone> = {
  intro: INTRO_OUTRO_TONE,
  outro: INTRO_OUTRO_TONE,
  buildup: { head: "bg-yellow-400 text-neutral-950", pole: "border-yellow-400" },
  drop: { head: "bg-red-500 text-white", pole: "border-red-500" },
  breakdown: { head: "bg-cyan-400 text-neutral-950", pole: "border-cyan-400" },
};

function toneOf(pad: Pad): Tone {
  return SECTION_TONES[pad.section_label ?? ""] ?? NEUTRAL_TONE;
}

function targetLabel(target: PadTarget): string {
  return target === "all" ? "All" : roleLabel(target);
}

function shownRole(target: PadTarget): PadRole {
  return target === "all" ? "mix" : target;
}

function sectionName(pad: Pad): string {
  return pad.section_label ?? "other";
}

function barsLabel(pad: Pad): string | null {
  if (pad.length_bars == null) return null;
  return `${pad.length_bars} ${pad.length_bars === 1 ? "bar" : "bars"}`;
}

function isMovedByHand(pad: Pad): boolean {
  return pad.source === "manual";
}

function padSummary(pad: Pad): string {
  return [
    `Pad ${pad.slot}`,
    sectionName(pad),
    formatDurationMs(pad.position_ms),
    barsLabel(pad),
    pad.silent ? "silent" : null,
    isMovedByHand(pad) ? "moved by hand" : null,
  ]
    .filter(Boolean)
    .join(", ");
}

function moveHint(selectedPad: Pad | null, target: PadTarget): string {
  if (selectedPad == null) return "Pick a pad, then click the waveform to move it.";
  const scope = target === "all" ? " for every stem and the song" : "";
  return `Click the waveform to move pad ${selectedPad.slot}${scope}.`;
}

export function PadEditor({ trackId }: { trackId: number }) {
  const pads = usePads(trackId);
  const waveform = useTrackWaveform(trackId);
  const movePad = useMovePad(trackId);
  const resetPad = useResetPad(trackId);
  const [target, setTarget] = useState<PadTarget>("all");
  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
  const [rejection, setRejection] = useState<string | null>(null);

  const durationMs = (waveform.data?.duration_seconds ?? 0) * 1000;
  const shownPads = (pads.data ?? []).filter((pad) => pad.role === shownRole(target));
  const selectedPad = shownPads.find((pad) => pad.slot === selectedSlot) ?? null;
  const saving = movePad.isPending || resetPad.isPending;
  const showRejection = (error: Error) => setRejection(serverMessage(error));

  function chooseTarget(next: PadTarget) {
    setTarget(next);
    setRejection(null);
  }

  function moveSelectedPadTo(ratio: number) {
    if (selectedPad == null) return;
    setRejection(null);
    movePad.mutate(
      { target, slot: selectedPad.slot, positionMs: Math.round(ratio * durationMs) },
      { onError: showRejection },
    );
  }

  function resetSelectedPad() {
    if (selectedPad == null) return;
    setRejection(null);
    resetPad.mutate({ target, slot: selectedPad.slot }, { onError: showRejection });
  }

  if (pads.isLoading) return <p className="text-xs text-neutral-500">Loading pads…</p>;
  if (pads.isError) {
    return (
      <p role="alert" className="text-xs text-rose-400">
        {serverMessage(pads.error)}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <TargetPicker target={target} onChoose={chooseTarget} />
        <span className="flex-1 text-[11px] text-neutral-500">
          {moveHint(selectedPad, target)}
        </span>
        <button
          type="button"
          onClick={resetSelectedPad}
          disabled={saving || selectedPad == null || !isMovedByHand(selectedPad)}
          title="Put the selected pad back on the app's pick"
          className="h-7 rounded border border-neutral-800 bg-neutral-900 px-2.5 text-[11px] text-neutral-300 transition-colors hover:bg-neutral-800 disabled:opacity-50"
        >
          Reset{selectedPad ? ` pad ${selectedPad.slot}` : ""}
        </button>
      </div>

      {shownPads.length === 0 ? (
        <p className="text-xs text-neutral-500">
          This track has no pads{target === "all" ? "" : ` for ${targetLabel(target)}`} yet.
        </p>
      ) : (
        <>
          <div className="relative isolate pt-5">
            <div className="rounded bg-neutral-900">
              <Waveform
                peaks={waveform.data?.peaks ?? []}
                height={72}
                className="text-neutral-500"
                onSeek={selectedPad && !saving ? moveSelectedPadTo : undefined}
              />
            </div>
            {durationMs > 0 &&
              shownPads.map((pad) => (
                <PadFlag
                  key={pad.slot}
                  pad={pad}
                  durationMs={durationMs}
                  selected={pad.slot === selectedSlot}
                  onSelect={() => setSelectedSlot(pad.slot)}
                />
              ))}
          </div>

          {waveform.isError && (
            <p className="text-xs text-neutral-500">
              No waveform for this track, so its pads cannot be moved here.
            </p>
          )}
          {rejection && (
            <p role="alert" className="text-xs text-rose-400">
              {rejection}
            </p>
          )}

          <div className="grid grid-cols-5 gap-1.5">
            {shownPads.map((pad) => (
              <PadChip
                key={pad.slot}
                pad={pad}
                selected={pad.slot === selectedSlot}
                onSelect={() => setSelectedSlot(pad.slot)}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function TargetPicker({
  target,
  onChoose,
}: {
  target: PadTarget;
  onChoose: (target: PadTarget) => void;
}) {
  return (
    <div
      role="group"
      aria-label="Which stem's pads"
      className="inline-flex overflow-hidden rounded border border-neutral-800"
    >
      {TARGETS.map((choice) => (
        <button
          key={choice}
          type="button"
          aria-pressed={choice === target}
          autoFocus={choice === target}
          onClick={() => onChoose(choice)}
          className={`h-7 px-2.5 text-[11px] font-medium transition-colors ${
            choice === target
              ? "bg-neutral-100 text-neutral-950"
              : "bg-neutral-900 text-neutral-300 hover:bg-neutral-800"
          }`}
        >
          {targetLabel(choice)}
        </button>
      ))}
    </div>
  );
}

function PadFlag({
  pad,
  durationMs,
  selected,
  onSelect,
}: {
  pad: Pad;
  durationMs: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const tone = toneOf(pad);
  return (
    <div
      className={`pointer-events-none absolute inset-y-0 border-l ${tone.pole} ${
        pad.silent ? "opacity-40" : ""
      } ${selected ? "z-10" : ""}`}
      style={{ left: `${Math.min(100, (pad.position_ms / durationMs) * 100)}%` }}
    >
      <button
        type="button"
        tabIndex={-1}
        data-testid="pad-flag"
        onClick={onSelect}
        aria-label={`Select pad ${pad.slot}`}
        title={padSummary(pad)}
        className={`pointer-events-auto flex h-4 items-center gap-0.5 rounded-r-sm px-1 font-mono text-[10px] font-bold leading-none ${
          tone.head
        } ${selected ? "ring-2 ring-amber-300" : ""}`}
      >
        <PadNumber pad={pad} />
      </button>
    </div>
  );
}

function PadNumber({ pad }: { pad: Pad }) {
  return (
    <>
      {pad.slot}
      {isMovedByHand(pad) && <span aria-hidden className="h-1 w-1 rounded-full bg-current" />}
    </>
  );
}

function PadChip({
  pad,
  selected,
  onSelect,
}: {
  pad: Pad;
  selected: boolean;
  onSelect: () => void;
}) {
  const bars = barsLabel(pad);
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      aria-label={padSummary(pad)}
      title={padSummary(pad)}
      className={`flex min-w-0 items-start gap-1.5 rounded border px-1.5 py-1 text-left transition-colors ${
        selected
          ? "border-amber-300/80 bg-neutral-800"
          : "border-neutral-800 bg-neutral-900 hover:bg-neutral-800"
      }`}
    >
      <span
        className={`flex h-4 min-w-4 shrink-0 items-center justify-center gap-0.5 rounded-sm px-1 font-mono text-[10px] font-bold leading-none ${
          toneOf(pad).head
        }`}
      >
        <PadNumber pad={pad} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[10px] font-semibold uppercase tracking-wide text-neutral-200">
          {sectionName(pad)}
        </span>
        <span className="block truncate text-[10px] text-neutral-500">
          <span className="font-mono tabular-nums text-neutral-400">
            {formatDurationMs(pad.position_ms)}
          </span>
          {bars && <span> · {bars}</span>}
          {pad.silent && <span className="italic"> · silent</span>}
        </span>
      </span>
    </button>
  );
}
