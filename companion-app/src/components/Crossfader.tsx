import { useMutation } from "@tanstack/react-query";
import { useRef } from "react";
import { abletonSetCrossfader } from "../api";
import { useAbletonState } from "../hooks/useAbletonState";

/**
 * Horizontal crossfader bar that lives between the Deck A and Deck B
 * panels. Mirrors Live's master crossfader (-1..+1 normalized to 0..1
 * for display). Driven primarily by APC40's hardware fader — when the
 * user moves the hardware, the on-screen handle follows. Also drag-
 * gable on screen as a fallback.
 *
 * Visual: gradient strip with a draggable handle. Click-anywhere snaps
 * the handle to that position; drag works too.
 */
export function Crossfader() {
  const state = useAbletonState();
  const setX = useMutation({ mutationFn: abletonSetCrossfader });
  const trackRef = useRef<HTMLDivElement>(null);

  // Live's crossfader: -1 (full A) to +1 (full B). Normalize to 0..1
  // for the on-screen handle position. ``null`` (Live not connected
  // yet) → render at center.
  const liveValue = state.crossfader ?? 0;
  const position = (liveValue + 1) / 2; // 0..1, 0 = full A, 1 = full B

  function ratioFromEvent(e: React.PointerEvent<HTMLDivElement>): number {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect) return 0.5;
    return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  }

  function commit(ratio: number) {
    // ratio 0..1 → crossfader -1..+1
    setX.mutate(ratio * 2 - 1);
  }

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    commit(ratioFromEvent(e));
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    commit(ratioFromEvent(e));
  }

  function handlePointerUp(e: React.PointerEvent<HTMLDivElement>) {
    e.currentTarget.releasePointerCapture(e.pointerId);
  }

  return (
    <div
      className="flex items-center gap-2"
      data-testid="crossfader"
      title={`Crossfader ${(liveValue * 100).toFixed(0)}% (${
        liveValue < -0.05 ? "→ A" : liveValue > 0.05 ? "→ B" : "centered"
      })`}
    >
      <span className="text-[10px] font-mono uppercase tracking-widest text-violet-300/80 shrink-0">
        A
      </span>
      <div
        ref={trackRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        className="relative flex-1 h-2 rounded-full bg-gradient-to-r from-violet-900/50 via-neutral-700/40 to-indigo-900/50 cursor-pointer"
        role="slider"
        aria-label="Master crossfader"
        aria-valuemin={-1}
        aria-valuemax={1}
        aria-valuenow={liveValue}
      >
        {/* Center tick */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-neutral-600/60 -translate-x-1/2 pointer-events-none" />
        {/* Handle */}
        <div
          className="absolute top-1/2 w-3 h-5 rounded-sm bg-neutral-100 shadow-md -translate-x-1/2 -translate-y-1/2 pointer-events-none transition-transform duration-75"
          style={{ left: `${position * 100}%` }}
        />
      </div>
      <span className="text-[10px] font-mono uppercase tracking-widest text-indigo-300/80 shrink-0">
        B
      </span>
    </div>
  );
}
