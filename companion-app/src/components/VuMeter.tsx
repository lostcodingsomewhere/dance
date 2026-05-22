import { useMemo } from "react";
import { useAbletonState } from "../hooks/useAbletonState";
import { useDeckMap } from "../hooks/useDeckMap";

/**
 * Live combo VU meter — bouncing twin bars showing the sum of the
 * currently-playing deck-column output meters. Subscribed at the bridge
 * layer via /live/track/start_listen/output_meter_level on each deck
 * column; the FE just renders whatever bridge state reports.
 *
 * "Live combo" rather than Ableton's master because Live's session also
 * routes effect returns and other tracks through master — and we don't
 * care about those. Summing the 5 deck columns gives us the level of
 * the live-remixing combo specifically.
 */
export function VuMeter() {
  const ableton = useAbletonState();
  const deckMap = useDeckMap();

  // Sum the deck-column meter levels. Each is 0-1; sum can exceed 1 when
  // multiple stems are playing — that's the honest signal of "loudness".
  const level = useMemo(() => {
    const columns = deckMap.data?.columns;
    const meters = ableton.track_meters ?? {};
    if (!columns) return 0;
    let sum = 0;
    for (const trackIdx of Object.values(columns)) {
      const m = meters[String(trackIdx)] ?? 0;
      sum += m;
    }
    // Clamp + smooth curve: square-root keeps quiet stems visible.
    return Math.min(1, Math.sqrt(sum / 1.5));
  }, [deckMap.data, ableton.track_meters]);

  return (
    <div
      className="flex items-center gap-1.5 px-2 border-l border-neutral-800 h-10 pt-1.5"
      title="Combo output level — sum of the playing deck-column meters. Subscribed via AbletonOSC."
    >
      <span className="text-neutral-500 text-[10px] uppercase tracking-wider">
        Lvl
      </span>
      <div
        className="relative w-16 h-3 rounded-full bg-neutral-900 overflow-hidden"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={level}
      >
        {/* fill bar: emerald → amber → rose as level rises */}
        <div
          className="absolute inset-y-0 left-0 transition-[width] duration-100 ease-out"
          style={{
            width: `${Math.round(level * 100)}%`,
            background:
              level < 0.55
                ? "rgb(52, 211, 153)" // emerald-400
                : level < 0.85
                ? "rgb(251, 191, 36)" // amber-400
                : "rgb(251, 113, 133)", // rose-400
          }}
        />
        {/* "0dB" tick at ~85% — visual reference for hot signal */}
        <div className="absolute inset-y-0 left-[85%] w-px bg-neutral-700" />
      </div>
    </div>
  );
}
